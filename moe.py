import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class ScaleAwareRouter(nn.Module):
    """Router that conditions on scale information for improved routing in MoVEAR"""
    
    def __init__(self, input_dim, num_experts, num_scales=10, top_k=2, jitter=0.01, temperature=1.0):
        super().__init__()
        self.input_dim = input_dim
        self.num_experts = num_experts
        self.top_k = top_k
        self.jitter = jitter
        self.temperature = temperature
        
        # Simple scale embedding - optimized for speed
        self.scale_embeddings = nn.Parameter(torch.randn(num_scales, input_dim) * 0.02)
        
        # Standard router - fast implementation
        self.router = nn.Linear(input_dim, num_experts)
        
    def forward(self, x, scale_idx=None, training=True):
        batch_size, seq_len = x.shape[0], x.shape[1]
        
        # Apply scale conditioning if provided
        if scale_idx is not None and scale_idx >= 0:
            # Apply scale-specific embedding
            scale_emb = self.scale_embeddings[scale_idx]
            # Efficient broadcasting to avoid creating new tensors
            x = x + scale_emb.view(1, 1, -1)
            
        # Get routing logits
        router_logits = self.router(x)  # [batch_size, seq_len, num_experts]
        
        # Add noise during training for exploration
        if training and self.jitter > 0:
            router_noise = torch.randn_like(router_logits) * self.jitter
            router_logits = router_logits + router_noise
            
        # Apply temperature
        if self.temperature != 1.0:
            router_logits = router_logits / self.temperature
        
        # Get top-k routing weights and indices
        routing_weights, routing_indices = torch.topk(router_logits, self.top_k, dim=-1)
        routing_weights = F.softmax(routing_weights, dim=-1)
        
        return routing_weights, routing_indices


class MoEFFN(nn.Module):
    """Mixture of Experts FFN replacement for VAR model with scale awareness."""
    
    def __init__(self, in_features, hidden_features=None, out_features=None, num_experts=8, top_k=2, 
                 drop=0., fused_if_available=True, router_type="softmax", 
                 router_jitter=0.01, router_temperature=1.0):
        super().__init__()
        self.fused_mlp_func = None  # Don't use fused MLP for MoE
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.in_features = in_features
        self.hidden_features = hidden_features
        self.out_features = out_features
        self.num_experts = num_experts
        self.top_k = top_k
        self.router_type = router_type
        self.router_jitter = router_jitter
        self.router_temperature = router_temperature
        
        # Create experts
        self.experts = nn.ModuleList([
            FFNExpert(in_features, hidden_features, out_features, drop)
            for _ in range(num_experts)
        ])
        
        # Create scale-aware router
        self.num_scales = 10  # Default value, can be configured externally
        self.router = ScaleAwareRouter(
            input_dim=in_features,
            num_experts=num_experts,
            num_scales=self.num_scales,
            top_k=top_k,
            jitter=router_jitter,
            temperature=router_temperature
        )
        
        # For tracking load balancing
        self.register_buffer('expert_counts', torch.zeros(num_experts))
        self.total_tokens = 0
        self.balance_loss = 0.0

    def forward(self, x, scale_idx=None):
        # Get routing weights and indices
        routing_weights, routing_indices = self.router(x, scale_idx, self.training)
        batch_size, seq_len = x.shape[0], x.shape[1]
        
        # Create output tensor
        outputs = torch.zeros_like(x)
        
        # Track expert counts for load balancing
        if self.training:
            # Compute expert assignment distribution for load balancing
            expert_mask = torch.zeros(batch_size * seq_len, self.num_experts, device=x.device)
            for k in range(self.top_k):
                flat_indices = routing_indices[:, :, k].reshape(-1)
                expert_mask.scatter_add_(
                    1, 
                    flat_indices.unsqueeze(-1),
                    routing_weights[:, :, k].reshape(-1, 1)
                )
            
            # Calculate load
            expert_load = expert_mask.mean(0)
            
            # Compute balance loss (simplified and efficient)
            # Penalize deviation from uniform distribution
            balance_factor = (expert_load - 1.0/self.num_experts).pow(2).sum() * self.num_experts
            self.balance_loss = balance_factor
        
        # More efficient implementation - only process tokens routed to each expert
        for expert_idx in range(self.num_experts):
            # Find which tokens are routed to this expert at any k position
            expert_mask = torch.zeros((batch_size, seq_len), dtype=torch.bool, device=x.device)
            expert_scale = torch.zeros((batch_size, seq_len, 1), device=x.device)
            
            for k in range(self.top_k):
                # Find locations where this expert is selected
                selected_locations = (routing_indices[:, :, k] == expert_idx)
                expert_mask = expert_mask | selected_locations
                expert_scale += selected_locations.unsqueeze(-1) * routing_weights[:, :, k].unsqueeze(-1)
            
            # Skip if no tokens are routed to this expert
            if not expert_mask.any():
                continue
                
            # Only process tokens that use this expert
            selected_indices = expert_mask.nonzero(as_tuple=True)
            selected_batch_indices, selected_seq_indices = selected_indices
            
            # Get the input tokens for this expert
            expert_input = x[selected_batch_indices, selected_seq_indices]
            
            # Process with expert
            expert_output = self.experts[expert_idx](expert_input)
            
            # Place outputs back in the right positions with proper scaling
            outputs[selected_batch_indices, selected_seq_indices] += expert_output * expert_scale[expert_mask]
        
        return outputs

    def extra_repr(self) -> str:
        return f'in_features={self.in_features}, hidden_features={self.hidden_features}, out_features={self.out_features}, num_experts={self.num_experts}, top_k={self.top_k}'


class FFNExpert(nn.Module):
    """Individual expert in the MoE layer, similar to FFN in VAR."""
    
    def __init__(self, in_features, hidden_features, out_features, drop=0.):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU(approximate='tanh')
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop, inplace=True) if drop > 0 else nn.Identity()
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


def compute_theoretical_losses(scale_reps, kappa=0.9, holder_const=2.0, jacobi_eta=1.0):
    """
    Compute unified theoretical loss combining Lyapunov stability, Hölder continuity, and Jacobi field.
    Optimized for minimal computational overhead.
    
    Args:
        scale_reps: List of representations at different scales [batch_size, hidden_dim]
        kappa: Stability parameter (< 1 for contraction)
        holder_const: Hölder continuity constant
        jacobi_eta: Maximum allowed expansion factor for neighborhood preservation
    """
    if len(scale_reps) < 2:
        # Need at least two scales
        return torch.tensor(0.0, device=scale_reps[0].device), 0.0, 0.0, 0.0
    
    device = scale_reps[0].device
    lyapunov_loss = 0.0
    holder_loss = 0.0
    jacobi_loss = 0.0
    
    # Process only a subset of transitions (every other transition) to reduce computation
    indices = range(0, len(scale_reps)-1, 2)
    
    for i in indices:
        z_i = scale_reps[i]
        z_i1 = scale_reps[i+1]
        
        # 1. Lyapunov Stability: ensures contraction between scales
        # Calculate energy (squared norm) efficiently
        v_zi = torch.sum(z_i ** 2, dim=1)  # [batch_size]
        v_zi1 = torch.sum(z_i1 ** 2, dim=1)  # [batch_size]
        
        # Measure violation of stability condition
        lyapunov_term = torch.clamp(v_zi1 - kappa**2 * v_zi, min=0.0)
        denominator = torch.mean(v_zi) + 1e-5  # Increase epsilon for stability
        lyapunov_loss += torch.clamp(lyapunov_term.mean() / denominator, max=1000.0)  # Prevent extreme values
        
        # 2. Hölder Continuity: ensures smooth transitions between scales
        # Calculate distance between representations
        diff_norm = torch.norm(z_i1 - z_i, dim=1)
        holder_term = torch.clamp(diff_norm - holder_const, min=0.0)
        holder_loss += holder_term.mean() / (torch.norm(z_i, dim=1).mean() + 1e-6)
        
        # 3. Jacobi Field Attractivity: ensures neighborhood preservation
        batch_size = z_i.shape[0]
        if batch_size > 1:
            # Sample at most 4 pairs for efficiency
            idx = torch.randperm(batch_size, device=device)[:min(4, batch_size)]
            if len(idx) > 1:
                # Compute all pairwise distances in one go
                z_i_sample = z_i[idx]
                z_i1_sample = z_i1[idx]
                
                # Calculate pairwise distances for all pairs
                jacobi_terms = []
                for m in range(len(idx)-1):
                    for n in range(m+1, len(idx)):
                        dist_i = torch.norm(z_i_sample[m] - z_i_sample[n]) + 1e-5  # Increase epsilon
                        dist_i1 = torch.norm(z_i1_sample[m] - z_i1_sample[n])
                        # Calculate expansion ratio
                        expansion = torch.clamp(dist_i1 / dist_i, min=0.0, max=100.0)
                        # Penalize excessive expansion
                        jacobi_terms.append(torch.clamp(expansion - jacobi_eta, min=0.0))
                
                if jacobi_terms:
                    jacobi_loss += torch.stack(jacobi_terms).mean()
    
    # Normalize by number of transitions processed
    num_transitions = len(indices)
    if num_transitions > 0:
        lyapunov_loss /= num_transitions
        holder_loss /= num_transitions
        jacobi_loss /= num_transitions
    
    # Combined theoretical loss (simplify to reduce operations)
    total_loss = lyapunov_loss + holder_loss + jacobi_loss
    
    return total_loss, lyapunov_loss.item(), holder_loss.item(), jacobi_loss.item()


def apply_moe_to_var(var_model, layer_index=-1, num_experts=8, top_k=2, router_type="softmax",
                     router_jitter=0.01, router_temperature=1.0):
    """
    Replace the FFN in a specific transformer block with a MoE layer.
    """
    # Ensure layer_index is an integer
    layer_index = int(layer_index)
    
    # Handle negative indexing
    total_layers = len(var_model.blocks)
    if layer_index < 0:
        layer_index = total_layers + layer_index
    
    # Validate layer index
    if not 0 <= layer_index < total_layers:
        raise ValueError(f"Layer index {layer_index} out of range (0 to {total_layers-1})")
    
    print(f"[MoE] Replacing FFN in layer {layer_index} (of {total_layers} total layers)")
    
    # Get the target block
    block = var_model.blocks[layer_index]
    
    # Save the original FFN state for initialization
    original_ffn = block.ffn
    
    # Get FFN parameters
    in_features = original_ffn.fc1.in_features
    hidden_features = original_ffn.fc1.out_features
    out_features = original_ffn.fc2.out_features
    drop = original_ffn.drop.p if hasattr(original_ffn.drop, 'p') else 0.0
    
    # Determine the device of the original FFN
    device = next(original_ffn.parameters()).device
    print(f"[MoE] Creating MoE layer on device: {device}")
    
    # Create MoE layer on the same device
    moe_ffn = MoEFFN(
        in_features=in_features,
        hidden_features=hidden_features,
        out_features=out_features,
        num_experts=num_experts,
        top_k=top_k,
        drop=drop,
        router_type=router_type,
        router_jitter=router_jitter,
        router_temperature=router_temperature
    ).to(device)
    
    # Initialize all experts with the weights from the original FFN
    print(f"[MoE] Copying weights from original FFN to all {num_experts} experts")
    with torch.no_grad():
        for expert in moe_ffn.experts:
            # Copy fc1 weights and bias
            expert.fc1.weight.copy_(original_ffn.fc1.weight.data)
            expert.fc1.bias.copy_(original_ffn.fc1.bias.data)
            
            # Copy fc2 weights and bias
            expert.fc2.weight.copy_(original_ffn.fc2.weight.data)
            expert.fc2.bias.copy_(original_ffn.fc2.bias.data)
    
    # Replace FFN with MoE FFN
    block.ffn = moe_ffn
    
    # Store the scale_idx in the model for later reference by the router
    var_model.current_scale_idx = -1
    
    # Freeze all parameters except MoE
    for name, param in var_model.named_parameters():
        if 'blocks.' + str(layer_index) + '.ffn' in name:
            param.requires_grad = True
        else:
            param.requires_grad = False
            
    # Double-check all parameters are on the right device
    for name, param in var_model.named_parameters():
        if param.device != device:
            print(f"[MoE] Moving parameter {name} from {param.device} to {device}")
            param.data = param.data.to(device)
    
    return var_model


def print_trainable_parameters(model):
    """Print the number of trainable parameters in the model."""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Percentage of trainable parameters: {trainable_params / total_params * 100:.2f}%")
    
    return trainable_params, total_params
