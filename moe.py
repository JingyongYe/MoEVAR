import torch
import torch.nn as nn
import torch.nn.functional as F


class MoEFFN(nn.Module):
    """Mixture of Experts FFN replacement for VAR model."""
    
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
        
        # Create experts (each one similar to original FFN)
        self.experts = nn.ModuleList([
            FFNExpert(in_features, hidden_features, out_features, drop)
            for _ in range(num_experts)
        ])
        
        # Create router
        self.router = nn.Linear(in_features, num_experts, bias=False)
        nn.init.normal_(self.router.weight, mean=0.0, std=0.01)

    def forward(self, x):
        # x shape: [batch_size, sequence_length, hidden_size]
        batch_size, seq_len, _ = x.shape
        flat_x = x.reshape(-1, self.in_features)  # [batch_size * seq_len, hidden_size]
        
        # Calculate routing probabilities
        router_logits = self.router(flat_x)  # [batch_size * seq_len, num_experts]
        
        # Add noise during training for load balancing
        if self.training and self.router_jitter > 0:
            router_logits += torch.randn_like(router_logits) * self.router_jitter
            
        if self.router_type == "softmax":
            # Scale logits by temperature
            router_logits = router_logits / self.router_temperature
            # Get top-k indices and routing probabilities
            routing_weights, routing_indices = torch.topk(router_logits, self.top_k, dim=-1)
            routing_weights = F.softmax(routing_weights, dim=-1)
        else:  # "noisy_gate" - Switch Transformer style
            routing_weights, routing_indices = torch.topk(router_logits, self.top_k, dim=-1)
            routing_weights = F.softmax(routing_weights, dim=-1)
        
        # Create output tensor
        output = torch.zeros((batch_size * seq_len, self.out_features), device=x.device)
        
        # For each expert, compute output and add weighted contribution
        for k in range(self.top_k):
            # Get the current expert indices for all tokens
            expert_idx = routing_indices[:, k]  # [batch * seq]
            weight = routing_weights[:, k].unsqueeze(-1)  # [batch * seq, 1]
            
            # Process tokens by experts (batched for efficiency)
            for e in range(self.num_experts):
                # Find where expert_idx == e
                expert_mask = (expert_idx == e)
                if not expert_mask.any():
                    continue
                
                # Get inputs for this expert
                expert_input = flat_x[expert_mask]
                
                # Calculate output for this expert
                expert_output = self.experts[e](expert_input)
                
                # Add weighted output to the total output
                output[expert_mask] += expert_output * weight[expert_mask]
        
        # Reshape output back to original shape
        return output.reshape(batch_size, seq_len, -1)
    
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
