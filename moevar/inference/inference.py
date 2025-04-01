import torch
from models.moevar import MOEVAR

def infer_moevar(
    model: MOEVAR,
    label: int,
    max_length: int,
    device: torch.device,
    top_k: int = 0,
    top_p: float = 0.9,
):
    model.to(device)
    model.eval()
    label_tensor = torch.tensor([label], device=device)
    generated = model.generate(label_tensor, max_length=max_length, top_k=top_k, top_p=top_p)
    return generated
