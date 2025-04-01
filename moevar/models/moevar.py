import math
from functools import partial
from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM

from moellava.model.language_model.qwen.modeling_qwen import QWenLMHeadModel
from models.helpers import sample_with_top_k_top_p_, gumbel_softmax_with_rng
from models.vqvae import VQVAE, VectorQuantizer2


class MOEVAR(nn.Module):
    def __init__(
        self,
        vae_local: VQVAE,
        qwenmoe_model_name: str,
        num_classes=1000,
        depth=16,
        embed_dim=1024,
        num_heads=16,
        mlp_ratio=4.0,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.0,
        norm_eps=1e-6,
        patch_nums=(1, 2, 3, 4, 5, 6, 8, 10, 13, 16),
    ):
        super().__init__()
        # 0. Hyperparameters
        assert embed_dim % num_heads == 0
        self.vae_local = vae_local
        self.vocab_size = vae_local.vocab_size
        self.num_classes = num_classes
        self.patch_nums = patch_nums
        self.depth = depth
        self.embed_dim = embed_dim
        self.num_heads = num_heads

        # 1. Input embedding
        self.word_embed = nn.Linear(vae_local.Cvae, embed_dim)

        # 2. Class embedding
        self.class_emb = nn.Embedding(num_classes + 1, embed_dim)
        nn.init.trunc_normal_(self.class_emb.weight, std=0.02)

        # 3. Position embedding
        self.pos_embed = nn.Parameter(torch.zeros(1, sum(pn ** 2 for pn in patch_nums), embed_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        # 4. Backbone: QwenMoE as the language model
        self.qwenmoe = QWenLMHeadModel.from_pretrained(qwenmoe_model_name)

        # 5. Dropout
        self.dropout = nn.Dropout(drop_rate)

    def forward(
        self,
        label_B: torch.LongTensor,
        x_BLCv: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass for training.
        :param label_B: Class labels (B,)
        :param x_BLCv: Input tokens (B, L, Cvae)
        :param attention_mask: Attention mask (B, L)
        :return: Logits (B, L, vocab_size)
        """
        B, L, _ = x_BLCv.shape
        class_emb = self.class_emb(label_B).unsqueeze(1)  # (B, 1, embed_dim)
        x_BLC = self.word_embed(x_BLCv) + self.pos_embed[:, :L, :]  # (B, L, embed_dim)
        x_BLC = torch.cat([class_emb, x_BLC], dim=1)  # (B, L+1, embed_dim)

        x_BLC = self.dropout(x_BLC)
        outputs = self.qwenmoe(inputs_embeds=x_BLC, attention_mask=attention_mask)
        return outputs.logits

    @torch.no_grad()
    def generate(
        self,
        label_B: torch.LongTensor,
        max_length: int,
        top_k: int = 0,
        top_p: float = 0.9,
    ) -> torch.Tensor:
        """
        Autoregressive generation.
        :param label_B: Class labels (B,)
        :param max_length: Maximum sequence length
        :param top_k: Top-k sampling
        :param top_p: Top-p sampling
        :return: Generated tokens (B, max_length)
        """
        B = label_B.size(0)
        class_emb = self.class_emb(label_B).unsqueeze(1)  # (B, 1, embed_dim)
        generated = class_emb

        for _ in range(max_length):
            outputs = self.qwenmoe(inputs_embeds=generated)
            logits = outputs.logits[:, -1, :]  # (B, vocab_size)
            next_token = sample_with_top_k_top_p_(logits, top_k=top_k, top_p=top_p)
            next_emb = self.word_embed(self.vae_local.quantize.embedding(next_token))  # (B, 1, embed_dim)
            generated = torch.cat([generated, next_emb], dim=1)

        return generated
