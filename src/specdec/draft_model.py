"""Small draft model used to propose token candidates."""
from __future__ import annotations

import torch
from torch import nn


class DraftModel(nn.Module):
    """Autoregressive draft model with a configurable depth cap."""

    def __init__(self, vocab_size: int, hidden_size: int = 256, layers: int = 4):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, hidden_size)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size, nhead=8, batch_first=True, dim_feedforward=1024
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=layers)
        self.head = nn.Linear(hidden_size, vocab_size, bias=False)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor | None = None):
        hidden = self.encoder(self.embed(input_ids), src_key_padding_mask=attention_mask)
        return self.head(hidden)

    @torch.no_grad()
    def propose(self, prefix: torch.Tensor, depth: int) -> torch.Tensor:
        """Greedily propose ``depth`` tokens from a prefix of shape (B, T)."""
        tokens = prefix
        proposals = []
        for _ in range(depth):
            logits = self.forward(tokens)
            next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
            proposals.append(next_token)
            tokens = torch.cat([tokens, next_token], dim=-1)
        return torch.cat(proposals, dim=-1)
