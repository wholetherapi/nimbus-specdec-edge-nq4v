"""Target-model verification of drafted token blocks."""
from __future__ import annotations

import torch


class Verifier:
    """Accepts a prefix of drafted tokens and rejects the remainder."""

    def __init__(self, target_model):
        self.target_model = target_model

    @torch.no_grad()
    def verify(self, prefix: torch.Tensor, draft: torch.Tensor) -> tuple[torch.Tensor, int]:
        """Return the accepted token sequence and the number of accepted drafts."""
        candidate = torch.cat([prefix, draft], dim=-1)
        logits = self.target_model(candidate).logits
        accepted = 0
        for position in range(draft.shape[1]):
            target_token = logits[:, prefix.shape[1] + position - 1, :].argmax(dim=-1)
            if torch.equal(target_token, draft[:, position]):
                accepted += 1
            else:
                break
        if accepted == 0:
            replacement = logits[:, prefix.shape[1] - 1, :].argmax(dim=-1, keepdim=True)
            return replacement, 0
        return draft[:, :accepted], accepted
