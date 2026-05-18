from __future__ import annotations

from typing import Any

import torch


class PrefixSuffixCollator:
    def __init__(self, pad_id: int) -> None:
        self.pad_id = pad_id

    def __call__(self, batch: list[dict[str, Any]]) -> dict[str, Any]:
        prefix_tokens = torch.tensor([item["prefix_tokens"] for item in batch], dtype=torch.long)
        suffix_tokens = torch.tensor([item["suffix_tokens"] for item in batch], dtype=torch.long)
        result: dict[str, Any] = {
            "prefix_tokens": prefix_tokens,
            "suffix_tokens": suffix_tokens,
            "prefix_padding_mask": prefix_tokens.eq(self.pad_id),
            "suffix_padding_mask": suffix_tokens.eq(self.pad_id),
            "metadata": [item.get("metadata") for item in batch],
            "prefix_text": [item.get("prefix_text", "") for item in batch],
            "suffix_text": [item.get("suffix_text", "") for item in batch],
        }
        # SAT-specific clause tensors (Phase 2 loss).
        if "clause_variables" in batch[0]:
            result["clause_variables"] = torch.tensor(
                [item["clause_variables"] for item in batch], dtype=torch.long,
            )
            result["clause_polarities"] = torch.tensor(
                [item["clause_polarities"] for item in batch], dtype=torch.long,
            )
        return result

