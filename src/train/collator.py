"""Data collator masking user/system prompt tokens with -100 so loss is computed strictly on assistant tokens."""

from __future__ import annotations

from typing import Any

import torch


class AssistantOnlyDataCollator:
    """Masks all tokens up to and including the response template with -100 in labels.

    Ensures that cross-entropy loss is computed exclusively on the assistant's
    target JSON output, not on system prompts or user chunks.
    """

    def __init__(
        self,
        response_template_ids: list[int],
        pad_token_id: int = 0,
        ignore_index: int = -100,
    ) -> None:
        self.response_template_ids = response_template_ids
        self.pad_token_id = pad_token_id
        self.ignore_index = ignore_index

    def _find_subsequence(self, sequence: list[int], sub: list[int]) -> int:
        """Find the start index of sub in sequence, or -1 if not found."""
        sub_len = len(sub)
        for i in range(len(sequence) - sub_len + 1):
            if sequence[i : i + sub_len] == sub:
                return i
        return -1

    def mask_labels_for_sequence(
        self,
        input_ids: list[int],
    ) -> list[int]:
        """Return a copy of input_ids where tokens before the response template are masked with -100."""
        labels = list(input_ids)
        sub_idx = self._find_subsequence(input_ids, self.response_template_ids)

        if sub_idx != -1:
            # Mask everything up to the end of the response template
            mask_end = sub_idx + len(self.response_template_ids)
            for i in range(mask_end):
                labels[i] = self.ignore_index
        else:
            # If template not found, mask entire sequence to prevent training on unformatted data
            labels = [self.ignore_index] * len(labels)

        return labels

    def __call__(self, batch: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        """Collate a batch of tokenized sequences into padded PyTorch tensors."""
        max_len = max(len(item["input_ids"]) for item in batch)

        padded_input_ids = []
        padded_attention_masks = []
        padded_labels = []

        for item in batch:
            input_ids = list(item["input_ids"])
            seq_len = len(input_ids)
            labels = self.mask_labels_for_sequence(input_ids)

            pad_len = max_len - seq_len
            padded_input_ids.append(input_ids + [self.pad_token_id] * pad_len)
            padded_attention_masks.append([1] * seq_len + [0] * pad_len)
            padded_labels.append(labels + [self.ignore_index] * pad_len)

        return {
            "input_ids": torch.tensor(padded_input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(padded_attention_masks, dtype=torch.long),
            "labels": torch.tensor(padded_labels, dtype=torch.long),
        }
