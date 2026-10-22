"""Unit tests for the assistant-only data collator."""

from src.train.collator import AssistantOnlyDataCollator


def test_collator_masking():
    # Response template token IDs: [10, 20]
    response_template = [10, 20]
    collator = AssistantOnlyDataCollator(
        response_template_ids=response_template,
        pad_token_id=0,
        ignore_index=-100,
    )

    # Sequence: prompt [1, 2, 3], template [10, 20], completion [99, 100]
    input_ids = [1, 2, 3, 10, 20, 99, 100]
    labels = collator.mask_labels_for_sequence(input_ids)

    # Tokens 1, 2, 3, 10, 20 should be masked with -100
    assert labels[:5] == [-100, -100, -100, -100, -100]
    # Tokens 99, 100 should remain intact
    assert labels[5:] == [99, 100]


def test_collator_missing_template():
    response_template = [10, 20]
    collator = AssistantOnlyDataCollator(
        response_template_ids=response_template,
        pad_token_id=0,
        ignore_index=-100,
    )

    # Sequence without template
    input_ids = [1, 2, 3, 4, 5]
    labels = collator.mask_labels_for_sequence(input_ids)

    # Everything masked
    assert labels == [-100, -100, -100, -100, -100]


def test_collator_batch_padding():
    response_template = [10, 20]
    collator = AssistantOnlyDataCollator(
        response_template_ids=response_template,
        pad_token_id=0,
        ignore_index=-100,
    )

    item1 = {"input_ids": [1, 10, 20, 50]}  # len 4
    item2 = {"input_ids": [2, 3, 4, 10, 20, 50, 60]}  # len 7

    batch = collator([item1, item2])

    input_ids = batch["input_ids"]
    labels = batch["labels"]
    attention_mask = batch["attention_mask"]

    # Both padded to max length 7
    assert input_ids.shape == (2, 7)
    assert labels.shape == (2, 7)
    assert attention_mask.shape == (2, 7)

    # Item 1 padded with 0 in input_ids and -100 in labels
    assert input_ids[0, 4:].tolist() == [0, 0, 0]
    assert labels[0, 4:].tolist() == [-100, -100, -100]
    assert attention_mask[0, 4:].tolist() == [0, 0, 0]

    # Target completion tokens unmasked
    assert labels[0, 3].item() == 50
    assert labels[1, 5:].tolist() == [50, 60]
