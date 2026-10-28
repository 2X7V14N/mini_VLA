import numpy as np
import torch

from mini_vla.data.dataset_v13 import (
    V13BalancedMixtureBatchSampler,
    V13GroupedDataset,
    V13TickBatchSampler,
)
from mini_vla.data.grasp_place_expert_v13 import valid_combo_mask_v13
from mini_vla.models.text import SimpleCharTokenizer


def test_v13_dataset_returns_one_observation_with_nine_labels(tmp_path):
    split_dir = tmp_path / "train"
    split_dir.mkdir()

    prompts = [
        "把红色方块放到红色放置区",
        "把红色方块放到蓝色放置区",
        "把红色方块放到黄色放置区",
        "把蓝色方块放到红色放置区",
        "把蓝色方块放到蓝色放置区",
        "把蓝色方块放到黄色放置区",
        "把黄色方块放到红色放置区",
        "把黄色方块放到蓝色放置区",
        "把黄色方块放到黄色放置区",
    ]
    vocab = sorted(set("".join(prompts)))
    tokenizer = SimpleCharTokenizer(vocab, max_length=16)

    np.savez_compressed(
        split_dir / "ep_0000.npz",
        image=np.zeros((9, 64, 64, 3), dtype=np.uint8),
        instruction=np.asarray(prompts),
        robot_state=np.zeros((9, 4), dtype=np.float32),
        action_chunk=np.zeros((9, 4, 3), dtype=np.float32),
        valid_combo_mask=np.ones(9, dtype=bool),
        combo_index=np.arange(9, dtype=np.int32),
    )

    dataset = V13GroupedDataset(tmp_path, "train", tokenizer)
    item = dataset[0]

    assert len(dataset) == 1
    assert item["image"].shape == (3, 64, 64)
    assert item["token_ids"].shape == (9, 16)
    assert item["robot_state"].shape == (4,)
    assert item["action_chunk"].shape == (9, 4, 3)
    assert torch.equal(item["valid_mask"], torch.ones(9, dtype=torch.bool))
    assert not torch.any(item["token_ids"] == 1)


def test_v13_valid_combo_mask_preserves_original_nine_label_rule():
    assert valid_combo_mask_v13(False, "red", "yellow").tolist() == [True] * 9
    assert valid_combo_mask_v13(True, "red", "yellow").tolist() == [True] * 9


def test_v13_batch_sampler_keeps_ticks_as_units():
    sampler = V13TickBatchSampler(n_ticks=10, ticks_per_batch=3, seed=7)
    batches = list(sampler)

    flattened = [tick for batch in batches for tick in batch]
    assert sorted(flattened) == list(range(10))
    assert all(len(batch) <= 3 for batch in batches)


def test_v13_balanced_sampler_draws_equally_from_all_source_phase_combo_groups():
    class DatasetStub:
        balance_buckets = [bucket for bucket in range(72) for _ in range(bucket % 3 + 1)]

        def __len__(self):
            return len(self.balance_buckets)

    sampler = V13BalancedMixtureBatchSampler(
        DatasetStub(), ticks_per_batch=8, samples_per_bucket=2, seed=19
    )
    batches = list(sampler)
    sampled = [index for batch in batches for index in batch]
    counts = [0] * 72
    for index in sampled:
        counts[DatasetStub.balance_buckets[index]] += 1

    assert len(sampled) == 72 * 2
    assert counts == [2] * 72


def test_v13_balanced_sampler_can_emphasize_correction_open_phase():
    class DatasetStub:
        balance_buckets = [bucket for bucket in range(72) for _ in range(2)]

        def __len__(self):
            return len(self.balance_buckets)

    sampler = V13BalancedMixtureBatchSampler(
        DatasetStub(), ticks_per_batch=8, samples_per_bucket=2,
        open_correction_multiplier=2, seed=19,
    )
    sampled = [index for batch in sampler for index in batch]
    counts = [0] * 72
    for index in sampled:
        counts[DatasetStub.balance_buckets[index]] += 1

    assert len(sampled) == 81 * 2
    assert counts[:63] == [2] * 63
    assert counts[63:] == [4] * 9


def test_v13_balanced_sampler_can_emphasize_all_yellow_combinations():
    class DatasetStub:
        balance_buckets = [bucket for bucket in range(72) for _ in range(2)]

        def __len__(self):
            return len(self.balance_buckets)

    sampler = V13BalancedMixtureBatchSampler(
        DatasetStub(), ticks_per_batch=8, samples_per_bucket=2,
        yellow_combo_multiplier=2, seed=19,
    )
    sampled = [index for batch in sampler for index in batch]
    counts = [0] * 72
    for index in sampled:
        counts[DatasetStub.balance_buckets[index]] += 1

    assert len(sampled) == 112 * 2
    for bucket in range(72):
        expected = 4 if bucket % 9 in (2, 5, 6, 7, 8) else 2
        assert counts[bucket] == expected


def test_v13_balanced_sampler_supports_seventy_five_twenty_five_source_mix():
    class DatasetStub:
        balance_buckets = [bucket for bucket in range(72) for _ in range(2)]

        def __len__(self):
            return len(self.balance_buckets)

    sampler = V13BalancedMixtureBatchSampler(
        DatasetStub(), ticks_per_batch=16, samples_per_bucket=16, seed=19,
        old_replay_multiplier=3, correction_multiplier=1,
    )
    sampled = [index for batch in sampler for index in batch]
    old = sum(DatasetStub.balance_buckets[index] < 36 for index in sampled)
    correction = sum(DatasetStub.balance_buckets[index] >= 36 for index in sampled)
    assert (old, correction) == (1728, 576)
