from scripts.train import PairedBatchSampler


def test_paired_batch_keeps_pairs_together():
    batches = list(PairedBatchSampler(n_samples=12, pairs_per_batch=2, seed=0))

    assert all(
        batch[offset] % 2 == 0
        and batch[offset + 1] == batch[offset] + 1
        for batch in batches
        for offset in range(0, len(batch), 2)
    )

    flattened = [index for batch in batches for index in batch]
    assert sorted(flattened) == list(range(12))
