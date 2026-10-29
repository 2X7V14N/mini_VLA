"""Repair success metadata in already-collected V1.5e rescue episodes."""

import argparse
import os
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    args = parser.parse_args()
    for temporary in args.root.glob("*.repair.npz"):
        temporary.unlink()
    paths = sorted(args.root.glob("ep_*.npz"))
    repaired = 0
    for path in paths:
        with np.load(path, allow_pickle=False) as data:
            fields = {key: data[key].copy() for key in data.files}
        fields["success"] = np.bool_(True)
        temporary = path.with_name(path.stem + ".repair.npz")
        np.savez_compressed(temporary, **fields)
        os.remove(path)
        os.replace(temporary, path)
        repaired += 1
    bad, frames, failures = [], 0, {}
    for path in paths:
        with np.load(path, allow_pickle=False) as data:
            count = len(data["image"])
            frames += count
            failure = str(data["source_failure"])
            failures[failure] = failures.get(failure, 0) + 1
            if not bool(data["success"]):
                bad.append(path.name)
            if data["image"].shape != (count, 64, 64, 3):
                bad.append(path.name + ":image")
            if data["action_chunk"].shape != (count, 4, 3):
                bad.append(path.name + ":action")
    print(f"repaired={repaired}; frames={frames}; failures={failures}; bad={len(bad)}")
    if bad:
        raise RuntimeError(f"bad files: {bad[:5]}")


if __name__ == "__main__":
    main()
