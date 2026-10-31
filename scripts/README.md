# Script Index

Version-specific executable scripts live in version folders. Run them from the
repository root so relative data and checkpoint paths resolve consistently.

| Folder | Contents |
| --- | --- |
| `v1/` | V1 model training, rollout, language evaluation, and baseline check |
| `v1_1/` | V1.1 environment, expert/data checks, recording, training, evaluation, and demo |
| `v1_2/` | V1.2 recording, correction data, training, migration, and evaluation |
| `v1_3/` | V1.3 scene/data checks, recording, training, migration, corrections, and `official` evaluation |
| `v1_5a/` | V1.5 moving-block/moving-zone data, training, evaluation, and GUI demo |

The `scripts/` root contains only this index. Tests live under `tests/`; version
plans and acceptance records live under `docs/`. Reusable Python modules remain
under `src/mini_vla/` and are imported by scripts and tests.

Example:

```powershell
python scripts\v1_3\eval_v13_closed_loop.py --ckpt checkpoints\v1.3_final.pt --scenes 200 --ablation-scenes 0 --results-log reports\v1.3_official_results.json
```

The final interactive entry point is:

```powershell
python scripts\v1_5a\demo_v15e.py --checkpoint checkpoints\v1.5e.pt
```
