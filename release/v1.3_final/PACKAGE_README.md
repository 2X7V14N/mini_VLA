# Mini-VLA V1.3 Final Package

This package contains the final V1.3 conservative fine-tuning checkpoint and
the exact configuration, official evaluation standard, machine-readable
results, failure records, and release notes.

The model achieved **90.56% (1630/1800)** on the official fixed comparison
set: `seed=20261105`, 200 scenes, 9 instructions per scene. The checkpoint is
for `LanguageConditionedMiniVLA3Color` with four-dimensional state input
`[x, y, holding, gripper_open]` and four-step `(dx, dy, dgrip)` action chunks.

The raw checkpoint is intentionally included in this package even though the
repository's normal `.gitignore` excludes standalone training weights.

See `V1.3_RELEASE.md` for the technical report and `V1.3_EVAL_STANDARD_OFFICIAL.md`
for the scoring contract.
