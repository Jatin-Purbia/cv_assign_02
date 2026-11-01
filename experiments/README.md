# Retained experiments

Completed runs retain `run.json`, `history.csv`, `best.json`, `best.pth`,
`summary.json` and any recorded validation/sweep JSON observations. Pilot and
refinement evidence is preserved to explain the baseline selection.

`final/unet16_bce_128/best.pth` initializes quality fine-tuning. Its
`inference_96.pth` is the selected baseline export used by report/freezing tools.
`quality/unet16_dice_aug_recovered/best.pth` is the completed six-epoch candidate.
The active deployment export is separately retained under `weights/releases/`.

Completed optimizer checkpoints, redundant resolution/threshold weight exports,
worker logs/PIDs and abandoned runs were pruned. Historical JSON/CSV checkpoint
paths may describe pruned exports; their metrics and hashes remain original
observations. Sweep/qualification scripts can regenerate exports from retained
best checkpoints without retraining. Do not rerun official DUTS-TE evaluation to
select models. New training runs still save `last_checkpoint.pth` for exact resume;
those intermediate states are ignored by Git.
