# Retained inference weights

- `final_model.pth`: original frozen 96-pixel baseline. Its existing report and
  official results remain available for comparison.
- `active_model.json`: pointer to the qualified 128-pixel channels-last release
  under `releases/`. That single release retains `model.pth`, its freeze sidecar
  and immutable qualification snapshot.

Prediction verifies release hashes and its qualified numerical runtime. Neither
checkpoint uses pretrained weights. Duplicate releases and generated validation
sweep checkpoints were removed; measured JSON evidence remains in `results/`.
Per-run best checkpoints needed for reproduction are retained under `experiments/`.
