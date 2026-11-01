# Optional pretrained reference-mask utility

The generated masks are stored only in `captured/masks_256/`, with their model,
source revision, prompts, hashes and limitations in `captured/mask_provenance.json`.
View `captured/reference_comparison.png`. These are AI-generated references,
not manual ground truth or official scoring labels.

This utility is separate from the assignment's scratch-trained segmentation
pipeline. It uses the official pretrained EfficientSAM-Ti model with
assistant-selected boxes/points recorded in `prompts.json`. No photos are uploaded.

To reproduce in a fresh output folder:

```powershell
python reference_masks/prepare_efficientsam.py
python reference_masks/generate_prompted_reference.py --out tmp/reference_masks_reproduction
```

Preparation downloads the exact revision and verifies each source/weight SHA-256
recorded in the captured provenance. Source, Apache license and weights are cached
under ignored `tmp/reference_models/efficient_sam_ti/`. Inference needs PyTorch,
NumPy and OpenCV from the project requirements. It loads the checkpoint with
`weights_only=True`. The unused U2-Net generator, rejected outputs, and duplicate
reference images were removed during repository cleanup.
