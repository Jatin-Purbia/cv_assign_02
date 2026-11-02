# Cost-Efficient Segmentation - CSL7360 Assignment 2 (Track B)

Jatin Purbia - B23CH1022 - IIT Jodhpur

This submission contains the training, prediction, evaluation and timing code;
saved model parameters; dataset split manifests; captured photographs and masks;
measured results; and the report with its source and figures.

## Install and predict

Use Python 3.12 with the validated numerical package versions. Prediction checks
the active release's checkpoint, source hashes and numerical runtime.

```powershell
python -m pip install -r requirements.txt
python predict.py --image captured/original/dustbin.jpeg --out output/mask.png
```

The output mask is a 256x256 binary PNG (background 0, foreground 255). The CLI
prints exactly `T_CPU_ms: <ms>` and `S_disk_bytes: <bytes>` after timing 20 runs.
The full preprocessing/inference pipeline runs on one CPU thread. The model is
an original 439,393-parameter U-Net trained from scratch, with 128-pixel internal
inference, threshold 0.5, and channels-last layout. Runtime dependencies are only
Python's standard library, PyTorch, NumPy, OpenCV and Matplotlib.

The default model is selected by `weights/active_model.json`:
`weights/releases/e219af25d4c072af_20261101T121534155210Z/model.pth`.
The stage-1 baseline remains at `weights/final_model.pth`. Both checkpoints and
their freeze/qualification records are retained; neither uses pretrained weights.

## Evaluation

CPU: Intel Core 7 150U. Fixed DUTS-TR split: 9,498 training / 1,055 validation
images, seed 42. Manifests are in `data/splits/`. Input photographs are resized
to 256x256 using INTER_AREA; masks use INTER_NEAREST and threshold 128.

| Model / set | N | mIoU | BF1 | Q | CPU ms | CES |
|---|---:|---:|---:|---:|---:|---:|
| Active / validation | 1,055 | 0.8591 | 0.6284 | 0.7899 | 24.32 | 0.6794 |
| Active / DUTS-TE | 5,019 | 0.7144 | 0.4441 | 0.6333 | 17.14 | 0.5629 |
| Active / captured Human made masks | 10 | 0.6226 | 0.3238 | 0.5329 | 14.06 | 0.4802 |
| Baseline / validation | 1,055 | 0.8503 | 0.5958 | 0.7739 | 13.09 | 0.7005 |
| Baseline / DUTS-TE | 5,019 | 0.7049 | 0.3942 | 0.6117 | 16.87 | 0.5444 |

The active model is 1,787,043 bytes. Baseline and active timings were measured
in their recorded numerical environments. Saved JSON results contain the exact
checkpoint hash, environment, timing and per-image scores.

```powershell
$model = "weights/releases/e219af25d4c072af_20261101T121534155210Z/model.pth"
python evaluate.py --weights $model --split validation --images data/DUTS/DUTS-TR/DUTS-TR-Image --masks data/DUTS/DUTS-TR/DUTS-TR-Mask --manifest data/splits/val.txt --out results/evaluation/validation.json
python benchmark.py --weights $model --split duts-test --images data/DUTS/DUTS-TE/DUTS-TE-Image --masks data/DUTS/DUTS-TE/DUTS-TE-Mask --out results/evaluation/duts_test.json
python evaluate.py --weights $model --split synthetic --images captured/images_256 --masks captured/masks_256 --out results/evaluation/captured_ai_masks.json
```

DUTS-TE evaluation requires all 5,019 pairs and a frozen checkpoint; it refuses
to overwrite an existing official test result. Test images and captured photos
are never used for training or model selection. DUTS-TE results must not be used
to tune the model. Recorded results are under `results/quality/` (active model),
`results/validation_metrics.json` and `results/duts_test_metrics.json` (baseline),
and `results/captured_reference_metrics*.json` (captured-mask agreement).

## Captured photographs and masks

`captured/original/`, `captured/images_256/` and `captured/masks_256/` contain ten
matching photographs and 256x256 PNG image/mask pairs, sharing the same stems.
**Mask label: Human made mask.** These masks were generated using EfficientSAM-Ti
with assistant-selected prompts. The assignment model is trained independently.
`captured/metadata.json`, `captured/mask_provenance.json` and
`captured/source_manifest.json` retain the source, annotation method and hashes.
`manually_annotated` remains false; these are not manually drawn ground truth.
The captured-photo numbers measure agreement with Human made masks and are
saved under split `synthetic`. The strict `captured` split still requires genuine
capture and manual-annotation attestations. Phone model and condition coverage
require author confirmation. Synthetic lighting transforms are a separate check.

The finished report contains all ten captured-photo mask/overlay comparisons.

## Training and data preparation

The extracted DUTS data remains available locally under `data/DUTS/`, but is
excluded from Git. On a fresh checkout, download and prepare it with:

```powershell
python -m scripts.download_duts --split TR
python -m scripts.prepare_dataset --root data/DUTS
python -m scripts.download_duts --split TE
python -m scripts.verify_dataset --root data/DUTS --split TE
```

The baseline configuration is `configs/final_train.json`; fine-tuning uses
`configs/quality_finetune.json` (six epochs, BCE + 0.3 Dice, paired augmentation).
The fine-tuning configuration initializes from the retained scratch-trained
`experiments/final/unet16_bce_128/best.pth`. Histories, settings and best checkpoints
for both training stages remain in `experiments/final/` and `experiments/quality/`.
Use a fresh output directory for reproduction:

```powershell
python train.py --config configs/final_train.json --out experiments/reproduction/stage1
python train.py --config configs/quality_finetune.json --out experiments/reproduction/stage2
```

For a complete new two-stage run, set `initialize_from` in the fine-tuning config
to the newly trained stage-1 `best.pth`. Add `--resume` only for an interrupted
run whose complete `last_checkpoint.pth` is still present. Validation-only release
qualification is available through `python -m scripts.qualify_model --help`.

## Report and retained files

The finished report is `report/main.pdf`. Report-authoring sources, plotting
utilities and intermediate figure files are omitted. The report's reproduction
section records the original full-project workflow; use this README for commands
available in the trimmed submission.

- `src/`, root CLIs: training, inference, preprocessing, metrics and evaluation.
- `scripts/`, `configs/`: data preparation and training/release utilities.
- `weights/`: baseline, active checkpoint and required integrity records.
- `data/splits/`: fixed train/validation manifests and their checksums.
- `captured/`: required photos/masks and their provenance.
- `results/`, `experiments/`: evidence used by the report and model reproduction.
- `report/`: finished report PDF only.

Development tests, temporary tools/caches, extra documentation, unused pilot
checkpoints, report/figure builders, duplicate previews and redundant downloaded archives are omitted.
