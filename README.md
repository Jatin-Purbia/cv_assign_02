# Cost-Efficient Segmentation - CSL7360 Assignment 2 (Track B)

Jatin Purbia - B23CH1022 - IIT Jodhpur

A 439,393-parameter U-Net trained from scratch for automatic salient-object
segmentation. `predict.py` uses the qualified active release: 128-pixel internal
resolution, threshold 0.5 and PyTorch channels-last layout. Inputs and output
masks are 256x256. All assignment runtime code uses only the standard library,
PyTorch, NumPy, OpenCV and Matplotlib; metrics are implemented from scratch.
The separate pretrained reference-mask utility is described below.

## Results

| Model / set | N | mIoU | BF1 | Q | CPU ms | CES |
|---|---:|---:|---:|---:|---:|---:|
| Active / validation | 1,055 | 0.8591 | 0.6284 | 0.7899 | 24.32 | 0.6794 |
| Active / DUTS-TE | 5,019 | 0.7144 | 0.4441 | 0.6333 | 17.14 | 0.5629 |
| Baseline / validation | 1,055 | 0.8503 | 0.5958 | 0.7739 | 13.09 | 0.7005 |
| Baseline / DUTS-TE | 5,019 | 0.7049 | 0.3942 | 0.6117 | 16.87 | 0.5444 |

CPU: Intel Core 7 150U, one thread. Split: 9,498 train / 1,055 validation, seed 42.
The active model is 1,787,043 bytes. Historical baseline and active timings used
different numerical environments; see [model quality](docs/MODEL_QUALITY.md).
The report source for the active model is [report/main.tex](report/main.tex)
(figures in `report/figures/`). `report/main.pdf` is still the old baseline PDF: no
LaTeX compiler is installed here, so compile `report/main.tex` (pdflatex, run twice,
or Overleaf) and inspect it before submitting. Release evidence is in
`results/quality/` and `weights/active_model.json`.

## Install, verify and predict

The active release was qualified with Python 3.12.4 and the numerical package
versions in `requirements-runtime.txt`. Its loader checks the runtime and hashes.
For a fresh compatible Python 3.12 environment:

```powershell
python -m pip install -r requirements-runtime.txt
python -m scripts.verify
python predict.py --image captured/original/dustbin.jpeg --out output/mask.png
```

Prediction saves a 0/255 PNG, warms up once and times 20 further runs, printing
exactly `T_CPU_ms: <ms>` and `S_disk_bytes: <bytes>`. Use `--weights` to select an
explicit experimental checkpoint instead of the active release.

Latest saved predictions: [captured/model_predictions/comparison.png](captured/model_predictions/comparison.png).
To generate a fresh folder without overwriting them:

```powershell
python -m scripts.predict_folder --images captured/original --out captured/prediction_runs
```

## Data and training

DUTS images are excluded from Git. To reconstruct the dataset:

```powershell
python -m scripts.download_duts --split TR
python -m scripts.prepare_dataset --root data/DUTS
python -m scripts.download_duts --split TE
python -m scripts.verify_dataset --root data/DUTS --split TE
```

Photos are directly resized with INTER_AREA; masks use INTER_NEAREST and threshold
128. DUTS-TE and captured photos are never used for training or model selection.

The original full training run is retained in `experiments/final/unet16_bce_128`.
Its best checkpoint initializes six further epochs at LR 0.0003 with BCE + 0.3 Dice
and paired geometry/color augmentation. The completed fine-tuning run is
`experiments/quality/unet16_dice_aug_recovered`. Reproduce fine-tuning in a fresh
run directory, preserving the recorded result:

```powershell
python train.py --config configs/quality_finetune.json --out experiments/quality/reproduction
python -m scripts.qualify_model --runs experiments/quality/reproduction --out results/quality_reproduction --activate
```

During an interrupted new run, add `--resume` with the same config/output. Completed
runs retain best weights and histories; optimizer checkpoints and derived sweep
weights were pruned. Full baseline experiments and evaluation instructions are in
[reproduction details](docs/REPRODUCIBILITY.md) and [experiment notes](docs/EXPERIMENTS.md).
Official DUTS-TE evaluation is already recorded; do not repeat it to tune parameters.

## Captured photos and masks

`captured/` follows the assignment layout: `original/` (photos as taken),
`images_256/` (INTER_AREA PNGs) and `masks_256/` (256x256 binary PNG masks,
object 255, background 0), all sharing one base name per photo. Latest
trained-model predictions are in `captured/model_predictions/`.

**Mask status.** The masks in `masks_256/` are AI reference masks from pretrained
EfficientSAM-Ti with assistant-selected prompts (see `mask_provenance.json`), not
hand-drawn. They are used as the ground truth for the captured-photo numbers, and the
report states this. On them the active model scores mIoU 0.623, BF1 0.324, Q 0.533,
CES 0.480 (`results/captured_reference_metrics.json`, split `synthetic`; per-photo
overlays in `results/captured_figures/`). To replace them with manual masks, draw each
with the polygon tool (click the boundary, Enter saves and overwrites that mask):

```powershell
foreach ($f in Get-ChildItem captured/original) { python -m scripts.annotate --image $f.FullName }
```

Then re-run the captured evaluation and overlays (commands in the report's
reproducibility section) and update the report note, `captured/README.md` and
`mask_provenance.json`. `captured/metadata.json` is filled in honestly
(`manually_annotated: false`; phone model and condition tags are to be confirmed by
you). The strict `scripts.audit --submission` still reports blockers that are
deliberate: the manual-annotation attestation, the baseline-versus-active release
mismatch and the `captured`-split metrics, so `scripts.package_submission` stays
blocked. The photos are never used for training or selection.
See [captured/README.md](captured/README.md) and
[reference_masks/README.md](reference_masks/README.md).

## Repository layout

- `src/`, `scripts/`, `tests/`: training, inference, metrics, evaluation and checks.
- `configs/`, `data/splits/`: reproducible settings and fixed manifests.
- `weights/`: baseline and the single active release with freeze/qualification evidence.
- `experiments/`, `results/`: best checkpoints, histories and measured evidence.
- `captured/`: personal photos, AI reference masks and saved model predictions.
- `reference_masks/`: optional pretrained reference-generation code; no downloaded weights.
- `report/`, `docs/`: baseline report and workflow documentation.

Generated sweep weights, optimizer checkpoints, caches, logs, downloaded DUTS and
reference-model downloads are not committed. Only the best per-run checkpoints,
the selected baseline export and the active deployment checkpoint are retained.
