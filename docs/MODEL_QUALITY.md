# Model improvement and release workflow

This remains an automatic salient-object segmentation model for the assignment's
single-thread CPU budget. It does not accept an object name, click, or box. In a
cluttered scene, the desired foreground must be defined before judging accuracy.
The assignment model uses no pretrained weights or prohibited segmentation/morphology libraries. The separate reference-mask utility is documented in `reference_masks/README.md`; its outputs are AI references only.

## Measured release, 1 November 2026

The active release is `weights/releases/e219af25d4c072af_20261101T121534155210Z/model.pth`.
It retains 439,393 parameters, uses 128-pixel inference, threshold 0.5 and
channels-last layout. All six fine-tuning epochs completed; epoch 6 was best.

| Set / profile | mIoU | BF1 | Q | Mean CPU ms | CES |
|---|---:|---:|---:|---:|---:|
| Original validation | 0.8503 | 0.5958 | 0.7739 | 13.09 | 0.7005 |
| Active validation | 0.8591 | 0.6284 | 0.7899 | 24.32 | 0.6794 |
| Original DUTS-TE | 0.7049 | 0.3942 | 0.6117 | 16.87 | 0.5444 |
| Active DUTS-TE | 0.7144 | 0.4441 | 0.6333 | 17.14 | 0.5629 |

Historical original rows used Python 3.10 / PyTorch 2.10; active rows used Python
3.12 / PyTorch 2.7.1. Timings describe their recorded environments, not an isolated
causal comparison of training changes. The controlled layout comparisons in
qualification use the same runtime for both layouts. The active profile prioritizes
quality within budget; its validation CES is below the original speed-focused row.

The paired 95% interval for validation Q gain over the original 96-pixel profile is
[0.01101, 0.02086]. Lighting stress Q gains are +0.02361 (dark), +0.01777 (bright),
and +0.02302 (low contrast). After selection and freezing, one final evaluation
covered all 5,019 DUTS-TE pairs. The test result did not feed back into selection.
The active file is 1,787,043 bytes. Verification evidence is recorded in `results/tests.json`; the release is pinned to its actually measured runtime.

Evidence: `results/quality/qualification.json`, the immutable qualification
snapshot alongside the released weights, and `results/quality/duts_test_metrics.json`.
The new comparison is `captured/model_predictions/comparison.png`, with ten binary masks
under `captured/model_predictions/masks/`. The CLI was checked using the active pointer
and produced exactly its two required lines.

Visual review still finds the rug merged with shoes, missed vegetation, and
merged bicycle structures. The cleaner dustbin and broader foreground coverage
do not establish production accuracy. Deployment-domain acceptance remains
pending explicit foreground definitions and representative manually labelled data.

## Candidate training

`configs/quality_finetune.json` starts from this project's own scratch-trained
U-Net, uses a fresh AdamW optimizer at 0.0003, and fine-tunes for up to six epochs
at 128 pixels. The loss is BCE + 0.3 soft Dice. Mild paired scale/translation and
brightness/contrast/saturation variation increase training diversity. This is a
combined improvement candidate, not an isolated causal ablation of each change.
Validation still uses the unchanged 1,055-image DUTS-TR manifest. Neither DUTS-TE
nor captured photos enter training or candidate selection.

```powershell
python train.py --config configs/quality_finetune.json --out experiments/quality/reproduction
# After an interruption, resume the same run/configuration:
python train.py --config configs/quality_finetune.json --out experiments/quality/reproduction --resume
```

Training checkpoints atomically preserve model, optimizer, scheduler, best state,
history, data-loader shuffle state, Python/NumPy/PyTorch RNG state, and CUDA RNG
state when applicable. Resume rejects changed configurations, manifests,
initialization weights, or recorded training source. An interrupted history write
is repaired from the committed checkpoint. The tests compare resumed learned
state bit-for-bit against uninterrupted CPU training. Old checkpoints without RNG
state can initialize a new fine-tuning run, but cannot claim exact resume.
Recovery restarts at the last committed epoch. If an interruption occurs before
the first epoch finishes, choose a fresh run directory; no learned epoch exists
to resume yet.

## Qualification and activation

Wait for training to finish before timing candidates. Close heavy competing work.
The active release was actually qualified and benchmarked with Python 3.12.4 /
PyTorch 2.7.1+cpu / NumPy 2.2.6 / OpenCV 4.12.0 / Matplotlib 3.10.8. The current
default `python` selects that Anaconda environment. `requirements-runtime.txt`
records its distribution package pins (the installed torch distribution reports
2.7.1; its runtime reports 2.7.1+cpu). The historical baseline used the separately
installed Python 3.10 stack. Release metadata records the measured runtime and is
authoritative; using another stack requires qualification for that stack.

```powershell
python -m scripts.qualify_model --runs experiments/quality/unet16_dice_aug_recovered --out results/quality_new_run --activate
```

This evaluates the old frozen model, the old model at 112/128 pixels with standard
and channels-last layouts as controls, and the fine-tuned candidate at 96/112/128/144
pixels and thresholds 0.45/0.5 with channels-last layout. A release
must gain at least 0.005 validation Q, preserve BF1, lose at most 0.002 mIoU, average
at most 40 ms CPU, have empirical p95 at most 50 ms, and have at most 25% of images
regress by more than 0.05 per-image Q. These are engineering gates on this
validation set, not a guarantee of real-world reliability or an official test score.
The paired 95% bootstrap interval for Q improvement must also lie above zero.
The bootstrap resamples images jointly for both models, recomputes global
confusion-matrix IoU and mean BF1, and uses 2,000 deterministic draws. It measures
validation-sampling uncertainty, not uncertainty on a new deployment domain.
Among passing candidates, quality Q takes priority; time breaks ties. A candidate
can improve quality while having lower CES than the speed-optimized baseline.
Before activation, the candidate is also compared against the baseline on a
deterministic 128-image validation subset under dark, bright, and low-contrast
transforms. Mean stress Q must not fall, and no condition may lose over 0.01 Q.
These are simulated checks, not a substitute for genuine deployment examples.

The selected checkpoint is copied into `weights/releases/<hash>_<UTC>/model.pth`, with
a freeze sidecar and a qualification snapshot. Only after evidence is written does
the atomic `weights/active_model.json` pointer switch. `predict.py` checks the
active release's hash, inference source hashes and validated numerical runtime,
and fails on invalid manifests or unsupported runtime versions.
Without an active pointer it uses the original frozen model. The original
`weights/final_model.pth` and original official results are preserved.

The channels-last option uses standard PyTorch tensor storage and convolution
kernels, not another inference engine. The configured predictor inherits the
original timed inference function; layout handling and operator conversions are
therefore included in timing. The checkpoint records the choice, and evaluation,
prediction, stress checks and overlays load it consistently. Loading and model
layout initialization occur outside timing, like parameter loading. See the
[PyTorch CPU memory-format discussion](https://pytorch.org/blog/accelerating-pytorch-vision-models-with-channels-last-on-cpu/).

```powershell
python predict.py --image photo.jpg --out output/mask.png
python -m scripts.predict_folder --images captured/original --out captured/prediction_runs
# Explicit baseline comparison, independent of the active pointer:
python predict.py --image photo.jpg --out output/baseline.png --weights weights/final_model.pth
```

The prediction CLI keeps its exact two-line grading output. The folder command
preserves existing output folders, generates a comparison montage, and records
checkpoint/input hashes and foreground fractions. Area warnings request review;
they are not confidence scores. There is no automatic cleanup that could erase
small or disconnected objects.
Masks are stored under the output's `masks/` folder. Contact sheets are paginated
at ten photos per image to keep memory bounded for larger folders.

## Limits before a production claim

- Genuine captured-photo masks and documented foreground definitions are still
  necessary to measure performance on the actual deployment scenes.
- The original DUTS-TE score belongs only to the original checkpoint. It must not
  be attributed to a new release. No repeated DUTS-TE tuning is performed.
- A validation-qualified candidate is a release candidate. A production claim
  requires an independently labelled representative holdout, agreed acceptance
  targets for failures, and monitoring after deployment.
- This CPU machine has no available CUDA device; training is a substantial local
  computation and exact elapsed time depends on other applications and cooling.

The original report and submission metrics describe the original frozen baseline.
New qualification evidence lives separately under `results/quality`.

## Checkpoint retention

Completed best checkpoints, selected baseline export and active release are retained. Optimizer states, derived sweep weights, abandoned runs and duplicate releases were pruned. Exact resume is available for new interrupted runs that still have their locally generated `last_checkpoint.pth`; it is not available for pruned completed runs. Recorded validation observations remain historical evidence, including paths/hashes of removed exports.
