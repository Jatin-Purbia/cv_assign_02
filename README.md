# Cost-Efficient Segmentation — CSL7360 Assignment 2

Jatin Purbia · B23CH1022 · Indian Institute of Technology Jodhpur

An original PyTorch Track B implementation, with three lightweight encoder–decoder candidates, manual assignment-specific metrics, deterministic DUTS splitting, real training, shared CPU inference, validation model selection, and a strict prediction CLI.

**Read [docs/STATUS.md](docs/STATUS.md) before submitting.** Missing personal photos, final experiments or a compiled report mean this is not a complete submission. No pretrained weights, invented metrics, or substitute photographs are included. Requirements and decisions are traced in [docs/REQUIREMENTS.md](docs/REQUIREMENTS.md).

## Setup and verification

Python 3.10+; CPU PyTorch is sufficient. A GPU may be used for training by adding `--device cuda`. All timed evaluation is CPU-only.

```powershell
python -m pip install -r requirements.txt
python -m unittest discover -v
python -m scripts.verify
python -m scripts.audit
```

The runtime imports only stdlib, PyTorch, NumPy, OpenCV and Matplotlib. See [docs/LIBRARY_COMPLIANCE.md](docs/LIBRARY_COMPLIANCE.md). No package installation occurs inside prediction. Launch each command from this directory. Run inference/evaluation in a fresh Python process so thread environment variables take effect before numerical imports.

## Official data

Source: [official DUTS page](https://saliencydetection.net/duts/). Download scripts use the links published there, record observed SHA-256 hashes, validate archive paths before extraction and do not substitute datasets. If network downloads fail, download the archives in a browser and put them in `data/DUTS` before rerunning the commands.

```powershell
python -m scripts.download_duts --split TR
python -m scripts.prepare_dataset --root data/DUTS
```

Expected layout: `data/DUTS/DUTS-TR/DUTS-TR-Image/` and `DUTS-TR-Mask/`. Preparation decodes all 10,553 pairs, rejects duplicates/unmatched stems and saves 9,498 training / 1,055 validation IDs with seed 42. No fake manifests are shipped if data preparation has not completed.

## Training and validation-only selection

```powershell
python train.py --config configs/baseline.json --out experiments/baseline
python -m scripts.run_experiments --config configs/baseline.json
```

The baseline is 40 epochs; early stopping may shorten it. The full experiment grid compares three architectures, BCE/Dice, and 128/160/192/256 resolutions. Expect substantial CPU training time. Training uses four CPU threads; all Predictor construction resets numerical runtimes to one thread before timing. Run directories are immutable: choose a new directory to change a configuration. The runner skips completed matching runs. `last_checkpoint.pth` preserves optimizer/scheduler states for research recovery; automatic mid-epoch resume is not implemented.

A smaller **pilot**, limited to two epochs per candidate, is available to verify real-data behavior. It does not establish convergence or optimal quality:

```powershell
python -m scripts.run_experiments --config configs/pilot.json --experiments configs/pilot_experiments.json --out experiments/pilot
```

Do not run the pilot and full grid concurrently: both write `results/model_comparison.csv` and `results/selection.json`. The pilot compares Tiny FCN vs separable FCN and BCE vs BCE+Dice. Use the full grid for resolution ablation and a larger training budget. Configured hyperparameters, actual epoch counts, package versions, split hashes and best metrics are saved per run.

To measure inference-resolution changes of an already trained run on validation only:

```powershell
python -m scripts.sweep_inference --run experiments/pilot/tiny4_bce_128 --resolutions 128 160 192 256
```

This appends actual validation observations and updates the selection. Training resolution remains recorded in `run.json`; each inference variant stores its own resolution. Run this before freezing.

Optionally compare the selected candidate at a bounded set of decision thresholds, also before freezing: `python -m scripts.sweep_threshold --thresholds 0.4 0.45 0.5 0.55 0.6`. All rows and the final choice use DUTS-TR validation only. Training best-epoch Q is measured at 0.5; the final exported inference threshold is a separate validation-selected setting recorded in the bundle and final configuration.

The bounded refinement uses a wider four-stage FCN and four epochs. After existing sweeps finish, preserve their measured rows with:

```powershell
python -m scripts.run_experiments --config configs/refinement.json --experiments configs/refinement_experiments.json --out experiments/refinement --append
python -m scripts.sweep_inference --run experiments/refinement/tiny8_depth4_bce_128 --resolutions 64 96 128 160 192 256 --resume
python -m scripts.sweep_threshold --thresholds 0.4 0.45 0.5 0.55 0.6
```

This changes capacity, depth and training budget together; it tests a practical refinement rather than isolating a single factor. The 40-epoch full grid remains separate and must not be described as completed unless its real artifacts exist.

Freeze only when validation experiments and design choices are finished:

```powershell
python -m scripts.freeze_model
python evaluate.py --split validation --weights weights/final_model.pth --images data/DUTS/DUTS-TR/DUTS-TR-Image --masks data/DUTS/DUTS-TR/DUTS-TR-Mask --manifest data/splits/val.txt --out results/validation_metrics.json
```

`weights/final_model.pth` is a plain uncompressed torch.save bundle containing architecture/settings, all learned weights and BN buffers. It is loaded using `weights_only=True`, CPU mapping and strict keys. The freeze record binds the selected checkpoint to validation evidence. Never use synthetic test checkpoints as final weights.

Freeze also writes the complete selected training configuration to `configs/final.json`; `resolution` records training resolution and `inference_resolution` records the selected inference setting. Reproduce training with `python train.py --config configs/final.json --out experiments/reproduce_final`, then repeat the documented validation selection/export. Do not overwrite existing final-test evidence.

## Final evaluation and timing

After freezing, download and evaluate DUTS-TE **once** for reporting. Never use it to tune anything. Timing starts with a 256-square BGR image in memory and ends with a 256-square 0/1 mask. It includes color conversion, normalization, tensor conversion, internal resizing, forward pass, output resizing, sigmoid and thresholding. All 5,019 images are measured separately with process CPU time after warmup; I/O, original resizing, metrics and parameter loading are excluded. CPU threads and four required environment variables are forced to one.

```powershell
python -m scripts.download_duts --split TE
python -m scripts.verify_dataset --root data/DUTS --split TE
python benchmark.py --split duts-test --images data/DUTS/DUTS-TE/DUTS-TE-Image --masks data/DUTS/DUTS-TE/DUTS-TE-Mask --out results/duts_test_metrics.json
python predict.py --image photo.jpg --out mask.png
```

`benchmark.py` and `evaluate.py` share the complete evaluator; `predict.py` shares its exact `Predictor`. Prediction uses project-relative default weights even when launched from another directory and prints exactly:

```text
T_CPU_ms: <measured mean to two decimal places>
S_disk_bytes: <actual parameter file bytes>
```

The first output-producing inference is warmup, followed by exactly 20 timed runs. Errors are written to stderr and return nonzero. Use `--weights` only for explicit experimental/test checkpoints. The average official CPU time must be <=50 ms; otherwise CES is zero. A short diagnostic benchmark cannot establish this requirement.

## Personal photographs

Follow [captured/README.md](captured/README.md). Supply 5–10 genuine phone photos, manually annotate masks at 256 square and document at least three required condition categories. Visual PDF review confirms literal underscores in folder names. The optional Matplotlib polygon tool needs an interactive display:

```powershell
python -m scripts.annotate --image my_phone_photo.jpg
python -m scripts.verify_dataset --root captured --split captured
python evaluate.py --split captured --images "captured/images_256" --masks "captured/masks_256" --out results/captured_metrics.json
python -m scripts.create_figures overlays --images "captured/images_256" --masks "captured/masks_256" --out results/captured_figures
```

All captures must appear in the report. The tool suggests the two lowest-Q observed images for failure review; inspect them and write evidence-based explanations in `report/failure_analysis.tex`. It does not invent explanations.

## Figures, report and packaging

Replace `<selected-run>` below with the directory recorded in `results/selection.json`.

```powershell
python -m scripts.create_figures history --csv <selected-run>/history.csv --out results/figures
python -m scripts.create_figures comparison
python -m scripts.build_report
python -m scripts.audit --submission
python -m scripts.package_submission
```

`report/main.tex` contains the full mathematical method. Build fragments pull actual saved results into its mandatory table; absent evidence is explicitly labelled not evaluated. Install a LaTeX distribution with pdflatex/TikZ to compile. Inspect every PDF page before submission. Packaging refuses missing mandatory artifacts, copies only an allowlisted clean folder, and excludes DUTS and training checkpoints. Human review remains required for report quality, actual photographic provenance, condition coverage and failure analysis.

An explicit portable compiler is also supported: `python -m scripts.build_report --engine C:/path/to/tectonic.exe`. Keep its cache under `tmp/` with `TECTONIC_CACHE_DIR`; compiler binaries/caches are not submission dependencies. `python -m scripts.package_submission --draft --out submission/INCOMPLETE_REVIEW` creates a clearly labelled review snapshot while mandatory personal-photo evidence is missing.

Portable compiler setup is described in the [official Tectonic installation guide](https://tectonic-typesetting.github.io/install.html).

Global mIoU uses one dataset confusion matrix. BF1 is averaged per image, uses four-connected foreground boundaries and centered 5x5 matching, with the specified empty-boundary handling. Zero-union class IoU is explicitly defined as 1. CES uses milliseconds, bytes and 4 MiB = 4,194,304 bytes.
