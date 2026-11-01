# Reproduction details

See the root README for exact executable commands in dependency order. Use a new Python process for each CLI. Python >=3.10 is required; torch >=2.6 is required for the explicit safe `weights_only=True` loading behavior. `results/environment.json` records actual verified versions and CPU identification where available. `requirements.txt` gives compatible minimums rather than claiming an untested universal lockfile.

## Determinism

Training seeds Python, NumPy and torch with 42; CUDA generators are seeded if training on a GPU. Deterministic algorithms are requested, cuDNN autotuning is disabled, and DataLoader workers are zero. Training threads are recorded in the config (10 for the final U-Net and quality fine-tuning). Every Predictor construction resets the runtime to one thread before inference, including validation timing in the experiment runner's process. Training shuffle uses a seeded generator. Augmentation uses seeded torch RNG. Reproducibility is expected within the same software/hardware stack; floating-point results can differ across stacks. The checkpoint and manifest SHA-256 identify the exact evaluated artifacts.

## Preprocessing contract

Read photographs as OpenCV BGR uint8. Resize originals directly to 256x256 with INTER_AREA outside the official timer. Convert to RGB CHW contiguous float32/255 inside inference. No mean/std normalization. Optional bilinear tensor downscaling is timed. Resize logits bilinearly to 256x256, apply sigmoid and >=0.5, convert uint8 0/1. PNG export converts 0/1 to 0/255 outside timing. Grayscale ground truth is resized with nearest neighbor and thresholded >=128.

## Saved state

Each run saves original config, package/CPU details, exact split counts/hashes, model parameter count, history, best epoch and metrics. New format-version-2 `last_checkpoint.pth` files also preserve RNG states, shuffle-generator state, best-model state, history and early-stopping state. Use `train.py --resume` with the identical config and run directory to recover from the last completed epoch. Older checkpoints lack RNG state and can initialize a new fine-tuning run through `initialize_from`, but cannot claim exact resume. `best.pth` contains complete inference state and settings. Exported state includes batch normalization's running buffers. No learned value is embedded in source code. `torch.save` is used without external compression. See `MODEL_QUALITY.md` for the versioned release workflow.

## Timing

Four thread variables are assigned "1" before numerical imports. cv2 and torch threads are set to 1, including torch interop. The eager CPU pipeline has no worker processes. Model loading is excluded, one full pipeline warmup is excluded, then each image is wrapped with time.process_time. Official result saves every image's process duration and exact count. Windows process time may have coarse clock granularity; the required sum/mean across all 5,019 samples is still used without substituting a wall clock. Avoid running training, tests or other heavy tasks concurrently with the official evaluation. All prediction/evaluation paths call the same Predictor.

## Reporting and audit

Run figure generation from actual experiment artifacts. `scripts.build_report` regenerates its results table from saved JSON, rejecting incorrect split labels. Compile with pdflatex/TikZ or `--engine /absolute/path/to/tectonic.exe`. Tectonic's cache can be kept under project `tmp/` using TECTONIC_CACHE_DIR; it is not a segmentation dependency. Visual review of every compiled page is required. `scripts.audit --submission` exits nonzero while required weights/data/report/results are missing. The package command copies an allowlisted folder only after this audit passes; `--draft` explicitly creates an incomplete review snapshot. A person must still verify capture authenticity, conditions, failure explanations and report fidelity.

The test suite uses manually constructed analytic arrays and synthetic images for software verification. Their results must never be relabelled DUTS or captured-photo performance. Fixture files live in workspace `tmp/tests`; they are cleaned after tests. This avoids a Windows sandbox ACL incompatibility with tempfile.mkdtemp mode-0700 directories.

## Clean repository retention

Completed runs retain best checkpoints, configuration, histories and metrics. Completed optimizer states and redundant sweep exports were pruned; new runs still create ignored resume checkpoints. The selected baseline export is retained for report/freezing commands, and quality qualification can regenerate its candidate exports from the retained best checkpoint. Start reproduction in a fresh run/output directory. `.gitattributes` preserves LF source bytes and the original CRLF split-manifest bytes so frozen source/manifest hashes survive Windows Git checkouts.
