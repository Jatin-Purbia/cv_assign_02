# Actual project status — 29 October 2026

**Not submission-ready.** This checklist records evidence, not promised results.

## Verified

- Read all three PDF pages; rendered all pages and visually inspected the exact contract. PDF text extraction loses underscore glyphs. Canonical captured folders are `images_256` and `masks_256`.
- Implemented original Tiny FCN, lightweight U-Net and depthwise-separable encoder–decoder models, BCE and explicit soft Dice.
- Implemented strict data loading, deterministic splitting, training, validation, checkpoint/export, inference, CPU timing, scoring, plotting and report-source generation.
- Downloaded the **official** DUTS-TR archive, 270,997,309 bytes, SHA-256 `ce61e023c8f59d022b4d46981cf16813b83d089242e6489a45630d83962ea058`.
- Decoded all 10,553 matched TR pairs. Real seed-42 manifests contain **9,498 training / 1,055 validation IDs**.
- **37 tests passed**, with zero failures/errors, including analytic metric/scoring tests and synthetic forward/backward, train/checkpoint, fresh-process CLI tests. Machine-readable evidence is in `results/tests.json`; full output is in `results/test_output.txt`. Rerun after further edits.
- Actual CPU detection: **Intel(R) Core(TM) 7 150U**. Available torch build: **2.7.1+cpu**, no CUDA.
- Genuine untrained synthetic latency observations are saved in `results/synthetic_latency.json`. These are not DUTS quality, not final model selection and not official timing compliance.
- Downloaded official portable Tectonic 0.17.0 into ignored tmp/, compiled a six-page method-report draft, and inspected its rendered pages. Layout fixes and final numerical/figure regeneration are in progress. No system LaTeX installation was changed.

## In progress

- Real CPU pilot in `experiments/pilot4` completed three two-epoch runs comparing Tiny/BCE, depthwise/BCE and Tiny/BCE+Dice, using four training threads. Validation resolution/threshold sweeps are in progress. The initial single-thread run in `experiments/pilot` was interrupted before completing an epoch after a genuine 8-batch diagnostic showed about a 2x training speedup with four threads. Its partial state is not a completed experiment. All inference/timing resets to one thread. The 40-epoch full grid is configured, not claimed executed.
- A bounded refinement in `experiments/refinement/tiny8_depth4_bce_128` uses width 8, four stages and four epochs in response to poor pilot boundary quality and substantial low-resolution timing margin. Its training budget differs from the pilot, which is explicitly documented.
- Expanded unit suite, source inventory, report fragments and submission audit are regenerated after implementation changes.

## Mandatory items still pending

- Completed real validation experiments, resolution comparison, measured model selection and frozen learned parameter file.
- Complete one-pass DUTS-TE evaluation/timing on all 5,019 images and measured <=50 ms compliance. **DUTS-TE has not been downloaded or used for tuning.**
- User's **5–10 authentic personal phone photographs**, manual masks and capture/condition metadata. Existing folders contain no fabricated examples.
- Captured-photo metrics and overlays for every photograph.
- At least two genuine observed failure explanations with actual image evidence.
- Final report regeneration and visual review after experiments. **pdflatex is unavailable, but portable Tectonic builds work**. Numerical rows come only from saved evaluations; missing evaluations are labelled explicitly.
- Clean final package; packaging intentionally refuses missing required artifacts.

## Contract clarification

The user explicitly clarified that the PDF stdout labels take precedence: `T_CPU_ms:` and `S_disk_bytes:`. All success-path tests enforce exactly two lines and exactly 20 timed calls after the first output-producing warmup. No random/test checkpoint is installed as the default final model.

## Reproduction

See root README and `docs/REPRODUCIBILITY.md`. The real pilot command is:

```powershell
python -m scripts.run_experiments --config configs/pilot.json --experiments configs/pilot_experiments.json --out experiments/pilot4
```

Do not interrupt/reconfigure experiments and then describe their partial logs as completed runs. After selection, freeze once and perform the final test without retuning. Photograph authenticity and condition coverage require human review in addition to automated file validation.
