# Experiment protocol and evidence

The full grid in `configs/experiments.json` compares Tiny FCN, deeper lightweight U-Net and depthwise FCN, then loss and internal-resolution changes. Keep seed, split, augmentation and optimizer fixed. Train with BCE as mandatory baseline; compare BCE + 0.5 Dice on the same split. Select each run's best epoch by validation Q. Measure validation CPU time through the exact full inference pipeline and actual parameter-file bytes. Select final candidate by validation CES, with attention to CPU margin. The script saves a comparison CSV and selection JSON. The CPU pilot uses two epochs and narrower features; it is explicitly a limited experiment and cannot establish convergence or highest attainable quality.

Do not reuse DUTS-TE for model selection, threshold selection, augmentation design or stopping. Freeze architecture, settings and parameter hash before loading it. A final evaluation output is not overwritten automatically. Captured photos are also testing-only.

`results/synthetic_latency.json` contains genuine measurements on a deterministic random image and **untrained** parameter files. It is a software/latency diagnostic, not a quality comparison, not a final model choice and not official T_CPU. Its small sample cannot prove the complete 5,019-image budget.

Train/validation history CSVs must be the source of learning curves. Comparison figures use actual measured CSV rows. Every captured photograph must get overlays. Failure candidates are the lowest per-image Q observations in the generated overlay index; manually inspect at least two to explain their actual causes. Do not infer poor lighting or occlusion from a numerical score alone.

Record actual executed runs and limitations in STATUS.md. No claim of optimal CES is supported merely by having a grid configuration.

The original 5x5 boundary match used permitted torch max pooling. A genuine CPU diagnostic found whole-array boolean windows much faster. The final metric implementation uses 25 NumPy OR passes (never per-pixel Python loops); analytic tests plus pooled-reference equality checks cover empty/full, tiny/non-square arrays and random boundaries. Completed earlier observations remain valid because the metric is exactly equivalent, and metric calculation was always outside the inference timer. Resolution sweeps can resume only when parameter state, configuration, checkpoint hash and manifest hash match saved evidence.

After the two-epoch pilot exposed low Boundary F1 and substantial latency headroom at 64, one bounded refinement uses base width 8, depth 4 and a four-epoch budget (`configs/refinement.json`). Its larger context/capacity and larger training budget are changed together; it is a practical refinement, **not an isolated causal ablation of depth**. All selection still uses the same DUTS-TR validation set. Append its observations after earlier sweeps have finished to avoid concurrent comparison-file writers.
