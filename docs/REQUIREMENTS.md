# Requirements traceability and live checklist

Authority: all three pages of `CV_Assignment2.pdf`, read 2026-10-29. The user development prompt adds engineering requirements. Status vocabulary: implemented (code exists), verified (tests/evidence exist), pending (requires data/experiment), not applicable (other track). Implementation alone is not experimental compliance.

| ID | Class | Original requirement / source | Component | Verification | Status |
|---|---|---|---|---|---|
| A01 | Mandatory | Read original PDF before implementation | docs/assignment_extracted.txt | Entire 3-page extraction reviewed | Verified |
| D01 | Mandatory | DUTS-TR 10,553; DUTS-TE 5,019 (1a) | scripts/prepare_dataset.py | Pair/count/decode audit | TR verified; TE pending freeze |
| D02 | Mandatory | Split TR; all design choices on validation; never tune on TE (1a) | manifests, freeze metadata | Disjoint partition; final evaluation requires freeze | Real TR split verified: 9498/1055 |
| D03 | Mandatory | Direct 256x256 INTER_AREA photos, INTER_NEAREST masks; >=128 foreground (1b) | src/preprocessing.py | Analytic preprocessing tests | Verified |
| D04 | Mandatory | 5–10 genuine phone photos, >=3/5 condition categories, testing only (1c) | captured/ | Validate manifest and images; human authenticity attestation | Pending user photos |
| D05 | Mandatory | PNG masks 0/255; same stems; literal `images_256`, `masks_256` (deliverables) | captured/ | Captured audit | Pending user annotations |
| L01 | Mandatory | Only stdlib, torch, numpy, cv2, matplotlib for listed functions (2a) | all runtime source | AST import/call audit + manual review | Verified source and dependency list |
| L02 | Prohibited | Pretraining, segmentation/clustering/maxflow/morphology/metric libraries; other engines (2a) | compliance audit | Prohibited import/function scan | No prohibited usage found |
| M01 | Mandatory | Track B FCN from scratch; encoder/decoder and skips; CE, optional Dice | src/models.py, losses.py | Forward/backward and checkpoint tests | Verified software; real pilot running |
| T01 | Mandatory | Timed 256 image in memory to binary 256 mask; all intermediate steps count (2b) | src/inference.py | Shared pipeline tests/review | Verified software |
| T02 | Mandatory | CPU, one thread, no extra processes; four env vars BEFORE imports (2b) | src/runtime.py, entrypoints | Fresh-process CLI tests | Verified |
| T03 | Mandatory | Warmup; process_time; all 5,019 TE one by one on own laptop (2b) | evaluate.py, benchmark.py | Full evaluation record/count | Pending real final evaluation |
| T04 | Mandatory | Mean <=50 ms else CES=0 (2b,d) | scoring; evaluator | Boundary tests; measured final timing | Pending final timing |
| S01 | Mandatory | Actual os.path.getsize bytes; all learned values in file; no extra compression (2c) | final export, inference | Fresh load, size check | Pending trained weights |
| E01 | Mandatory | Global 2x2 confusion, average FG/BG IoU (metrics) | src/metrics.py | Analytic aggregation tests | Verified |
| E02 | Mandatory | Four-connected inner FG boundary; outside BG (metrics) | src/metrics.py | Corners/borders/full/empty tests | Verified |
| E03 | Mandatory | Centered 5x5 matches both directions; per-image BF1 mean, empty conventions (metrics) | src/metrics.py | Shift/diagonal tests | Verified |
| E04 | Mandatory | Q=.7 mIoU+.3 BF1; exact CES formula (2d) | src/metrics.py | Units/caps/50 ms/example tests | Verified |
| P01 | Mandatory | predict.py --image --out; PNG 0/255; first inference then 20 runs; exactly two lines (deliverables) | predict.py | Fresh-process integration tests | Verified on test checkpoint; final weights pending |
| R01 | Mandatory | Report method, efficiency, exact split sizes, CPU model (deliverables) | report/main.tex | Compare with experiment evidence | Pending real results |
| R02 | Mandatory | One val/TE/captured table mIoU BF1 Q plus T,size,CES (deliverables) | report generator | Values sourced from JSON only | Pending real results |
| R03 | Mandatory | Every captured photo true/pred overlays; >=2 real failure cases (deliverables) | figures/report | Visual review and evidence review | Pending real photos/results |
| U01 | Mandatory (user) | >=2 candidate models, validation CES selection, BCE/Dice and resolution experiments | configs, scripts/run_experiments.py | Comparison CSV and frozen selection provenance | Pending real experiments |
| U02 | Mandatory (user) | Full train/evaluate/checkpoint/early stopping/scheduler/log/plot/config/repro pipeline | trainer, scripts | Synthetic smoke plus real run | Pending verification |
| U03 | Mandatory (user) | Rigorous unittest, CLI/timing/compliance tests; no invented outputs | tests/ | unittest discovery | 37 tests passed; results/tests.json |
| U04 | Mandatory (user) | Complete mathematical LaTeX report; compile if available; clean audit/package | report/, scripts/ | Build, PDF visual QA, audit | Pending empirical completion/compiler |
| O01 | Optional | Soft Dice, low precision storage, TorchScript/compile (2a,c,Track B) | losses / export | Experiments if used | Dice supported; other options deferred |
| I01 | Recommended improvement | Safety margin under 30ms; three candidates | model configs | Measured synthetic latency is diagnostic only | Pending |
| X01 | Mandatory, other track | Track A clustering/probabilities/graph-cut | None | Selected Track B | Not applicable |

## Decisions and ambiguities

- Visual PDF review confirms underscores: use `captured/images_256` and `captured/masks_256` exactly. Text extraction lost the underscore glyphs; the rendered original takes precedence.
- User clarified that the PDF takes precedence for stdout labels: print `T_CPU_ms:` and `S_disk_bytes:` exactly, with two lines total.
- PDF CES example rounds to .635; exact arithmetic is .63525. Implement the formula.
- Undefined absent-class IoU: include both classes, define zero-union IoU as 1 (perfect absence), consistently across sets. This convention is documented, not attributed to the PDF.
- Output interpolation: bilinear logits to 256 before sigmoid and >= threshold. Fixed RGB float32 [0,1], no mean/std normalization.
- No original data, captured photographs, or weights are present at initialization. Never treat synthetic smoke evidence as DUTS performance.
- CPU model discovery must be truthful; record unavailable if OS access prevents discovery.
- Use PyTorch tensor/nn operations for augmentation and boundary matching; avoid OpenCV GUI/morphology/drawing dependencies.
- Analysis-only PDF tooling is not a project runtime dependency.

## Completion rule

The project is **not submission-ready** until every pending mandatory empirical item above is resolved. See `docs/STATUS.md` for actual executed evidence. Empty manifests/weights are not fabricated to satisfy existence checks.
