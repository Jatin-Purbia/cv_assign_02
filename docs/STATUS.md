# Status - 1 November 2026

The active release completed six fine-tuning epochs; epoch 6 was best. It runs
at 128 pixels with threshold 0.5 and channels-last layout. Validation Q is 0.78986.
One frozen evaluation covered all 5,019 DUTS-TE pairs: mIoU 0.71445, BF1 0.44408,
Q 0.63334, mean CPU 17.141 ms, CES 0.56293. Baseline results are preserved.
The qualified runtime is Python 3.12.4 / PyTorch 2.7.1+cpu; see the runtime pins.

Captured photos, resized images, pretrained EfficientSAM reference masks and
model predictions are consolidated in `captured/`, with explicit AI provenance.
They do not provide manually verified captured-photo ground truth. Captured-photo
accuracy and capture metadata remain unavailable. The report still describes
the original baseline and must not be presented as the active release's report.

Duplicate releases, derived sweep weights, completed optimizer states, abandoned
runs and duplicate reference outputs were removed. Completed experiment metrics,
best checkpoints and active-release evidence are retained. See `results/tests.json`
for the latest executed verification and `results/submission_audit.json` for the
remaining submission requirements.

Current comparisons: `captured/model_predictions/comparison.png` and
`captured/reference_comparison.png`. Full release details: `MODEL_QUALITY.md`.
