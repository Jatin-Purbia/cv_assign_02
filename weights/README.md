# Final learned parameters

`final_model.pth` is created by `python -m scripts.freeze_model` after real training and validation-only selection. No random, fabricated or pretrained final weights are supplied. `final_model.freeze.json` records the selected weights' SHA-256, validation evidence and configuration. Prediction fails clearly if trained weights are absent.
