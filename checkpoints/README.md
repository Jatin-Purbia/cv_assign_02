# Training checkpoints

The trainer writes best inference exports and last optimizer/scheduler checkpoints in the chosen experiment folder. Synthetic unit-test checkpoints are temporary and never final submission weights. Only the validation-selected frozen inference parameter file belongs in `weights/final_model.pth`.
