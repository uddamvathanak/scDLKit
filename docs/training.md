# Training

`Trainer` provides:

- Adam optimization
- device auto-detection
- optional mixed precision on CUDA
- early stopping
- best-checkpoint restoration
- pandas training history for downstream plotting

The training core uses plain PyTorch rather than a higher-level orchestration framework, keeping the implementation compact and transparent.
