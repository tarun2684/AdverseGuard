# 🧾 ChangeLog

## [Day 2] - 2025-07-24
### Added
- Implemented `fgsm.py` with FGSM adversarial attack generation.
- Added `evaluate_impact.py` to compare clean vs adversarial accuracy.
- Updated `main.py` to support single-image demo + batch testing.
- Visualized original and adversarial image predictions.
- Printed accuracy and prediction shifts to console.
- Created `docs/attack_theory.md` with math and diagrams for FGSM.

## [Day 1] - 2025-07-23
### Added
- Set up project structure (`src`, `data`, `models`, `docs`).
- Downloaded and loaded MNIST dataset.
- Built a basic CNN classifier for MNIST.
- Added image normalization and preprocessing.
- Set up `main.py` CLI entry point for evaluation.
