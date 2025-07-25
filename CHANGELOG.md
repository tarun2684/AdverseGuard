# 🧾 ChangeLog
### [Day 3] - 2025-07-25  

**Improved**  
- Refactored `load_model()` to fix incorrect keyword usage (`model_path` ➝ `path`) in `main.py`.  
- Handled exceptions for missing model files and provided meaningful error messages.  

**Tested**  
- Ran `main.py` with `--attack fgsm` successfully.  
- Verified adversarial example generation with FGSM.  
- Evaluated model accuracy on clean (97.80%) and FGSM adversarial samples (75.16%).  
- Verified individual prediction output and console summary.  

**Documented**  
- Interpreted and logged the accuracy drop from clean to adversarial inputs.  
- Understood how FGSM perturbations affect batch predictions even when single predictions may still succeed.  


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
