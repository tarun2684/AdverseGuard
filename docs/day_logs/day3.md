# 📅 Day 3 – 2025-07-25

## ✅ Summary of Tasks Completed

- ✅ Implemented **Projected Gradient Descent (PGD)** adversarial attack in `pgd.py`
- ✅ Refactored CLI in `main.py` to accept `--attack` flag (e.g., `fgsm`, `pgd`)
- ✅ Trained and saved a **stronger CNN model** (`models/strong_cnn.py`)
- ✅ Added CLI support to train models (`--train` flag)
- ✅ Cleaned up code structure across `src/attacks`, `src/models`, `src/evaluation`
- ✅ Verified both **FGSM and PGD attacks** against clean and adversarial accuracy
- ✅ Enhanced logs to print prediction shifts clearly
- ✅ Used Docker image to test everything inside a container

---

## 📁 Files Worked On

| File/Module               | Description                                                                 |
|---------------------------|-----------------------------------------------------------------------------|
| `src/attacks/pgd.py`      | New file implementing iterative PGD adversarial attack                      |
| `models/mnist_cnn.pt`| Deeper CNN architecture with better baseline accuracy  (After Training)                     |
| `main.py`                 | Refactored to support CLI arguments for attack type and training            |
| `evaluate_impact.py`      | Updated to support PGD alongside FGSM for impact comparison                 |

---

## 🧠 Key Concepts Learned

- **PGD Attack**:  
  A stronger iterative attack than FGSM, projecting perturbed samples back into the epsilon-ball.

- **Attack Selector**:  
  User can now choose attack method via CLI (`--attack fgsm`, `--attack pgd`) — allowing for experimentation.

- **Clean Code Principles**:  
  Broke logic into `src/attacks`, `src/models`, `src/utils` for maintainability and reuse.

---

## 🚀 CLI Example Commands

```bash

# Run FGSM attack
python main.py --attack fgsm 

# Run PGD attack
python main.py --attack pgd