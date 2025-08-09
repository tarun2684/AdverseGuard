# Adversarial ML Red Team Toolkit

🧠 **Adversarial ML Red Team Toolkit** is a Python-based utility for testing machine learning models against common adversarial attacks in **computer vision** and **NLP** domains.  
It supports attacks such as **FGSM**, **PGD**, and **TextFooler**, helping you evaluate your model’s robustness under malicious perturbations.

---

## Features

- 🔍 **Multiple attack methods**: FGSM, PGD, TextFooler (extendable to more).
- 📊 **Performance evaluation** before and after attack.
- 🛠 **Plug-and-play** with scikit-learn, PyTorch, or TensorFlow models.
- 📦 **Customizable** attack parameters via CLI.
- 🧪 Works on **image** and **text** datasets.

---

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/adversarial-ml-redteam.git
cd adversarial-ml-redteam

# Create and activate a virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate   # On macOS/Linux
venv\Scripts\activate      # On Windows

# Install dependencies
pip install -r requirements.txt
