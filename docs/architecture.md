# 🔧 Project Architecture: Adversarial ML CLI Framework

## 📦 Model Format Support

This framework supports loading the following model types:

| Format | Extension | Framework |
|--------|-----------|-----------|
| Pickle | `.pkl`    | Scikit-learn, XGBoost, LightGBM |
| PyTorch | `.pt` / `.pth` | PyTorch |
| Keras / TensorFlow | `.h5` | TensorFlow / Keras |

### Future Support (Planned)
- ONNX models
- HuggingFace Transformers

---

## 🧑‍💻 CLI Input Format

```bash
python src/cli.py --model ./data/sample_model.pkl \
                  --data ./data/test_data.csv \
                  --attack fgsm --attack pgd \
                  --output ./results/report.json
