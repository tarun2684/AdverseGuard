# test_model_accuracy.py
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.models.model_loader import load_model

from src.data_loader import get_data_loaders
import torch

model = load_model()
model.load_state_dict(torch.load("models/mnist_cnn.pth"))
model.eval()

_, test_loader = get_data_loaders()

correct = 0
total = 0

for images, labels in test_loader:
    outputs = model(images)
    _, predicted = torch.max(outputs.data, 1)
    total += labels.size(0)
    correct += (predicted == labels).sum().item()

print(f"Clean accuracy: {100 * correct / total:.2f}%")
