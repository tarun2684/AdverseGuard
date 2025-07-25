import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.attacks.pgd import pgd_attack
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch
from src.models.model_loader import load_model
model = load_model()
import matplotlib.pyplot as plt

# Load model

model.eval()
model.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(model.device)

# Load data
transform = transforms.ToTensor()
testset = datasets.MNIST(root='./data', train=False, transform=transform, download=True)
loader = DataLoader(testset, batch_size=1, shuffle=True)
images, labels = next(iter(loader))
images, labels = images.to(model.device), labels.to(model.device)

# Run PGD
adv_images = pgd_attack(model, images, labels, epsilon=0.3, alpha=0.01, num_iter=40)

# Visualize
def show_images(original, adversarial):
    original = original.cpu().squeeze().detach().numpy()
    adversarial = adversarial.cpu().squeeze().detach().numpy()
    fig, axs = plt.subplots(1, 2)
    axs[0].imshow(original, cmap="gray")
    axs[0].set_title("Original")
    axs[1].imshow(adversarial, cmap="gray")
    axs[1].set_title("Adversarial")
    plt.show()

show_images(images, adv_images)

# Before attack prediction
output_before_attack = model(images)
print("Prediction before attack:", output_before_attack.argmax(dim=1).item())

# Run PGD attack
adv_images = pgd_attack(model, images, labels, epsilon=0.3, alpha=0.01, num_iter=40)

# After attack prediction
output_after_attack = model(adv_images)
print("Prediction after attack:", output_after_attack.argmax(dim=1).item())
