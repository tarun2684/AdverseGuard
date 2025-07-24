from torch import device
import torch

from src.attacks.fgsm import fgsm_attack


def evaluate_model_impact(model, test_loader, epsilon):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    correct_orig = 0
    correct_adv = 0
    total = 0

    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)

        # Original prediction
        with torch.no_grad():
            output = model(images)
        pred = output.argmax(dim=1)
        correct_orig += (pred == labels).sum().item()

        # Generate adversarial image
        adv_images = fgsm_attack(model, images, labels, epsilon)

        # Adversarial prediction
        with torch.no_grad():
            adv_output = model(adv_images)
        adv_pred = adv_output.argmax(dim=1)
        correct_adv += (adv_pred == labels).sum().item()

        total += labels.size(0)

    orig_acc = correct_orig / total * 100
    adv_acc = correct_adv / total * 100

    print(f"Accuracy on clean data: {orig_acc:.2f}%")
    print(f"Accuracy on adversarial data: {adv_acc:.2f}%")
    print(f"Accuracy drop: {orig_acc - adv_acc:.2f}%")
