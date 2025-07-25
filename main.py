import argparse
import torch
import matplotlib.pyplot as plt
import numpy as np

from src.evaluation.evaluate_impact import evaluate_model_impact
from src.visualize.show_adversarial import show_images
from src.models.model_loader import load_model
from src.core.dataset_loader import get_dataloaders
from src.attacks.fgsm import fgsm_attack
from src.attacks.bim import bim_attack
from src.attacks.pgd import pgd_attack

def show(img, title="Image"):
    if isinstance(img, torch.Tensor):
        npimg = img.detach().cpu().squeeze().numpy()
    elif isinstance(img, np.ndarray):
        npimg = img.squeeze()
    else:
        raise TypeError("Unsupported image type: {}".format(type(img)))
    
    plt.imshow(npimg, cmap='gray')
    plt.title(title)
    plt.axis('off')
    plt.show()

def main(attack_type="fgsm", epsilon=0.2):
    # 1. Load model
    model = load_model(pretrained=True, model_path="models/mnist_cnn.pth")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # 2. Load data
    train_loader, test_loader = get_dataloaders(dataset='MNIST', batch_size=1)

    # 3. Pick one sample
    data_iter = iter(test_loader)
    images, labels = next(data_iter)
    images, labels = images.to(device), labels.to(device)

    # 4. Select attack
    if attack_type == "fgsm":
        adv_images = fgsm_attack(model, images, labels, epsilon)
    elif attack_type == "bim":
        adv_images = bim_attack(model, images, labels, epsilon, alpha=0.01, num_iter=10)
    elif attack_type == "pgd":
        adv_images = pgd_attack(model, images, labels, epsilon, alpha=0.01, num_iter=10)
    else:
        raise ValueError("Invalid attack type. Choose from: fgsm, bim, pgd")

    # 5. Predictions
    model.eval()
    with torch.no_grad():
        orig_pred = model(images).argmax(dim=1)
        adv_pred = model(adv_images).argmax(dim=1)

    print(f"Original Label: {labels.item()}, Prediction: {orig_pred.item()}")
    print(f"Adversarial Prediction: {adv_pred.item()}")

    # 6. Show images
    show(images[0], title="Original Image")
    show(adv_images[0], title=f"Adversarial Image ({attack_type.upper()})")

    # 7. Evaluate on full test set
    print("\nEvaluating model robustness on full test set:")
    evaluate_model_impact(model, test_loader, attack_type=attack_type, epsilon=epsilon)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Adversarial Attack CLI")
    parser.add_argument('--attack', type=str, default='fgsm', choices=['fgsm', 'bim', 'pgd'], help="Type of attack to run")
    parser.add_argument('--epsilon', type=float, default=0.2, help="Perturbation strength")

    args = parser.parse_args()
    main(attack_type=args.attack, epsilon=args.epsilon)
