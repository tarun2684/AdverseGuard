import torch
from src.evaluation.evaluate_impact import evaluate_model_impact
from src.models.model_loader import load_model
from src.core.dataset_loader import get_dataloaders
from src.attacks.fgsm import fgsm_attack
import matplotlib.pyplot as plt
import numpy as np


# 1. Load model
model = load_model(pretrained=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# 2. Load data
train_loader, test_loader = get_dataloaders(dataset='MNIST', batch_size=1)

# 3. Pick one sample
data_iter = iter(test_loader)
images, labels = next(data_iter)
images, labels = images.to(device), labels.to(device)

# 4. FGSM Attack
epsilon = 0.2
adv_images = fgsm_attack(model, images, labels, epsilon)

# 5. Predictions before and after
model.eval()
with torch.no_grad():
    orig_pred = model(images).argmax(dim=1)
    adv_pred = model(adv_images).argmax(dim=1)

print(f"Original Label: {labels.item()}, Prediction: {orig_pred.item()}")
print(f"Adversarial Prediction: {adv_pred.item()}")

# 6. Visualize

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
show(images[0], title="Original Image")
show(adv_images[0], title="Adversarial Image")
evaluate_model_impact(model, test_loader, epsilon=0.3)

