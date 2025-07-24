# Example (fgsm.py)
import torch
import torch.nn.functional as F
import logging
logging.basicConfig(level=logging.INFO)
logging.info("Loading MNIST dataset...")

def fgsm_attack(model, images, labels, epsilon):
    images.requires_grad = True
    outputs = model(images)
    loss = F.nll_loss(outputs, labels)
    model.zero_grad()
    loss.backward()
    data_grad = images.grad.data

    perturbed_image = images + epsilon * data_grad.sign()
    perturbed_image = torch.clamp(perturbed_image, 0, 1)
    return perturbed_image
