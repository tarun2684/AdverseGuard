# src/attacks/bim.py

import torch
import torch.nn.functional as F

def bim_attack(model, images, labels, epsilon=0.3, alpha=0.01, num_iter=10):
    images = images.clone().detach().to(torch.float).to(model.device)
    labels = labels.to(model.device)
    original_images = images.clone().detach()

    for _ in range(num_iter):
        images.requires_grad = True
        outputs = model(images)
        loss = F.cross_entropy(outputs, labels)

        model.zero_grad()
        loss.backward()
        grad = images.grad.data

        images = images + alpha * grad.sign()
        perturbation = torch.clamp(images - original_images, min=-epsilon, max=epsilon)
        images = torch.clamp(original_images + perturbation, 0, 1).detach_()

    return images
