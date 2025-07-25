import torch
import torch.nn as nn
import torch.nn.functional as F

def pgd_attack(model, images, labels, epsilon, alpha, num_iter):
    # Ensure gradients are tracked
    images = images.clone().detach().to(images.device)
    labels = labels.to(images.device)
    images.requires_grad = True 

    ori_images = images.clone().detach()

    for _ in range(num_iter):
        outputs = model(images)
        loss = nn.functional.nll_loss(outputs, labels)

        model.zero_grad()
        loss.backward()
        # Check if gradients are available
        if images.grad is None:
            raise ValueError("Gradients are not available. Ensure `images.requires_grad=True` is set.")

        grad = images.grad.data

        images = images + alpha * grad.sign()
        perturbation = torch.clamp(images - ori_images, min=-epsilon, max=epsilon)
        images = torch.clamp(ori_images + perturbation, min=0, max=1).detach()
        images.requires_grad = True  # Re-enable grad tracking after detach

    return images
