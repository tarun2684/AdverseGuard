
import matplotlib.pyplot as plt

def show_images(original, adversarial):
    original = original.cpu().squeeze().detach().numpy()
    adversarial = adversarial.cpu().squeeze().detach().numpy()
    
    fig, axs = plt.subplots(1, 2, figsize=(6, 3))
    axs[0].imshow(original, cmap="gray")
    axs[0].set_title("Original")
    
    axs[1].imshow(adversarial, cmap="gray")
    axs[1].set_title("Adversarial")
    
    plt.tight_layout()
    plt.show()
