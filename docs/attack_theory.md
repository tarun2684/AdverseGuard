# FGSM (Fast Gradient Sign Method) — Adversarial Attack Theory

The **Fast Gradient Sign Method (FGSM)** is one of the simplest and most widely used adversarial attacks on neural networks. It was introduced in 2015 by Ian Goodfellow et al. in the paper *"Explaining and Harnessing Adversarial Examples."*

## 🧠 Core Idea

FGSM creates a small perturbation to the input image that is imperceptible to humans but causes the model to make a wrong prediction. The perturbation is calculated by **taking the sign of the gradient of the loss with respect to the input image**.

## 📐 Mathematical Formula

Let:
$$

- \( x \): Original input image  
- \( y \): True label  
- \( \theta \): Model parameters  
- \( J(\theta, x, y) \): Loss function  
- \( \epsilon \): Small scalar that controls the perturbation strength  
$$

Then, the adversarial image \( x_{adv} \) is:

\[
x_{adv} = x + \epsilon \cdot \text{sign}(\nabla_x J(\theta, x, y))
\]

- \( \nabla_x J(\theta, x, y) \) is the gradient of the loss with respect to the input.
- `sign(...)` gives the direction to perturb each pixel (positive or negative).

## 📊 Diagram

  ```python
     Clean Image (x)
            │
            ▼
  Compute Gradient: ∇ₓ J(θ, x, y)
            │
            ▼
 Take Sign: sign(∇ₓ J)
            │
            ▼
Scale Noise: ε · sign(∇ₓ J)
            │
            ▼
Adversarial Image: x_adv = x + ε · sign(∇ₓ J)
  ```


---

## 🧪 Practical Implications

- **Target**: Classification models (CNNs on images, etc.)
- **Use Case**: Evaluate how vulnerable a model is to slight, crafted perturbations
- **Detection**: Difficult for human eyes to spot differences
- **Defense**: Training with adversarial examples (Adversarial Training)

---

## ✅ Summary

| Property            | FGSM                                 |
|---------------------|--------------------------------------|
| Type                | White-box attack                     |
| Speed               | Very fast (1 gradient step)          |
| Perturbation        | Controlled by \( \epsilon \)         |
| Visual Distortion   | Typically imperceptible              |
| Goal                | Mislead model with minimal effort    |
| Defense Strategy    | Adversarial training, input smoothing|

---

## 📚 References

- Goodfellow, I. J., Shlens, J., & Szegedy, C. (2015). [Explaining and Harnessing Adversarial Examples](https://arxiv.org/abs/1412.6572)
- MIT 6.S191 Lecture Notes on Adversarial Examples

---

