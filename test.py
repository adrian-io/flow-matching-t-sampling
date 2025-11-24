import torch
import matplotlib.pyplot as plt

def sample_t(dist_type, n=50000, device="cpu"):
    b = n
    if dist_type == "ln":
        nt = torch.randn((b,), device=device)
        t = torch.sigmoid(nt)  # Logit-normal distribution
    elif dist_type == "sinusoidal":
        u = torch.rand((b,), device=device)
        t = 0.5 + 0.5 * torch.sin(torch.pi * (u - 0.5))
    elif dist_type == "quadratic":
        u = torch.rand((b,), device=device)
        t = u ** 2
    else:
        t = torch.rand((b,), device=device)  # Uniform
    return t.cpu().numpy()

# Sample each distribution
t_uniform = sample_t("uniform")
t_ln = sample_t("ln")
t_sin = sample_t("sinusoidal")
t_quad = sample_t("quadratic")

# Plot all distributions
plt.figure(figsize=(10, 6))
plt.hist(t_uniform, bins=200, density=True, alpha=0.6, label="Uniform")
plt.hist(t_ln, bins=200, density=True, alpha=0.6, label="Logit-normal")
plt.hist(t_sin, bins=200, density=True, alpha=0.6, label="Sinusoidal")
plt.hist(t_quad, bins=200, density=True, alpha=0.6, label="Quadratic")
plt.legend()
plt.xlabel("t")
plt.ylabel("Density")
plt.title("Comparison of t-Sampling Distributions")
plt.show()
