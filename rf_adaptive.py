# implementation of Rectified Flow with adaptive Beta t-sampling (REINFORCE)
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import os

# Enable CPU fallback for unsupported MPS operations
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

class PolicyNet(nn.Module):
    """
    Small conv policy that maps image x -> (alpha, beta) per sample.
    Outputs positive alpha/beta via softplus.
    """
    def __init__(self, in_channels, hidden_dim=64):
        super().__init__()
        # small conv block -> global pool -> linear head
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, hidden_dim, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1)  # global pooling to 1x1
        )
        self.fc = nn.Linear(hidden_dim, 2)  # outputs raw alpha, beta

    def forward(self, x):
        # x: (B, C, H, W)
        h = self.conv(x).view(x.size(0), -1)  # (B, hidden_dim)
        out = self.fc(h)                       # (B, 2)
        # ensure positivity: softplus + small eps
        out = F.softplus(out) + 1e-3
        alpha = out[:, 0].unsqueeze(-1)  # (B, 1)
        beta = out[:, 1].unsqueeze(-1)   # (B, 1)
        return alpha, beta


class RF:
    def __init__(self, model, policy=None, t_sampling="ln", timesteps=100, device="cpu"):
        self.model = model
        self.t_sampling = t_sampling
        self.timesteps = timesteps
        self.device = device

        # policy network (optional)
        self.policy = policy
        # buffers to keep last sampled info (filled when adaptive sampling used)
        self.last_log_probs = None
        self.last_entropy = None
        self.last_alpha = None
        self.last_beta = None
        self.last_t_raw = None  # raw Beta sample in (0,1)

    def t_sampling_method(self, b, device, x=None):
        """
        Implements t-sampling methods: uniform, logit-normal (ln), sinusoidal, quadratic,
        and 'adaptive' Beta-based sampling with a policy network.
        """
        if self.t_sampling == "adaptive":
            if self.policy is None:
                raise ValueError("Adaptive t-sampling requires a policy network (self.policy).")
            if x is None:
                raise ValueError("Adaptive t-sampling requires input x for the policy network.")

            # policy outputs (B,1) alpha and beta
            alpha, beta = self.policy(x)  # shapes (B,1)
            alpha = alpha.squeeze(-1)     # (B,)
            beta = beta.squeeze(-1)       # (B,)

            if device.type == "mps":
                # Move alpha and beta to CPU for fallback
                alpha_cpu = alpha.to("cpu")
                beta_cpu = beta.to("cpu")
                with torch.device("cpu"):
                    dist = torch.distributions.Beta(alpha_cpu, beta_cpu)
                    t = dist.sample()  # (B,) in (0,1)
                    log_probs = dist.log_prob(t)  # (B,)
                    entropy = dist.entropy()      # (B,)
                t = t.to(device)  # Move back to MPS
                log_probs = log_probs.to(device)  # Move back to MPS
                entropy = entropy.to(device)      # Move back to MPS
            else:
                dist = torch.distributions.Beta(alpha, beta)
                t = dist.sample()  # (B,) in (0,1)
                log_probs = dist.log_prob(t)  # (B,)
                entropy = dist.entropy()      # (B,)

            # save for policy update
            self.last_log_probs = log_probs
            self.last_entropy = entropy
            self.last_alpha = alpha
            self.last_beta = beta
            self.last_t_raw = t

        elif self.t_sampling == "ln":
            nt = torch.randn((b,)).to(device)
            t = torch.sigmoid(nt)  # Logit-normal distribution

        elif self.t_sampling == "sinusoidal":
            u = torch.rand((b,)).to(device)
            t = 0.5 + 0.5 * torch.sin(torch.pi * (u - 0.5))  # Sinusoidal distribution

        elif self.t_sampling == "quadratic":
            u = torch.rand((b,)).to(device)
            t = u ** 2  # Quadratic distribution

        else:  # Default to uniform
            t = torch.rand((b,)).to(device)  # Uniform distribution

        # Clip to [0, 1)
        t = torch.clamp(t, max=1 - 1e-6)
        return t

    def forward(self, x, cond):
        """
        Single forward batch for RF:
          - sample t according to configured sampler
          - form z_t = (1-t) * x + t * z1
          - predict vtheta = model(zt, t, cond)
          - compute per-sample mse: ((z1 - x - vtheta)^2).mean over non-batch dims
        Returns:
          - batch_loss (scalar)
          - list of (t_value, per_sample_loss) for logging/CSV
        """
        b = x.size(0)
        # pass x into t sampler when adaptive (policy wants x)
        t = self.t_sampling_method(b, x.device, x if self.t_sampling == "adaptive" else None)

        # reshape t to broadcast to image dims
        texp = t.view([b, *([1] * (len(x.shape) - 1))])  # shape (B,1,1,1) for image

        z1 = torch.randn_like(x)                          # Gaussian noise prior
        zt = (1 - texp) * x + texp * z1                   # interpolated point

        # model expects t as shape (B,) scalar or (B,1) — we pass (B,)
        vtheta = self.model(zt, t, cond)                  # model output same shape as x

        # per-sample MSE across dims excluding batch
        reduce_dims = list(range(1, len(x.shape)))
        batchwise_mse = ((z1 - x - vtheta) ** 2).mean(dim=reduce_dims)  # (B,)
        
        # prepare logging list
        tlist = batchwise_mse.detach().cpu().reshape(-1).tolist()
        ttloss = [(tv, tloss) for tv, tloss in zip(t.detach().cpu(), tlist)]

        return batchwise_mse.mean(), ttloss, batchwise_mse.detach()  # return per-sample mse tensor too

    @torch.no_grad()
    def sample(self, z, cond, null_cond=None, sample_steps=50, cfg=2.0):
        b = z.size(0)
        dt = 1.0 / sample_steps
        dt = torch.tensor([dt] * b).to(z.device).view([b, *([1] * (len(z.shape) - 1))])
        images = [z]
        for i in range(sample_steps, 0, -1):
            t = i / sample_steps
            t = torch.tensor([t] * b).to(z.device)

            vc = self.model(z, t, cond)
            if null_cond is not None:
                vu = self.model(z, t, null_cond)
                vc = vu + cfg * (vc - vu)

            z = z - dt * vc
            images.append(z)
        return images


if __name__ == "__main__":
    # train class conditional RF on mnist/cifar with adaptive policy
    import numpy as np
    import torch.optim as optim
    from PIL import Image
    from torch.utils.data import DataLoader
    from torchvision import datasets, transforms
    from torchvision.utils import make_grid
    from tqdm import tqdm

    import wandb
    from dit import DiT_Llama  # user model

    parser = argparse.ArgumentParser(description="use cifar?")
    parser.add_argument("--cifar", action="store_true")
    parser.add_argument("--adaptive", default=True, action="store_true", help="use adaptive Beta t-sampling")
    args = parser.parse_args()
    CIFAR = args.cifar
    USE_ADAPTIVE = args.adaptive

    device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    if CIFAR:
        dataset_name = "cifar"
        fdatasets = datasets.CIFAR10
        transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.RandomCrop(32),
                transforms.RandomHorizontalFlip(),
                transforms.Normalize((0.5,), (0.5,)),
            ]
        )
        channels = 3
        model = DiT_Llama(
            channels, 32, dim=256, n_layers=10, n_heads=8, num_classes=10
        ).to(device)

    else:
        dataset_name = "mnist"
        fdatasets = datasets.MNIST
        transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Pad(2),
                transforms.Normalize((0.5,), (0.5,)),
            ]
        )
        channels = 1
        model = DiT_Llama(
            channels, 32, dim=64, n_layers=4, n_heads=2, num_classes=10
        ).to(device)

    model_size = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Number of parameters: {model_size}, {model_size / 1e6}M")

    # create policy net if adaptive
    policy_net = None
    policy_optimizer = None
    policy_lr = 1e-4
    if USE_ADAPTIVE:
        policy_net = PolicyNet(channels).to(device)
        policy_optimizer = optim.Adam(policy_net.parameters(), lr=policy_lr)

    rf = RF(model, policy=policy_net, t_sampling=("adaptive" if USE_ADAPTIVE else "sinusoidal"), timesteps=100, device=device)

    LR = 5e-4
    optimizer = optim.Adam(model.parameters(), lr=LR)
    criterion = torch.nn.MSELoss()

    mnist = fdatasets(root="./data", train=True, download=True, transform=transform)
    BATCH_SIZE = 128
    dataloader = DataLoader(mnist, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)

    wandb.init(project=f"rf_{dataset_name}", config={"model_size": model_size, "lr": LR, "batch_size": BATCH_SIZE, "adaptive": USE_ADAPTIVE})

    import csv, os
    csv_file = "rf_adaptive.csv"
    # ensure folder
    os.makedirs("contents", exist_ok=True)
    with open(csv_file, mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["epoch", "batch", "t_bin", "t", "t_loss", "last_alpha", "last_beta"])

    # hyper for policy update
    entropy_coef = 0.01

    for epoch in range(5):
        lossbin = {i: 0.0 for i in range(10)}
        losscnt = {i: 1e-6 for i in range(10)}
        for batch_idx, (x, c) in tqdm(enumerate(dataloader)):
            x, c = x.to(device), c.to(device)
            optimizer.zero_grad()

            loss, blsct, per_sample_mse = rf.forward(x, c)  # per_sample_mse shape (B,)
            loss.backward()
            # collect gradient norm scalar for logging
            gradients_t = []
            for p in model.parameters():
                if p.grad is not None:
                    gradients_t.append(p.grad.detach().abs().mean().item())
            grad_scalar = sum(gradients_t) / (len(gradients_t) + 1e-12)

            optimizer.step()

            # ---------------------------
            # Policy update (REINFORCE)
            # ---------------------------
            if USE_ADAPTIVE:
                # reward: negative per-sample mse (lower mse -> higher reward)
                rewards = (-per_sample_mse).detach()  # shape (B,)

                # baseline: batch mean
                baseline = rewards.mean()

                # retrieve last log_probs & entropy from rf (set in t_sampling_method)
                log_probs = rf.last_log_probs  # (B,)
                entropy = rf.last_entropy      # (B,)
                if (log_probs is None) or (entropy is None):
                    # fallback - should not happen if adaptive sampling used
                    pass
                else:
                    # advantage
                    advantage = (rewards - baseline).detach()

                    # policy loss (maximize expected reward -> minimize -log_prob * advantage)
                    policy_loss = -(log_probs * advantage).mean() - entropy_coef * entropy.mean()

                    # update policy
                    policy_optimizer.zero_grad()
                    policy_loss.backward()
                    policy_optimizer.step()

                    # wandb logging for policy
                    wandb.log({
                        "policy_loss": policy_loss.item(),
                        "policy_entropy": entropy.mean().item(),
                        "policy_alpha_mean": rf.last_alpha.mean().item(),
                        "policy_beta_mean": rf.last_beta.mean().item(),
                    }, step=epoch * len(dataloader) + batch_idx)

            # distribute loss into 10 equally sized t-bins and log to csv
            for t_val, l in blsct:
                bin_idx = int(min(max(int(t_val * 10), 0), 9))
                lossbin[bin_idx] += l
                losscnt[bin_idx] += 1

            with open(csv_file, mode="a", newline="") as file:
                writer = csv.writer(file)
                for t_val, t_loss in blsct:
                    t_bin = int(min(max(int(t_val * 10), 0), 9))
                    writer.writerow([epoch, batch_idx, t_bin, t_val.item(), t_loss, rf.last_alpha.mean().item() if USE_ADAPTIVE else "", rf.last_beta.mean().item() if USE_ADAPTIVE else ""])

            # W&B logging
            wandb.log({
                "epoch": epoch,
                "batch": batch_idx,
                "batch_loss": loss.item(),
                "lossbin_t": [lossbin[i] for i in range(10)],
                "losscnt_t": [losscnt[i] for i in range(10)],
                "loss_t": [(lossbin[i] / losscnt[i]) for i in range(10)],
                "grad_norm_t": grad_scalar,
            }, step=epoch * len(dataloader) + batch_idx)

        # end epoch print
        for i in range(10):
            print(f"Epoch: {epoch}, bin {i} avg loss: {lossbin[i] / losscnt[i]}")

        # sample for visualization
        rf.model.eval()
        with torch.no_grad():
            cond = torch.arange(0, 16).to(device) % 10
            uncond = torch.ones_like(cond) * 10

            init_noise = torch.randn(16, channels, 32, 32).to(device)
            images = rf.sample(init_noise, cond, uncond)
            gif = []
            for image in images:
                image = image * 0.5 + 0.5
                image = image.clamp(0, 1)
                x_as_image = make_grid(image.float(), nrow=4)
                img = x_as_image.permute(1, 2, 0).cpu().numpy()
                img = (img * 255).astype(np.uint8)
                gif.append(Image.fromarray(img))

            gif[0].save(
                f"contents/sample_{epoch}.gif",
                save_all=True,
                append_images=gif[1:],
                duration=100,
                loop=0,
            )
            last_img = gif[-1]
            last_img.save(f"contents/sample_{epoch}_last.png")

        rf.model.train()

    print("Training finished.")
