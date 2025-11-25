# implementation of Rectified Flow for simple minded people like me.
import argparse
import torch


class RF:
    def __init__(self, model, t_sampling="ln"):
        self.model = model
        self.t_sampling = t_sampling

    def t_sampling_method(self, b, device):
        """
        Implements t-sampling methods: uniform, logit-normal (ln), sinusoidal, and quadratic.

        Uniform:
            t ~ U(0, 1)
        Logit-Normal:
            t = sigmoid(nt), where nt ~ N(0, 1)
        Sinusoidal:
            t = 0.5 + 0.5 * sin(pi * (u - 0.5)), where u ~ U(0, 1)
        Quadratic:
            t = u^2, where u ~ U(0, 1)
        """
        if self.t_sampling == "ln":
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

        # Clip t values to [0, 1)
        t = torch.clamp(t, max=1 - 1e-6)
        return t

    def forward(self, x, cond):
        b = x.size(0)
        t = self.t_sampling_method(b, x.device)

        texp = t.view([b, *([1] * len(x.shape[1:]))])
        z1 = torch.randn_like(x)
        zt = (1 - texp) * x + texp * z1
        vtheta = self.model(zt, t, cond)
        batchwise_mse = ((z1 - x - vtheta) ** 2).mean(dim=list(range(1, len(x.shape))))
        tlist = batchwise_mse.detach().cpu().reshape(-1).tolist()
        ttloss = [(tv, tloss) for tv, tloss in zip(t, tlist)]
        return batchwise_mse.mean(), ttloss

    @torch.no_grad()
    def sample(self, z, cond, null_cond=None, sample_steps=50, cfg=2.0):
        b = z.size(0)
        dt = 1.0 / sample_steps
        dt = torch.tensor([dt] * b).to(z.device).view([b, *([1] * len(z.shape[1:]))])
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
    # train class conditional RF on mnist.
    import numpy as np
    import torch.optim as optim
    from PIL import Image
    from torch.utils.data import DataLoader
    from torchvision import datasets, transforms
    from torchvision.utils import make_grid
    from tqdm import tqdm

    import wandb
    from dit import DiT_Llama

    parser = argparse.ArgumentParser(description="use cifar?")
    parser.add_argument("--cifar", action="store_true")
    args = parser.parse_args()
    CIFAR = args.cifar

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
        ).to("mps")

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
            # channels, 32, dim=64, n_layers=6, n_heads=4, num_classes=10
            channels, 32, dim=64, n_layers=4, n_heads=2, num_classes=10
        ).to("mps")

    model_size = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Number of parameters: {model_size}, {model_size / 1e6}M")

    rf = RF(model, t_sampling="quadratic")
    LR = 5e-4
    optimizer = optim.Adam(model.parameters(), lr=LR)
    criterion = torch.nn.MSELoss()

    mnist = fdatasets(root="./data", train=True, download=True, transform=transform)
    BATCH_SIZE = 128
    # dataloader = DataLoader(mnist, batch_size=256, shuffle=True, drop_last=True)
    dataloader = DataLoader(mnist, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)

    wandb.init(project=f"rf_{dataset_name}", config={"model_size": model_size, "lr": LR, "batch_size": BATCH_SIZE})


    import csv

    # Initialize CSV file
    csv_file = "sample_log.csv"
    with open(csv_file, mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["epoch", "batch", "t_bin", "t", "t_loss"])

    for epoch in range(5):
        lossbin = {i: 0 for i in range(10)}
        losscnt = {i: 1e-6 for i in range(10)}
        for batch_idx, (x, c) in tqdm(enumerate(dataloader)):
            x, c = x.to("mps"), c.to("mps")
            optimizer.zero_grad()
            loss, blsct = rf.forward(x, c)
            loss.backward()
            
            # -------------------------------------------------
            # Collect gradient norms per t-bin
            # -------------------------------------------------
            gradients_t = []
            for p in model.parameters():
                if p.grad is not None:
                    gradients_t.append(p.grad.detach().abs().mean().item())
            # store as a single number or array
            grad_scalar = sum(gradients_t) / len(gradients_t)

            optimizer.step()

            # distribute loss into 10 equally sized t-bins
            for t, l in blsct:
                lossbin[int(t * 10)] += l
                losscnt[int(t * 10)] += 1
                
            # Log each sample's t, t_loss, and t_bin
            with open(csv_file, mode="a", newline="") as file:
                writer = csv.writer(file)
                for t, t_loss in blsct:
                    t_bin = int(t * 10)  # Calculate t_bin
                    writer.writerow([epoch, batch_idx, t_bin, t.item(), t_loss])

            # ------------------------------------------
            # UPGRADED W&B LOGGING PER EPOCH-BATCH
            # ------------------------------------------
            wandb.log({
                "epoch": epoch,
                "batch": batch_idx,
                # scalar batch loss
                "batch_loss": loss.item(),

                # sum of per-t-bin loss list for this batch
                "lossbin_t": [lossbin[i] for i in range(10)],
                # count per-t-bin for this batch
                "losscnt_t": [losscnt[i] for i in range(10)],
                # average per-t-bin loss list for this batch
                "loss_t": [(lossbin[i] / losscnt[i]) for i in range(10)],

                # gradient information
                "grad_norm_t": grad_scalar,     # if array: log list too
                # "grad_norm_per_param": gradients_t  # optional

                # expected loss reduction
                "elr_t": loss,
            })

        # log
        for i in range(10):
            print(f"Epoch: {epoch}, {i} range loss: {lossbin[i] / losscnt[i]}")

        # wandb.log({f"lossbin_{i}": lossbin[i] / losscnt[i] for i in range(10)})
   

        rf.model.eval()
        with torch.no_grad():
            cond = torch.arange(0, 16).to("mps") % 10
            uncond = torch.ones_like(cond) * 10

            init_noise = torch.randn(16, channels, 32, 32).to("mps")
            images = rf.sample(init_noise, cond, uncond)
            # image sequences to gif
            gif = []
            for image in images:
                # unnormalize
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
