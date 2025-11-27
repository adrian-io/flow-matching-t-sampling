# train.py
"""
Main training script integrating ATS into Rectified Flow.
- reads config.yaml
- builds model, policy (optional)
- trains model and optionally the policy with REINFORCE
- supports 'use_true_delta' (compute loss reduction) or 'use_approx_delta' (instant -L)
- supports gradient-variance diagnostics (norm or full)
"""

import os
import yaml
import argparse
import random
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.utils import make_grid
from PIL import Image
from tqdm import tqdm

# adapt to your repo: replace DiT_Llama with your model import
from dit import DiT_Llama

from rf.rf_model import RFWrapper
from rf.policy import PolicyNet

# ---------------------
# Helper functions
# ---------------------
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def load_config(path):
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)
    return cfg

def build_model(cfg, device):
    if cfg["dataset"] == "cifar":
        channels = 3
    else:
        channels = cfg["channels"]

    model = DiT_Llama(channels, cfg["image_size"], dim=cfg["model_dim"],
                      n_layers=cfg["model_layers"], n_heads=cfg["model_heads"],
                      num_classes=10).to(device)
    return model

# ---------------------
# Utility to compute true Delta_t^k (expensive)
# ---------------------
def compute_true_delta(rf_wrapper, x_batch, cond_batch, model_optimizer, device, cfg):
    """
    Compute Delta_t^k = (1/T) sum_tau [ L_tau(theta_k) - L_tau(theta_k+1) ]
    following paper Eq. (12).
    This function performs:
      - L_before for selected taus (or full grid)
      - simulate one gradient update on theta using sampled t
      - evaluate L_after and compute delta
    Warning: expensive. Use with flag use_true_delta=True.
    """
    # 1. evaluate L_tau(theta_k) for subset S_grid (we use cfg['eval_T'] bins or discrete timesteps)
    T_eval = cfg.get("eval_T", 100)
    t_grid = np.linspace(0.0, 1.0, T_eval, endpoint=False)

    model = rf_wrapper.model
    model.eval()
    with torch.no_grad():
        L_before = []
        for tt in t_grid:
            t_tensor = torch.tensor([tt] * x_batch.size(0), dtype=torch.float32).to(device)
            texp = t_tensor.view(x_batch.size(0), *([1] * (len(x_batch.shape) - 1)))
            z1 = torch.randn_like(x_batch).to(device)
            zt = (1 - texp) * x_batch + texp * z1
            v = model(zt, t_tensor, cond_batch)
            reduce_dims = list(range(1, len(x_batch.shape)))
            per_sample_mse = ((z1 - x_batch - v) ** 2).mean(dim=reduce_dims)
            L_before.append(per_sample_mse.mean().item())
    L_before = np.array(L_before)  # shape (T_eval,)

    # 2. simulate a single model update using the sampled t from rf_wrapper.last_t
    # We need to compute gradient of L_t (for sampled t) and step the optimizer.
    # Save current params, perform one update, then evaluate L_after and restore params.
    model.train()
    saved_state = {k: v.clone().detach() for k, v in model.state_dict().items()}
    # compute sampled t forward/backward

    # Recompute t_sampled for the current batch size
    t_sampled = rf_wrapper.t_sampling_method(x_batch.size(0), device, x_batch if rf_wrapper.t_sampling == "adaptive" else None)
    texp = t_sampled.view([t_sampled.size(0)] + [1] * (len(x_batch.shape) - 1))  # Ensure correct shape
    z1 = torch.randn_like(x_batch).to(device)
    zt = (1 - texp) * x_batch + texp * z1
    v = model(zt, t_sampled, cond_batch)
    reduce_dims = list(range(1, len(x_batch.shape)))
    per_sample_mse = ((z1 - x_batch - v) ** 2).mean(dim=reduce_dims)
    loss_sampled = per_sample_mse.mean()
    loss_sampled.backward()
    model_optimizer.step()

    # 3. evaluate L_after on same grid
    model.eval()
    with torch.no_grad():
        L_after = []
        for tt in t_grid:
            t_tensor = torch.tensor([tt] * x_batch.size(0), dtype=torch.float32).to(device)
            texp = t_tensor.view(x_batch.size(0), *([1] * (len(x_batch.shape) - 1)))
            z1 = torch.randn_like(x_batch).to(device)
            zt = (1 - texp) * x_batch + texp * z1
            v = model(zt, t_tensor, cond_batch)
            per_sample_mse = ((z1 - x_batch - v) ** 2).mean(dim=reduce_dims)
            L_after.append(per_sample_mse.mean().item())
    L_after = np.array(L_after)

    # restore model params
    model.load_state_dict(saved_state)
    delta_vec = L_before - L_after  # (T_eval,)
    # following paper: Delta_t^k = (1/T) sum_tau delta_{k, tau}
    Delta_tk = float(delta_vec.mean())
    return Delta_tk, delta_vec, t_grid

# # ---------------------
# # Gradient variance utilities
# # ---------------------
# def grad_variance_for_t(model, dataloader, device, t_scalar, mode="norm", batches=4):
#     """
#     Estimate gradient variance for a fixed t.
#     mode: "norm" -> return variance of gradient norms (scalar)
#           "full" -> return variance across flattened gradient vectors (returns scalar as norm of variance vector)
#     """
#     grads = []
#     losses = []
#     model.train()
#     for i, (x, c) in enumerate(dataloader):
#         if i >= batches:
#             break
#         x, c = x.to(device), c.to(device)
#         model.zero_grad()
#         b = x.size(0)
#         t_tensor = torch.tensor([t_scalar] * b, device=device)
#         texp = t_tensor.view([b, *([1] * (len(x.shape) - 1))])
#         z1 = torch.randn_like(x)
#         zt = (1 - texp) * x + texp * z1
#         v = model(zt, t_tensor, c)
#         reduce_dims = list(range(1, len(x.shape)))
#         per_sample_mse = ((z1 - x - v) ** 2).mean(dim=reduce_dims)
#         loss = per_sample_mse.mean()
#         loss.backward()
#         # collect flattened grads vector
#         grad_list = []
#         for p in model.parameters():
#             if p.grad is not None:
#                 grad_list.append(p.grad.detach().cpu().view(-1))
#         if len(grad_list) == 0:
#             flat = torch.zeros(1)
#         else:
#             flat = torch.cat(grad_list)
#         grads.append(flat.numpy())   # numpy arrays may vary in length across models -> flatten consistent
#         losses.append(loss.item())
#     import numpy as np
#     G = np.stack([g if g.shape == grads[0].shape else np.pad(g, (0, grads[0].shape[0]-g.shape[0]), mode='constant') for g in grads])
#     if mode == "norm":
#         norms = np.linalg.norm(G, axis=1)
#         return float(np.var(norms)), float(np.mean(losses))
#     else:
#         # full variance vector, reduce to a scalar (norm of variance vector)
#         var_vec = np.var(G, axis=0)
#         return float(np.linalg.norm(var_vec)), float(np.mean(losses))

# ---------------------
# TRAIN LOOP
# ---------------------
def main(config_path):
    cfg = load_config(config_path)
    # device
    if cfg["device"] == "auto":
        device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    else:
        device = cfg["device"]

    set_seed(cfg["seed"])

    # Data
    if cfg["dataset"] == "cifar":
        fdatasets = datasets.CIFAR10
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.RandomCrop(cfg["image_size"]),
            transforms.RandomHorizontalFlip(),
            transforms.Normalize((0.5,), (0.5,))
        ])
        channels = 3
    else:
        fdatasets = datasets.MNIST
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Pad(2),
            transforms.Normalize((0.5,), (0.5,))
        ])
        channels = cfg["channels"]

    trainset = fdatasets(root="./data", train=True, download=True, transform=transform)
    dataloader = DataLoader(trainset, batch_size=cfg["batch_size"], shuffle=True, drop_last=True)

    # model and policy
    model = build_model(cfg, device)
    policy_net = None
    if cfg["t_sampling"] == "adaptive":
        policy_net = PolicyNet(in_channels=channels, hidden=cfg["policy_hidden"]).to(device)

    rf = RFWrapper(model, policy=policy_net, t_sampling=cfg["t_sampling"], timesteps=cfg["timesteps"], device=device)

    # optimizers
    optimizer = optim.Adam(model.parameters(), lr=cfg["lr_model"])
    policy_optimizer = optim.Adam(policy_net.parameters(), lr=cfg["lr_policy"]) if policy_net is not None else None

    # csv logging
    os.makedirs(cfg["contents_dir"], exist_ok=True)
    csv_file = cfg["log_csv"]
    import csv
    with open(csv_file, mode="w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "batch", "t_bin", "t", "t_loss"])

    # main loop
    for epoch in range(cfg["epochs"]):
        lossbin = {i: 0.0 for i in range(10)}
        losscnt = {i: 1e-6 for i in range(10)}
        for batch_idx, (x, c) in tqdm(enumerate(dataloader), total=len(dataloader)):
            x, c = x.to(device), c.to(device)
            optimizer.zero_grad()
            # forward
            batch_loss, blsct, per_sample_mse = rf.forward(x, c)
            # backward model update
            batch_loss.backward()
            # collect grad-norm scalar proxy
            # gradients_t = []
            # for p in model.parameters():
            #     print(p.shape)  # debug print
            #     print(p.grad.shape)  # debug print
            #     print(p.grad.detach().abs().mean().item())  # debug print
            #     if p.grad is not None:
            #         gradients_t.append(p.grad.detach().abs().mean().item())
            # print(len(gradients_t))
            # grad_scalar = sum(gradients_t) / (len(gradients_t) + 1e-12)
            optimizer.step()

            # Policy update (REINFORCE)
            do_policy_update = (cfg["t_sampling"] == "adaptive") and ((batch_idx % cfg["update_policy_every"]) == 0)
            if do_policy_update:
                # Option 1: true Delta (expensive)
                if cfg["use_true_delta"]:
                    # use a small subset / one sample to approximate Delta as in Algorithm 2
                    # sample one x0 from current batch to compute Delta via compute_true_delta (which itself simulates an update)
                    x0 = x[: min(8, x.size(0))]   # small subset
                    cond0 = c[: min(8, c.size(0))]
                    Delta_tk, delta_vec, t_grid = compute_true_delta(rf, x0, cond0, optimizer, device, cfg)
                    reward_batch = torch.tensor([Delta_tk] * x.size(0), device=device)  # broadcast scalar
                elif cfg["use_approx_delta"]:
                    # cheap instantaneous reward: negative per-sample loss
                    reward_batch = (-per_sample_mse).detach()  # (B,)
                else:
                    # fallback: negative mean loss
                    reward_batch = (-batch_loss.detach()).repeat(x.size(0)).to(device)

                # baseline update & advantage
                baseline = rf.baseline_update(reward_batch)
                advantage = (reward_batch - baseline).detach()  # (B,)

                # Recompute log_probs and entropy for the current batch size
                if rf.t_sampling == "adaptive":
                    t, log_probs, entropy, _, _ = rf.sample_t_policy(x)
                else:
                    log_probs = None
                    entropy = None

                if (log_probs is None) or (entropy is None):
                    # if missing, skip policy update
                    pass
                else:
                    # policy loss: - (advantage * log_prob).mean() - entropy_coef * entropy.mean()
                    policy_loss = -(log_probs * advantage).mean() - cfg["policy_entropy_coef"] * entropy.mean()
                    policy_optimizer.zero_grad()
                    policy_loss.backward()
                    policy_optimizer.step()

                    # optional logging (wandb omitted for brevity)
                    # print metrics
                    if batch_idx % 100 == 0:
                        print(f"Epoch {epoch} B {batch_idx} policy_loss {policy_loss.item():.6f} adv_mean {advantage.mean().item():.6f} ent {entropy.mean().item():.6f}")

            # distribute loss into bins for CSV/wandb logging
            import csv
            with open(csv_file, mode="a", newline="") as f:
                writer = csv.writer(f)
                for t_val, t_loss in blsct:
                    bin_idx = int(min(max(int(t_val * 10), 0), 9))
                    lossbin[bin_idx] += t_loss
                    losscnt[bin_idx] += 1
                    writer.writerow([epoch, batch_idx, bin_idx, t_val.item(), t_loss])

        # end epoch prints
        for i in range(10):
            print(f"Epoch: {epoch}, bin {i} avg loss: {lossbin[i] / losscnt[i]}")

        # run gradient-variance & loss evaluation optionally per epoch
        # create a small eval loader using a subset
        eval_subset = torch.utils.data.Subset(trainset, list(range(min(cfg["eval_subset_size"], len(trainset)))))
        eval_loader = DataLoader(eval_subset, batch_size=cfg["batch_size"], shuffle=False)
        # evaluate grad variance across t grid
        import numpy as np
        t_grid = np.linspace(0.0, 1.0, cfg["eval_T"], endpoint=False)
        grad_var = []
        loss_mean = []
        for tt in t_grid:
            gv, lm = grad_variance_for_t(model, eval_loader, device, float(tt), mode=cfg["grad_var_mode"], batches=cfg["grad_var_batches_per_t"])
            grad_var.append(gv)
            loss_mean.append(lm)
        # save small numpy arrays
        np.save(os.path.join(cfg["contents_dir"], f"grad_var_epoch_{epoch}.npy"), np.array(grad_var))
        np.save(os.path.join(cfg["contents_dir"], f"loss_mean_epoch_{epoch}.npy"), np.array(loss_mean))

        # sampling visualize (reuse earlier sample function)
        rf.model.eval()
        with torch.no_grad():
            cond = torch.arange(0, 16).to(device) % 10
            uncond = torch.ones_like(cond) * 10
            init_noise = torch.randn(16, channels, cfg["image_size"], cfg["image_size"]).to(device)
            images = rf.model.sample(init_noise, cond, uncond) if hasattr(rf.model, "sample") else []
            # try to save last (no assumption of sample function)
        rf.model.train()

    print("Training finished.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.yaml")
    args = parser.parse_args()
    main(args.config)
