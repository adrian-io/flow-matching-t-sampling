# eval_rf_stats.py
import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

def evaluate_rf_gradient_variance_and_loss(
    rf,                        # RF object with rf.model
    eval_loader,               # DataLoader for evaluation (small subset)
    device,
    epochs_to_record,          # list of epoch indices to record (e.g. [0,1,2,...])
    T_eval=101,                # number of discrete t bins (0..T_eval-1)
    grad_batches_per_t=8,      # how many batches per t to use for gradient samples
):
    """
    Returns:
      grad_var: array shape (len(epochs_to_record), T_eval)
      loss_mean: array shape (len(epochs_to_record), T_eval)
      t_grid: array of continuous t in [0,1)
    """

    t_grid = np.linspace(0.0, 1.0, T_eval, endpoint=False)  # exclude 1.0
    grad_var = np.zeros((len(epochs_to_record), T_eval), dtype=float)
    loss_mean = np.zeros((len(epochs_to_record), T_eval), dtype=float)

    # Helper: compute gradient vector for a batch at a fixed t
    def grad_stat_for_batch(model, x, cond, t_scalar, device):
        # compute RF loss L_t for the batch and return a scalar gradient norm per-parameter
        model.zero_grad()
        b = x.size(0)
        t_tensor = torch.tensor([t_scalar] * b, device=device)
        # build z1 and zt as in RF forward
        z1 = torch.randn_like(x).to(device)
        texp = t_tensor.view(b, *([1] * (len(x.shape) - 1)))
        zt = (1.0 - texp) * x + texp * z1
        vtheta = model(zt, t_tensor, cond)
        reduce_dims = list(range(1, len(x.shape)))
        per_sample_mse = ((z1 - x - vtheta) ** 2).mean(dim=reduce_dims) # (B,)
        loss_batch = per_sample_mse.mean()
        # compute gradients
        loss_batch.backward()
        # collect a scalar gradient-statistic: we use flattened gradient norm
        grads = []
        for p in model.parameters():
            if p.grad is not None:
                grads.append(p.grad.detach().view(-1).cpu().numpy())
        if len(grads) == 0:
            return 0.0, loss_batch.item()
        grads_flat = np.concatenate(grads)
        grad_norm = np.linalg.norm(grads_flat)  # scalar
        return grad_norm, loss_batch.item()

    # Loop over requested epochs (this function should be called at each epoch or after loading model state)
    for e_idx, epoch in enumerate(epochs_to_record):
        print(f"Evaluating epoch {epoch} (index {e_idx}) ...")
        # assume rf.model is already the model at that epoch
        model = rf.model
        model.eval()
        with torch.no_grad():
            # but we need grads -> temporarily allow grad
            pass
        # For each timestep compute gradient samples and loss
        for ti, t in enumerate(tqdm(t_grid, desc="timesteps")):
            grad_samples = []
            loss_samples = []
            # For gradient variance we need backward passes so switch to train/grad mode
            model.zero_grad()
            model.train()  # allow gradients
            batch_iter = iter(eval_loader)
            for bidx in range(grad_batches_per_t):
                try:
                    x, c = next(batch_iter)
                except StopIteration:
                    batch_iter = iter(eval_loader)
                    x, c = next(batch_iter)
                x, c = x.to(device), c.to(device)
                # compute grad norm for this batch at t
                gnorm, loss_val = grad_stat_for_batch(model, x, c, float(t), device)
                grad_samples.append(gnorm)
                loss_samples.append(loss_val)
                # zero grads for next step
                model.zero_grad()
            # compute variance of grad-norms and mean loss
            grad_var[e_idx, ti] = float(np.var(grad_samples))
            loss_mean[e_idx, ti] = float(np.mean(loss_samples))
            model.zero_grad()
            model.eval()

    return grad_var, loss_mean, t_grid

def plot_grad_and_loss_over_timesteps(grad_var, loss_mean, t_grid, epoch_colors=None, out_prefix="rf_ats"):
    """
    Create two-panel figure similar to paper Fig.2:
      top: gradient variance vs timesteps (colored by epoch)
      bottom: loss vs timesteps (colored by epoch)
    grad_var, loss_mean shape: (n_epochs, T_eval)
    """
    n_epochs = grad_var.shape[0]
    cmap = plt.get_cmap("viridis")
    colors = [cmap(i / max(1, n_epochs-1)) for i in range(n_epochs)]

    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    for e in range(n_epochs):
        axes[0].plot(np.arange(len(t_grid)), grad_var[e], color=colors[e], alpha=0.8)
        axes[1].plot(np.arange(len(t_grid)), loss_mean[e], color=colors[e], alpha=0.8)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=0, vmax=n_epochs-1))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=axes.ravel().tolist(), pad=0.01)
    cbar.set_label("epoch (0 = early, high = late)")
    axes[0].set_ylabel("Gradient variance (proxy)")
    axes[1].set_ylabel("RF loss mean")
    axes[1].set_xlabel("timestep index")
    axes[0].set_yscale("log")
    axes[1].set_yscale("log")
    plt.tight_layout()
    plt.savefig(f"{out_prefix}_grad_loss_by_t.png", dpi=200)
    plt.show()

# Example usage in training script (call at end of epoch or periodically):
# epochs_to_record = [0, 2, 5, 10, 15, 20]  # or every epoch indices you want
# grad_var, loss_mean, t_grid = evaluate_rf_gradient_variance_and_loss(rf, eval_loader, device, epochs_to_record, T_eval=100, grad_batches_per_t=4)
# plot_grad_and_loss_over_timesteps(grad_var, loss_mean, t_grid)
