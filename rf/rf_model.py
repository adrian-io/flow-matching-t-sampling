# rf/rf_model.py
"""
Rectified Flow implementation with integrated adaptive timestep sampler
and optional policy updates following the ATS paper notation.

Notation:
- model: v_theta(z_t, t, c)
- sampler: pi_phi(t | x0) = Beta(a(x0), b(x0))
- a, b are outputs of policy network
"""

import torch
import torch.nn as nn
from torch.distributions import Beta
from .utils import clamp_01

class RFWrapper:
    """
    RF training wrapper:
    - forwards: sample t via chosen sampling method
    - computes RF loss per-sample
    - stores policy-related tensors for later policy update (logprobs, entropy, a, b, t)
    """

    def __init__(self, model, policy=None, t_sampling="adaptive", timesteps=100, device="cpu"):
        self.model = model
        self.policy = policy            # PolicyNet or None
        self.t_sampling = t_sampling
        self.timesteps = timesteps
        self.device = device

        # buffers for last sampled policy outputs (set when t_sampling == 'adaptive')
        self.last_logprob = None
        self.last_entropy = None
        self.last_a = None
        self.last_b = None
        self.last_t = None

        # baseline for REINFORCE (running mean)
        self._baseline = None
        self.baseline_momentum = 0.95

    # -----------------------
    # t-sampling utilities
    # -----------------------
    def sample_t_policy(self, x):
        """
        Paper sampling: a,b ~ pi_phi(· | x0) ; t ~ Beta(a,b)
        Return t (B,), logprob (B,), entropy(B,), a(B,), b(B,)
        """
        if self.policy is None:
            raise ValueError("Adaptive sampling requires a policy network (pi_phi).")
        a, b = self.policy(x)         # both (B,)
        device = x.device

        if device.type == "mps":
            # Move a and b to CPU for fallback (Beta not supported on MPS)
            a_cpu = a.to("cpu")
            b_cpu = b.to("cpu")
            with torch.device("cpu"):
                dist = Beta(a_cpu, b_cpu)
                t = dist.sample()  # (B,) in (0,1)
                logprob = dist.log_prob(t)  # (B,)
                entropy = dist.entropy()    # (B,)
            # Move results back to MPS
            t = t.to(device)
            logprob = logprob.to(device)
            entropy = entropy.to(device)
        else:
            dist = Beta(a, b)
            t = dist.sample().to(device)  # continuous in (0,1)
            logprob = dist.log_prob(t)
            entropy = dist.entropy()

        # store
        self.last_logprob = logprob
        self.last_entropy = entropy
        self.last_a = a
        self.last_b = b
        self.last_t = t
        return t, logprob, entropy, a, b

    def t_sampling_method(self, b, device, x=None):
        """
        Non-adaptive sampling options from your earlier code plus 'adaptive' which uses pi_phi.
        Returns t as continuous in [0,1).
        """
        if self.t_sampling == "adaptive":
            if x is None:
                raise ValueError("Adaptive sampling requires x input")
            t, logprob, entropy, a, b = self.sample_t_policy(x)
            t = clamp_01(t)
            return t

        if self.t_sampling == "ln":
            nt = torch.randn((b,), device=device)
            t = torch.sigmoid(nt)
        elif self.t_sampling == "sinusoidal":
            u = torch.rand((b,), device=device)
            t = 0.5 + 0.5 * torch.sin(torch.pi * (u - 0.5))
        elif self.t_sampling == "quadratic":
            u = torch.rand((b,), device=device)
            t = u ** 2
        else:
            t = torch.rand((b,), device=device)

        t = clamp_01(t) # ensure t in [0,1)
        return t

    # -----------------------
    # forward & loss
    # -----------------------
    def forward(self, x, cond):
        """
        Perform one forward for a batch:
        - sample t (possibly adaptive)
        - form z_t = (1-t)x + t z1
        - compute v_theta and per-sample MSE
        returns: batch_loss (scalar), list[(t_item, per_sample_loss_value)], per_sample_tensor
        """
        b = x.size(0)
        t = self.t_sampling_method(b, x.device, x if self.t_sampling == "adaptive" else None)

        # broadcast t for image dims
        t_expanded = t.view([b, *([1] * (len(x.shape) - 1))])  # (B,1,1,1)
        z1 = torch.randn_like(x)
        zt = (1.0 - t_expanded) * x + t_expanded * z1 # t=0: x, t=1: z1
        vtheta = self.model(zt, t, cond)      # expect model handles t shape (B,) or (B,1)

        # per-sample mse across dims (exclude batch dim)
        reduce_dims = list(range(1, len(x.shape)))
        per_sample_mse = ((z1 - x - vtheta) ** 2).mean(dim=reduce_dims)  # (B,)
        batch_loss = per_sample_mse.mean()
        # loggable list
        tlist = per_sample_mse.detach().cpu().reshape(-1).tolist()
        ttloss = [(tv, tloss) for tv, tloss in zip(t.detach().cpu(), tlist)]
        return batch_loss, ttloss, per_sample_mse.detach()

    # -----------------------
    # baseline utilities
    # -----------------------
    def baseline_update(self, rewards):
        """
        rewards: Tensor (B,) or scalar
        maintain EMA baseline
        """
        r = float(rewards.mean().item()) if torch.is_tensor(rewards) else float(rewards)
        if self._baseline is None:
            self._baseline = r
        else:
            self._baseline = self.baseline_momentum * self._baseline + (1 - self.baseline_momentum) * r
        return self._baseline

    def baseline_value(self):
        return 0.0 if self._baseline is None else float(self._baseline)
