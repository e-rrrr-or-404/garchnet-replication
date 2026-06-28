"""
GARCHNet: GARCH model where the conditional variance is parameterized by an LSTM.

The network is trained by minimizing the negative log-likelihood of the assumed
innovation distribution (normal, Student-t, or Hansen's skewed-t) — i.e. the
likelihood is the loss.

This module contains:
- The three NLL losses as differentiable PyTorch functions
- The GARCHNet nn.Module
- A train_garchnet() helper that fits one model on one window
- A forecast_var_garchnet() helper that produces the one-step-ahead VaR

Reference: Buczynski & Chlebus (2024), GARCHNet, Computational Economics.
"""
from __future__ import annotations

import math
from typing import Literal

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# -----------------------------------------------------------------------------
# Losses (negative log-likelihoods, summed over batch)
#
# All assume the innovation z_t = eps_t / sigma_t is from the named distribution
# standardized to zero mean and unit variance. The conditional variance sigma^2
# is the network output; for t and skewt, eta (df) and lambda (skew) are also
# network outputs (per-observation, per the paper Section 2.3).
# -----------------------------------------------------------------------------

LOG_2PI = math.log(2.0 * math.pi)


def nll_normal(eps: torch.Tensor, sigma2: torch.Tensor) -> torch.Tensor:
    """NLL for eps_t ~ N(0, sigma_t^2), per the paper eq. (6).

    Returns mean NLL (averaged over the batch) for stable optimization.
    """
    return 0.5 * (LOG_2PI + torch.log(sigma2) + eps.pow(2) / sigma2).mean()


def nll_student_t(eps: torch.Tensor, sigma2: torch.Tensor, eta: torch.Tensor) -> torch.Tensor:
    """NLL for eps_t = sigma_t * z_t where z_t ~ standardized Student-t(eta),
    per the paper eq. (7).

    eta is a per-observation tensor of degrees of freedom, eta > 2.
    """
    # Use torch.lgamma for differentiable log-gamma.
    log_const = (
        torch.lgamma((eta + 1.0) / 2.0)
        - torch.lgamma(eta / 2.0)
        - 0.5 * torch.log(math.pi * (eta - 2.0) * sigma2)
    )
    log_kernel = -((eta + 1.0) / 2.0) * torch.log1p(eps.pow(2) / (sigma2 * (eta - 2.0)))
    return -(log_const + log_kernel).mean()


def nll_skewed_t(
    eps: torch.Tensor,
    sigma2: torch.Tensor,
    eta: torch.Tensor,
    lam: torch.Tensor,
) -> torch.Tensor:
    """NLL for Hansen's skewed-t, per the paper eq. (8) using a, b, c from eq. (9).

    eta > 2, -1 < lam < 1, both per-observation.
    """
    log_c = (
        torch.lgamma((eta + 1.0) / 2.0)
        - torch.lgamma(eta / 2.0)
        - 0.5 * torch.log(math.pi * (eta - 2.0))
    )
    c = torch.exp(log_c)
    a = 4.0 * lam * c * (eta - 2.0) / (eta - 1.0)
    b = torch.sqrt(1.0 + 3.0 * lam.pow(2) - a.pow(2))

    sigma = torch.sqrt(sigma2)
    z = eps / sigma
    thresh = -a / b
    denom = torch.where(z < thresh, 1.0 - lam, 1.0 + lam)
    inner = 1.0 + (1.0 / (eta - 2.0)) * ((b * z + a) / denom).pow(2)
    log_pdf = torch.log(b) + torch.log(c) - torch.log(sigma) - ((eta + 1.0) / 2.0) * torch.log(inner)
    return -log_pdf.mean()


# -----------------------------------------------------------------------------
# The GARCHNet module
# -----------------------------------------------------------------------------

DistName = Literal["normal", "t", "skewt"]


class GARCHNet(nn.Module):
    """LSTM-based conditional variance specification.

    Architecture (matches the paper Figure 1):
      input  : sequence of past p log-returns, shape (batch, p, 1)
      LSTM   : hidden_size=100
      FC     : 100 -> 64 (ReLU)
      FC     : 64 -> 32 (ReLU)
      Heads  : 32 -> 1 (softplus -> sigma^2)
               32 -> 1 (softplus + 2 -> eta) if dist in {t, skewt}
               32 -> 1 (tanh -> lambda) if dist == skewt

    The 'p' sequence length is a hyperparameter analogous to the GARCH(p,q) lag count.
    """

    def __init__(
        self,
        dist: DistName = "normal",
        lstm_hidden: int = 100,
        fc1: int = 64,
        fc2: int = 32,
    ):
        super().__init__()
        if dist not in ("normal", "t", "skewt"):
            raise ValueError(f"dist must be 'normal', 't', or 'skewt'; got {dist!r}")
        self.dist = dist

        self.lstm = nn.LSTM(input_size=1, hidden_size=lstm_hidden, batch_first=True)
        self.fc = nn.Sequential(
            nn.Linear(lstm_hidden, fc1),
            nn.ReLU(),
            nn.Linear(fc1, fc2),
            nn.ReLU(),
        )
        self.sigma2_head = nn.Linear(fc2, 1)
        if dist in ("t", "skewt"):
            self.eta_head = nn.Linear(fc2, 1)
        if dist == "skewt":
            self.lam_head = nn.Linear(fc2, 1)

    def forward(self, x: torch.Tensor):
        """
        x: (batch, p, 1) tensor of past returns (in decimal units).
        Returns a dict with sigma2, and optionally eta and lam.
        """
        h, _ = self.lstm(x)               # (batch, p, hidden)
        h_last = h[:, -1, :]              # (batch, hidden)  -- last timestep
        features = self.fc(h_last)        # (batch, fc2)

        # softplus guarantees positivity. A tiny epsilon avoids log(0) downstream.
        sigma2 = F.softplus(self.sigma2_head(features)).squeeze(-1) + 1e-8
        out = {"sigma2": sigma2}

        if self.dist in ("t", "skewt"):
            # eta > 2 enforced via softplus(.) + 2
            out["eta"] = F.softplus(self.eta_head(features)).squeeze(-1) + 2.0 + 1e-4

        if self.dist == "skewt":
            # -1 < lam < 1 enforced via tanh; pull back from the boundary slightly.
            out["lam"] = 0.999 * torch.tanh(self.lam_head(features)).squeeze(-1)

        return out


# -----------------------------------------------------------------------------
# Training one model on one rolling window
# -----------------------------------------------------------------------------

def _make_sequences(returns: np.ndarray, p: int):
    """Build (X, y) where X[i] = returns[i:i+p], y[i] = returns[i+p].

    Returns torch tensors on CPU; caller moves to device.
    """
    n = len(returns) - p
    X = np.empty((n, p, 1), dtype=np.float32)
    y = np.empty(n, dtype=np.float32)
    for i in range(n):
        X[i, :, 0] = returns[i : i + p]
        y[i] = returns[i + p]
    return torch.from_numpy(X), torch.from_numpy(y)


def train_garchnet(
    returns: np.ndarray,
    dist: DistName = "normal",
    p: int = 20,
    epochs: int = 100,
    batch_size: int = 512,
    lr: float = 3e-4,
    seed: int = 1,
    device: str | None = None,
    verbose: bool = False,
):
    """Fit a GARCHNet on a single window of returns.

    Returns (model, scale_var) where scale_var is the variance used to rescale
    the network's output back to the original returns' scale.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    torch.manual_seed(seed)
    np.random.seed(seed)
    if device == "cuda":
        torch.cuda.manual_seed_all(seed)

    # Standardize: subtract sample mean (~0 for returns), divide by sample std.
    # This makes LSTM inputs O(1) and gradients propagate properly.
    train_mean = float(np.mean(returns))
    train_var = float(np.var(returns))
    train_std = float(np.sqrt(train_var))
    returns_scaled = (returns - train_mean) / train_std

    X_scaled, y_scaled = _make_sequences(returns_scaled, p)
    # The TARGET for the NLL must be in scaled units to match the predicted scaled sigma2
    X_scaled = X_scaled.to(device)
    y_scaled = y_scaled.to(device)

    model = GARCHNet(dist=dist).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    n = X_scaled.shape[0]
    for ep in range(epochs):
        perm = torch.randperm(n, device=device)
        running = 0.0
        for start in range(0, n, batch_size):
            idx = perm[start : start + batch_size]
            xb, yb = X_scaled[idx], y_scaled[idx]
            out = model(xb)
            sigma2 = out["sigma2"]
            if dist == "normal":
                loss = nll_normal(yb, sigma2)
            elif dist == "t":
                loss = nll_student_t(yb, sigma2, out["eta"])
            else:
                loss = nll_skewed_t(yb, sigma2, out["eta"], out["lam"])
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            opt.step()
            running += loss.item() * xb.shape[0]
        if verbose and (ep + 1) % 50 == 0:
            print(f"    epoch {ep+1:3d}  nll={running/n:.5f}")

    model.eval()
    # Stash the scaling info on the model so forecast_var can find it
    model._train_mean = train_mean
    model._train_var = train_var
    return model


def forecast_var_garchnet(
    model,
    last_p_returns: np.ndarray,
    alpha: float = 0.025,
) -> float:
    """One-step-ahead VaR forecast on the original return scale."""
    device = next(model.parameters()).device
    train_mean = getattr(model, "_train_mean", 0.0)
    train_var = getattr(model, "_train_var", 1.0)
    train_std = math.sqrt(train_var)

    # Scale the input the same way the training data was scaled
    last_p_scaled = (last_p_returns - train_mean) / train_std
    x = torch.from_numpy(last_p_scaled.astype(np.float32)).view(1, -1, 1).to(device)

    with torch.no_grad():
        out = model(x)
        sigma2_scaled = out["sigma2"].item()

    # Convert from scaled-return variance back to original-return variance
    sigma2_original = sigma2_scaled * train_var
    sigma = math.sqrt(sigma2_original)

    if model.dist == "normal":
        from scipy.stats import norm
        q = norm.ppf(alpha)
    elif model.dist == "t":
        eta = out["eta"].item()
        from scipy.stats import t as t_dist
        q = t_dist.ppf(alpha, df=eta) * math.sqrt((eta - 2.0) / eta)
    else:
        eta = out["eta"].item()
        lam = out["lam"].item()
        from src.skewed_t import skewt_cdf
        from scipy.optimize import brentq
        q = brentq(lambda z: skewt_cdf(z, eta, lam) - alpha, -15, 15)

    return train_mean + sigma * q