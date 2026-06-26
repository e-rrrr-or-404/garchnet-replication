"""GARCH(1,1) baseline models with normal, t, and skewed-t innovations.

Wraps the `arch` package for the three GARCH baselines used in
Buczynski & Chlebus (2024). All returns are passed in as decimals
(e.g. 0.01 for 1%) and internally scaled to percent for numerical
stability in the optimizer.
"""
import numpy as np
import pandas as pd
from arch import arch_model
from scipy import stats

# arch's distribution name -> internal label
DIST_MAP = {
    "normal": "normal",
    "t": "t",
    "skewt": "skewt",
}


def fit_garch(returns: np.ndarray, dist: str = "normal", p: int = 1, q: int = 1):
    """
    Fit GARCH(p, q) on log returns with the chosen innovation distribution.

    Parameters
    ----------
    returns : np.ndarray
        1-d array of log returns (decimal, not percent).
    dist : {'normal', 't', 'skewt'}
        Innovation distribution.
    p : int
        Number of GARCH lags (sigma^2_{t-i}).
    q : int
        Number of ARCH lags (eps^2_{t-i}).

    Returns
    -------
    arch.univariate.base.ARCHModelResult
        Fitted model. Use .params and .forecast() on this.
    """
    if dist not in DIST_MAP:
        raise ValueError(f"dist must be one of {list(DIST_MAP)}, got {dist!r}")
    # arch's optimizer is much better-behaved with returns scaled to percent
    am = arch_model(returns * 100, mean="Zero", vol="GARCH",
                    p=p, q=q, dist=DIST_MAP[dist])
    res = am.fit(disp="off", show_warning=False)
    return res


def _skewt_quantile(alpha: float, eta: float, lam: float) -> float:
    """Numerical quantile of Hansen skewed-t (standardized, mean 0 var 1)."""
    from scipy.optimize import brentq
    from src.skewed_t import skewt_cdf  # written in next step
    # Search a wide bracket; -10 and +10 are well beyond any reasonable financial quantile
    return brentq(lambda x: skewt_cdf(x, eta, lam) - alpha, -15, 15)


def forecast_var(res, alpha: float = 0.025, use_skewt_helper: bool = True) -> float:
    """
    One-step-ahead Value-at-Risk forecast at level alpha.

    VaR_alpha = mu_t + sigma_t * F^{-1}(alpha)

    Since we use mean='Zero', mu_t = 0. Sigma is rescaled from percent back to decimal.
    """
    f = res.forecast(horizon=1, reindex=False)
    sigma_pct = np.sqrt(f.variance.values[-1, 0])
    sigma = sigma_pct / 100.0  # back to decimal

    dist_name = res.model.distribution.name.lower()

    dist_name = res.model.distribution.name.lower()

    if "normal" in dist_name or "gaussian" in dist_name:
        q = stats.norm.ppf(alpha)

    elif "skew" in dist_name:
        eta = res.params["eta"]
        lam = res.params["lambda"]
        if use_skewt_helper:
            q = _skewt_quantile(alpha, eta, lam)
        else:
            q = res.model.distribution.ppf(alpha, [eta, lam])

    elif "student" in dist_name or dist_name.strip() == "t":
        # arch v7 names it "Standardized Student's t"; v6 was "StudentsT"
        eta = res.params["nu"]
        q = res.model.distribution.ppf(alpha, [eta])

    else:
        raise ValueError(f"Unknown distribution name from arch: {dist_name!r}")

    return float(sigma * q)


def rolling_garch_var(returns: np.ndarray, n_train: int = 1000, n_test: int = 250,
                      dist: str = "normal", alpha: float = 0.025,
                      verbose: bool = False) -> np.ndarray:
    """
    Rolling-window one-step-ahead VaR forecasts.

    For each t in 0..n_test-1:
      train on returns[t : t+n_train]
      forecast sigma_{t+n_train}
      VaR forecast = sigma * F^{-1}(alpha)

    Returns array of length n_test.
    """
    if len(returns) < n_train + n_test:
        raise ValueError(f"Need {n_train + n_test} obs, got {len(returns)}")
    var_forecasts = np.zeros(n_test)
    for t in range(n_test):
        window = returns[t : t + n_train]
        try:
            res = fit_garch(window, dist=dist)
            var_forecasts[t] = forecast_var(res, alpha=alpha)
        except Exception as e:
            # If a single fit fails (rare), forward-fill from previous
            var_forecasts[t] = var_forecasts[t - 1] if t > 0 else np.nan
            if verbose:
                print(f"  fit failed at t={t}: {e}, forward-filled")
        if verbose and (t + 1) % 50 == 0:
            print(f"    {t+1}/{n_test} fits done")
    return var_forecasts