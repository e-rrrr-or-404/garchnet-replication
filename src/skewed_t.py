"""
Hansen (1994) skewed-t distribution.

Implements pdf, log-pdf, cdf, and the negative log-likelihood (NLL) that
GARCHNet uses as its loss function for the skewed-t case.

The standardized Hansen skewed-t has zero mean and unit variance.

Parameters
----------
eta : float
    Degrees of freedom. eta > 2 required for finite variance.
lam : float
    Skewness. -1 < lam < 1 required.

Reference
---------
Hansen, B. E. (1994). Autoregressive Conditional Density Estimation.
International Economic Review, 35(3), 705-730.
"""
import numpy as np
from scipy.special import gammaln


def _abc(eta, lam):
    """Hansen's a, b, c reparameterization (eq. 9 of Buczynski & Chlebus 2024)."""
    c = np.exp(gammaln((eta + 1) / 2) - gammaln(eta / 2)) / np.sqrt(np.pi * (eta - 2))
    a = 4 * lam * c * (eta - 2) / (eta - 1)
    b = np.sqrt(1 + 3 * lam ** 2 - a ** 2)
    return a, b, c


def skewt_logpdf(x, eta, lam, sigma=1.0):
    """
    Log-pdf of Hansen skewed-t with scale sigma.

    x can be a scalar or 1-d numpy array. eta, lam are scalars.

    Note: 'sigma' here is the standard deviation, not the GARCH conditional
    variance. The pdf includes the -log(sigma) Jacobian.
    """
    a, b, c = _abc(eta, lam)
    x = np.asarray(x, dtype=float)
    z = x / sigma
    thresh = -a / b
    denom = np.where(z < thresh, 1 - lam, 1 + lam)
    inner = 1.0 + (1.0 / (eta - 2.0)) * ((b * z + a) / denom) ** 2
    return np.log(b) + np.log(c) - np.log(sigma) - ((eta + 1) / 2) * np.log(inner)


def skewt_pdf(x, eta, lam, sigma=1.0):
    return np.exp(skewt_logpdf(x, eta, lam, sigma))


def skewt_nll(x, eta, lam, sigma=1.0):
    """Negative log-likelihood, summed over x. Used as fit metric, not as loss
    inside a NN (PyTorch needs the differentiable torch version — that lives
    inside garchnet.py)."""
    return -float(np.sum(skewt_logpdf(x, eta, lam, sigma)))


def skewt_cdf(x, eta, lam, sigma=1.0):
    """CDF by numerical integration. Slow but correct. Used for VaR quantile."""
    from scipy.integrate import quad
    val, _ = quad(lambda t: skewt_pdf(t, eta, lam, sigma),
                  -50 * sigma, float(x), limit=200)
    return val


def skewt_rvs(n, eta, lam, sigma=1.0, seed=None):
    """Sample n draws via inverse-CDF on a precomputed grid."""
    rng = np.random.default_rng(seed)
    grid = np.linspace(-15 * sigma, 15 * sigma, 20000)
    pdf_vals = skewt_pdf(grid, eta, lam, sigma)
    # Trapezoidal CDF
    cdf_vals = np.concatenate([[0], np.cumsum(0.5 * (pdf_vals[:-1] + pdf_vals[1:]) * np.diff(grid))])
    cdf_vals /= cdf_vals[-1]
    u = rng.uniform(size=n)
    return np.interp(u, cdf_vals, grid)