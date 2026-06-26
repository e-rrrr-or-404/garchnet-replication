"""Sanity tests for Hansen skewed-t implementation."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from scipy.integrate import quad
from scipy.optimize import minimize
from scipy.stats import t as t_dist
from src.skewed_t import skewt_pdf, skewt_logpdf, skewt_nll, skewt_rvs


def test_pdf_integrates_to_one():
    for eta in [3.0, 5.0, 10.0]:
        for lam in [-0.5, 0.0, 0.3]:
            val, _ = quad(lambda x: skewt_pdf(x, eta, lam), -50, 50, limit=200)
            assert abs(val - 1.0) < 1e-3, f"PDF integral = {val} for eta={eta}, lam={lam}"
    print("[OK] PDF integrates to 1")


def test_zero_mean_unit_variance():
    for eta in [4.0, 8.0]:
        for lam in [-0.3, 0.0, 0.3]:
            mean, _ = quad(lambda x: x * skewt_pdf(x, eta, lam), -50, 50, limit=200)
            var, _ = quad(lambda x: x ** 2 * skewt_pdf(x, eta, lam), -50, 50, limit=200)
            assert abs(mean) < 1e-2, f"Mean = {mean} for eta={eta}, lam={lam}"
            assert abs(var - 1.0) < 1e-2, f"Var = {var} for eta={eta}, lam={lam}"
    print("[OK] Zero mean, unit variance")


def test_symmetric_reduces_to_t():
    """When lambda = 0, skewed-t pdf should equal a unit-variance Student-t pdf."""
    eta = 5.0
    xs = np.linspace(-3, 3, 20)
    skewt_vals = skewt_pdf(xs, eta, 0.0)
    scale = np.sqrt((eta - 2) / eta)  # makes Student-t have unit variance
    t_vals = t_dist.pdf(xs / scale, df=eta) / scale
    np.testing.assert_allclose(skewt_vals, t_vals, atol=1e-6)
    print("[OK] Symmetric case (lambda=0) matches Student-t")


def test_mle_recovers_parameters():
    """Generate samples with known (eta, lam); MLE should recover them."""
    np.random.seed(0)
    true_eta, true_lam = 6.0, 0.3
    samples = skewt_rvs(5000, true_eta, true_lam, seed=0)

    def neg_ll(params):
        eta, lam = params
        if eta <= 2.01 or abs(lam) >= 0.99:
            return 1e10
        return skewt_nll(samples, eta, lam)

    result = minimize(neg_ll, x0=[5.0, 0.0], method="Nelder-Mead")
    eta_hat, lam_hat = result.x
    print(f"   True: eta={true_eta}, lam={true_lam}")
    print(f"   MLE:  eta={eta_hat:.3f}, lam={lam_hat:.3f}")
    assert abs(eta_hat - true_eta) < 1.5, f"eta not recovered: got {eta_hat}"
    assert abs(lam_hat - true_lam) < 0.1, f"lambda not recovered: got {lam_hat}"
    print("[OK] MLE recovers parameters")


if __name__ == "__main__":
    test_pdf_integrates_to_one()
    test_zero_mean_unit_variance()
    test_symmetric_reduces_to_t()
    test_mle_recovers_parameters()
    print("\nAll skewed-t tests passed.")