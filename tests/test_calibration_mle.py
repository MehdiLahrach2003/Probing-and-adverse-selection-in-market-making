import numpy as np

from optimal_quoting.backtest.engine import MMParams, run_mm_toy
from optimal_quoting.calibration.dataset import build_intensity_dataset_from_mm
from optimal_quoting.calibration.mle import fit_intensity_exp_mle


def _run_calibration(dt: float, probing: bool, seed: int):
    """
    Small end-to-end calibration run for tests.
    Returns (est, params).
    """
    # Choose parameters consistent with your project toy setup
    T = 20000.0

    p = MMParams(
        dt=float(dt),
        T=T,
        mid0=100.0,
        sigma=0.006 if dt < 1.0 else 0.02,   # keep volatility reasonable when dt changes
        A=1.2,
        k=1.0,
        base_spread=0.4 if dt < 1.0 else 0.2,
        phi=0.0,
        order_size=0.01,
        fee_bps=0.0,
        seed=int(seed),
        policy="probing" if probing else "baseline",
        probing_p=0.20 if probing else 0.0,
        probing_jitter=0.80 if probing else 0.0,
        probing_widen_only=True,
    )

    df = run_mm_toy(p)
    delta, n = build_intensity_dataset_from_mm(df, dt=p.dt)

    est = fit_intensity_exp_mle(delta, n, dt=p.dt, k_bounds=(0.0, 5.0), grid_size=300)
    return est, p

def test_mle_recovers_params_with_probing():
    est, p = _run_calibration(dt=0.1, probing=True, seed=123)

    assert np.isfinite(est.A) and est.A > 0
    assert np.isfinite(est.k) and est.k > 0

    assert abs(est.A - p.A) < 0.3
    assert abs(est.k - p.k) < 0.3


def test_mle_is_biased_without_probing():
    est, p = _run_calibration(dt=1.0, probing=False, seed=123)

    assert np.isfinite(est.k)

    # We EXPECT bias in non-identifiable regime
    assert abs(est.k - p.k) > 0.2
