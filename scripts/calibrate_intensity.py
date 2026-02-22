from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import yaml

from optimal_quoting.backtest.engine import MMParams, run_mm_toy
from optimal_quoting.calibration.dataset import build_intensity_dataset_from_mm
from optimal_quoting.calibration.diagnostics import empirical_intensity_binned
from optimal_quoting.calibration.mle import fit_intensity_exp_mle, profile_nll_over_k


def main() -> None:
    cfg = yaml.safe_load(Path("configs/mm_toy.yaml").read_text(encoding="utf-8"))

    # ---- read mm_params block (matches your YAML structure)
    mm = cfg["mm_params"]

    p = MMParams(
        dt=float(mm.get("dt", 1.0)),
        T=float(mm.get("T", 20000.0)),
        mid0=float(mm.get("mid0", 100.0)),
        sigma=float(mm.get("sigma", 0.02)),

        # intensity true params (used by simulator)
        A=float(mm.get("A", 1.2)),
        k=float(mm.get("k", 1.0)),

        # quoting / strategy base params
        base_spread=float(mm.get("base_spread", 0.2)),
        phi=float(mm.get("phi", 0.0)),
        order_size=float(mm.get("order_size", 0.01)),

        # costs
        fee_bps=float(mm.get("fee_bps", 0.0)),

        # runtime
        seed=int(mm.get("seed", 123)),
    )

    # ---- run simulator and build intensity dataset
    df = run_mm_toy(p)
    delta, n = build_intensity_dataset_from_mm(df, dt=p.dt)

    # ---- calibration hyperparams from YAML (with defaults)
    cal = cfg.get("intensity_calibration", {})
    kb = cal.get("k_bounds", [0.0, 10.0])
    k_bounds = (float(kb[0]), float(kb[1])) if len(kb) >= 2 else (0.0, 10.0)
    grid_size = int(cal.get("grid_size", 300))

    est = fit_intensity_exp_mle(delta, n, dt=p.dt, k_bounds=k_bounds, grid_size=grid_size)

    print("=== True params ===")
    print(f"A_true={p.A:.6f}, k_true={p.k:.6f}")
    print("=== MLE estimate ===")
    print(f"A_hat={est.A:.6f}, k_hat={est.k:.6f}, nll={est.nll:.3f}")

    Path("reports/figures").mkdir(parents=True, exist_ok=True)

    # --- 1) true vs fitted intensity curve
    xs = np.linspace(0.0, float(np.quantile(delta, 0.995)), 200)
    lam_true = p.A * np.exp(-p.k * xs)
    lam_hat = est.A * np.exp(-est.k * xs)

    plt.figure()
    plt.plot(xs, lam_true, label="true")
    plt.plot(xs, lam_hat, label="fitted")
    plt.title("Intensity fit: λ(δ)=A exp(-kδ)")
    plt.xlabel("delta")
    plt.ylabel("lambda")
    plt.legend()
    plt.tight_layout()
    plt.savefig("reports/figures/intensity_fit.png")
    plt.close()

    # --- 2) empirical binned intensity vs fitted
    emp = empirical_intensity_binned(delta, n, dt=p.dt, nbins=40, dmax_quantile=0.995)
    lam_fit_centers = est.A * np.exp(-est.k * emp.bin_centers)

    plt.figure()
    plt.plot(emp.bin_centers, emp.lambda_hat, label="empirical (binned)")
    plt.plot(emp.bin_centers, lam_fit_centers, label="fitted")
    plt.title("Empirical intensity vs fitted")
    plt.xlabel("delta (bin centers)")
    plt.ylabel("lambda")
    plt.legend()
    plt.tight_layout()
    plt.savefig("reports/figures/intensity_empirical_fit.png")
    plt.close()

    # --- 3) profile likelihood for k
    k_grid = np.linspace(0.0, 5.0, 250)
    _, nlls = profile_nll_over_k(delta, n, dt=p.dt, k_grid=k_grid)
    nlls = nlls - np.nanmin(nlls)

    plt.figure()
    plt.plot(k_grid, nlls)
    plt.title("Profile NLL(k) (shifted)")
    plt.xlabel("k")
    plt.ylabel("NLL(k) - min")
    plt.tight_layout()
    plt.savefig("reports/figures/intensity_profile_nll.png")
    plt.close()

    print("Saved plots:")
    print(" - reports/figures/intensity_fit.png")
    print(" - reports/figures/intensity_empirical_fit.png")
    print(" - reports/figures/intensity_profile_nll.png")


if __name__ == "__main__":
    main()
