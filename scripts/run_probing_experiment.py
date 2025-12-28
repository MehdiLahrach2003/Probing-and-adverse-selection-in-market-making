from __future__ import annotations

from pathlib import Path
import copy

import matplotlib.pyplot as plt
import yaml

from optimal_quoting.backtest.engine import MMParams, run_mm_toy
from optimal_quoting.calibration.dataset import build_intensity_dataset_from_mm
from optimal_quoting.calibration.mle import fit_intensity_exp_mle


def load_mm_params(path: str) -> MMParams:
    cfg = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    return MMParams(
        dt=float(cfg["dt"]),
        T=float(cfg["T"]),
        mid0=float(cfg["mid0"]),
        sigma=float(cfg["sigma"]),
        A=float(cfg["intensity"]["A"]),
        k=float(cfg["intensity"]["k"]),
        base_spread=float(cfg["strategy"]["base_spread"]),
        phi=float(cfg["strategy"]["phi"]),
        order_size=float(cfg["strategy"]["order_size"]),
        fee_bps=float(cfg["costs"]["fee_bps"]),
        seed=int(cfg["seed"]),
    )


def main() -> None:
    pcfg = yaml.safe_load(Path("configs/probing.yaml").read_text(encoding="utf-8"))
    base_cfg_path = pcfg["base_config"]

    base = load_mm_params(base_cfg_path)
    probing = copy.deepcopy(base)

    probing_cfg = pcfg["probing"]
    probing = MMParams(**{**probing.__dict__,
                          "probing_p": float(probing_cfg["p_explore"]),
                          "probing_jitter": float(probing_cfg["jitter"]),
                          "probing_widen_only": bool(probing_cfg["widen_only"])})
    calib_cfg = pcfg["calibration"]
    kmin, kmax = calib_cfg["k_bounds"]
    grid_size = int(calib_cfg["grid_size"])

    runs = []
    for name, params in [("baseline", base), ("probing", probing)]:
        df = run_mm_toy(params)
        delta, n = build_intensity_dataset_from_mm(df, dt=params.dt)
        est = fit_intensity_exp_mle(delta, n, dt=params.dt, k_bounds=(float(kmin), float(kmax)), grid_size=grid_size)
        runs.append((name, est.A, est.k, est.nll))

    print("=== True ===")
    print(f"A_true={base.A:.4f}, k_true={base.k:.4f}")
    print("=== Estimates ===")
    for name, Ahat, khat, nll in runs:
        print(f"{name:8s}  A_hat={Ahat:.4f}  k_hat={khat:.4f}  nll={nll:.1f}")

    Path("reports/figures").mkdir(parents=True, exist_ok=True)

    # simple bar plot for k_hat comparison
    labels = [r[0] for r in runs]
    ks = [r[2] for r in runs]

    plt.figure()
    plt.bar(labels, ks)
    plt.axhline(base.k, linestyle="--")
    plt.title("k_hat: baseline vs probing")
    plt.ylabel("k_hat")
    plt.tight_layout()
    plt.savefig("reports/figures/probing_khat_comparison.png")
    plt.close()

    print("Saved reports/figures/probing_khat_comparison.png")


if __name__ == "__main__":
    main()
