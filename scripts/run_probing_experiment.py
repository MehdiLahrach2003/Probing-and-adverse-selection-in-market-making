from __future__ import annotations

from dataclasses import fields, replace
from pathlib import Path
import copy

import matplotlib.pyplot as plt
import yaml

from optimal_quoting.backtest.engine import MMParams, run_mm_toy
from optimal_quoting.calibration.dataset import build_intensity_dataset_from_mm
from optimal_quoting.calibration.mle import fit_intensity_exp_mle


def _mmparams_field_names() -> set[str]:
    return {f.name for f in fields(MMParams)}


def load_mm_params(path: str) -> MMParams:
    cfg = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    mm = cfg.get("mm_params", cfg)  # allow both styles just in case

    allowed = _mmparams_field_names()
    filtered = {k: v for k, v in mm.items() if k in allowed}

    # Type normalization
    for k in ("dt", "T", "mid0", "sigma", "A", "k", "base_spread", "phi", "order_size", "fee_bps", "gamma"):
        if k in filtered:
            filtered[k] = float(filtered[k])
    if "seed" in filtered:
        filtered["seed"] = int(filtered["seed"])
    if "probing_p" in filtered:
        filtered["probing_p"] = float(filtered["probing_p"])
    if "probing_jitter" in filtered:
        filtered["probing_jitter"] = float(filtered["probing_jitter"])
    if "probing_widen_only" in filtered:
        filtered["probing_widen_only"] = bool(filtered["probing_widen_only"])

    # Fallback defaults
    filtered.setdefault("dt", 1.0)
    filtered.setdefault("T", 20000.0)
    filtered.setdefault("mid0", 100.0)
    filtered.setdefault("sigma", 0.02)
    filtered.setdefault("A", 1.2)
    filtered.setdefault("k", 1.0)
    filtered.setdefault("base_spread", 0.2)
    filtered.setdefault("phi", 0.0)
    filtered.setdefault("order_size", 0.01)
    filtered.setdefault("fee_bps", 0.0)
    filtered.setdefault("seed", 0)

    return MMParams(**filtered)


def main() -> None:
    pcfg = yaml.safe_load(Path("configs/probing.yaml").read_text(encoding="utf-8"))
    base_cfg_path = pcfg["base_config"]

    base = load_mm_params(base_cfg_path)

    # Build probing variant (MMParams is frozen -> use dataclasses.replace)
    probing_cfg = pcfg["probing"]
    allowed = _mmparams_field_names()

    updates: dict[str, object] = {}
    if "probing_p" in allowed:
        updates["probing_p"] = float(probing_cfg["p_explore"])
    if "probing_jitter" in allowed:
        updates["probing_jitter"] = float(probing_cfg["jitter"])
    if "probing_widen_only" in allowed:
        updates["probing_widen_only"] = bool(probing_cfg["widen_only"])

    probing = replace(base, **updates) if updates else copy.deepcopy(base)

    calib_cfg = pcfg["calibration"]
    kmin, kmax = calib_cfg["k_bounds"]
    grid_size = int(calib_cfg["grid_size"])

    runs: list[tuple[str, float, float, float]] = []
    for name, params in [("baseline", base), ("probing", probing)]:
        df = run_mm_toy(params)
        delta, n = build_intensity_dataset_from_mm(df, dt=params.dt)
        est = fit_intensity_exp_mle(
            delta,
            n,
            dt=params.dt,
            k_bounds=(float(kmin), float(kmax)),
            grid_size=grid_size,
        )
        runs.append((name, float(est.A), float(est.k), float(est.nll)))

    print("=== True ===")
    print(f"A_true={base.A:.4f}, k_true={base.k:.4f}")
    print("=== Estimates ===")
    for name, Ahat, khat, nll in runs:
        print(f"{name:8s}  A_hat={Ahat:.4f}  k_hat={khat:.4f}  nll={nll:.1f}")

    Path("reports/figures").mkdir(parents=True, exist_ok=True)

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