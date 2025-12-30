from __future__ import annotations

from dataclasses import dataclass
import pandas as pd

from optimal_quoting.backtest.engine import MMParams, run_mm_toy
from optimal_quoting.metrics.performance import performance_summary
from optimal_quoting.calibration.dataset import build_intensity_dataset_from_mm
from optimal_quoting.calibration.mle import fit_intensity_exp_mle


@dataclass(frozen=True)
class FrontierConfig:
    p_grid: list[float]
    jitter_grid: list[float]
    seeds: list[int]
    k_bounds: tuple[float, float] = (0.0, 5.0)
    grid_size: int = 300


def run_probing_frontier(base: MMParams, cfg: FrontierConfig) -> pd.DataFrame:
    """
    Sweep (probing_p, probing_jitter) and seeds, run toy MM backtest,
    compute trading metrics + intensity MLE identifiability metrics.

    Notes:
    - MMParams is frozen/immutable => we rebuild a new MMParams per run.
    """
    rows: list[dict] = []

    # Build a dict copy once; remove seed to avoid duplicate kwarg on rebuild.
    base_dict = dict(base.__dict__)
    base_dict = dict(base.__dict__)
    for k in ("seed", "policy", "probing_p", "probing_jitter", "probing_widen_only"):
        base_dict.pop(k, None)

    for p_explore in cfg.p_grid:
        for jitter in cfg.jitter_grid:
            for seed in cfg.seeds:
                policy = "probing" if (p_explore > 0.0 and jitter > 0.0) else "baseline"

                p = MMParams(
                    **base_dict,
                    seed=int(seed),
                    policy=policy,
                    probing_p=float(p_explore),
                    probing_jitter=float(jitter),
                    probing_widen_only=True,
                )

                # Run backtest
                df = run_mm_toy(p)

                # Trading performance metrics (expects keys like pnl_final, inv_std, inv_max_abs, etc.)
                perf = performance_summary(df)

                # Intensity dataset + MLE fit (A_hat, k_hat)
                delta, n = build_intensity_dataset_from_mm(df, dt=p.dt)
                est = fit_intensity_exp_mle(
                    delta,
                    n,
                    dt=p.dt,
                    k_bounds=cfg.k_bounds,
                    grid_size=cfg.grid_size,
                )

                rows.append(
                    {
                        "p_explore": float(p_explore),
                        "jitter": float(jitter),
                        "seed": int(seed),
                        "A_hat": float(est.A),
                        "k_hat": float(est.k),
                        "k_abs_error": float(abs(est.k - p.k)),
                        **perf,
                    }
                )

    return pd.DataFrame(rows)

