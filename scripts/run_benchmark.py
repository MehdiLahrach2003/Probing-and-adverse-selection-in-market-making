from __future__ import annotations

from pathlib import Path
import copy

import pandas as pd
import yaml

from optimal_quoting.backtest.engine import MMParams, run_mm_toy
from optimal_quoting.metrics.performance import performance_summary


def main() -> None:
    cfg = yaml.safe_load(Path("configs/benchmark.yaml").read_text(encoding="utf-8"))

    base = cfg["base"]
    seeds = cfg["seeds"]
    policies = cfg["policies"]

    rows = []

    for name, pcfg in policies.items():
        for seed in seeds:
            p = MMParams(
                dt=float(base["dt"]),
                T=float(base["T"]),
                mid0=float(base["mid0"]),
                sigma=float(base["sigma"]),
                A=float(base["intensity"]["A"]),
                k=float(base["intensity"]["k"]),
                base_spread=float(base["strategy"]["base_spread"]),
                phi=float(base["strategy"]["phi"]),
                order_size=float(base["strategy"]["order_size"]),
                fee_bps=float(base["costs"]["fee_bps"]),
                seed=int(seed),
                policy=pcfg["policy"],
                gamma=float(pcfg.get("gamma", 0.0)),
                probing_p=float(pcfg.get("probing_p", 0.0)),
                probing_jitter=float(pcfg.get("probing_jitter", 0.0)),
                probing_widen_only=bool(pcfg.get("probing_widen_only", True)),
            )

            df = run_mm_toy(p)
            stats = performance_summary(df)
            stats["policy"] = name
            stats["seed"] = seed
            rows.append(stats)

    res = pd.DataFrame(rows)

    Path("reports").mkdir(exist_ok=True)
    res.to_csv("reports/benchmark_results.csv", index=False)

    print(res.groupby("policy").mean(numeric_only=True))
    print("Saved reports/benchmark_results.csv")


if __name__ == "__main__":
    main()
