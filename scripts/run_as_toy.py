from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import yaml

from optimal_quoting.backtest.engine import MMParams, run_mm_toy


def main() -> None:
    cfg = yaml.safe_load(Path("configs/as_toy.yaml").read_text(encoding="utf-8"))

    p = MMParams(
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
        policy=str(cfg["policy"]["name"]),
        gamma=float(cfg["policy"]["gamma"]),
    )

    df = run_mm_toy(p)

    Path("reports/figures").mkdir(parents=True, exist_ok=True)

    plt.figure()
    plt.plot(df["time_s"], df["equity"])
    plt.title("Equity curve (AS policy)")
    plt.tight_layout()
    plt.savefig("reports/figures/as_equity.png")
    plt.close()

    print(df.tail())
    print("Saved reports/figures/as_equity.png")


if __name__ == "__main__":
    main()
