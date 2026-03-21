from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import yaml

from optimal_quoting.backtest.engine import MMParams, run_mm_toy


def main() -> None:
    cfg = yaml.safe_load(Path("configs/mm_toy.yaml").read_text(encoding="utf-8"))
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
        seed=int(mm.get("seed", 0)),
    )

    df = run_mm_toy(p)

    Path("reports/figures").mkdir(parents=True, exist_ok=True)

    plt.figure()
    plt.plot(df["time_s"], df["equity"])
    plt.title("Equity curve (toy MM)")
    plt.tight_layout()
    plt.savefig("reports/figures/mm_toy_equity.png")
    plt.close()

    print(df.tail())
    print("Saved reports/figures/mm_toy_equity.png")


if __name__ == "__main__":
    main()
