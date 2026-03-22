from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import yaml

from optimal_quoting.data.loader import CSVSpec, load_top_of_book_csv
from optimal_quoting.features.microstructure import add_log_returns, add_mid_spread, realized_vol


def main() -> None:
    cfg = yaml.safe_load(Path("configs/data_example.yaml").read_text(encoding="utf-8"))
    d = cfg["data"]

    spec = CSVSpec(
        ts_col=d["ts_col"],
        bid_col=d["bid_col"],
        ask_col=d["ask_col"],
    )

    df = load_top_of_book_csv(d["path"], spec)
    df = add_mid_spread(df)
    df = add_log_returns(df)

    print("Rows:", len(df))
    print("Mean spread:", float(df["spread"].mean()))
    print("Realized vol (per step):", realized_vol(df))

    Path("reports/figures").mkdir(parents=True, exist_ok=True)

    plt.figure()
    plt.plot(df["ts"], df["mid"])
    plt.title("Mid price")
    plt.tight_layout()
    plt.savefig("reports/figures/mid.png")
    plt.close()

    plt.figure()
    plt.plot(df["ts"], df["spread"])
    plt.title("Spread")
    plt.tight_layout()
    plt.savefig("reports/figures/spread.png")
    plt.close()

    print("Saved plots to reports/figures/: mid.png, spread.png")


if __name__ == "__main__":
    main()
