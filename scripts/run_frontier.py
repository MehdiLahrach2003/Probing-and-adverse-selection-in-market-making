from __future__ import annotations

from pathlib import Path
import yaml
import pandas as pd
import matplotlib.pyplot as plt

from optimal_quoting.backtest.engine import MMParams
from optimal_quoting.experiments.probing_frontier import FrontierConfig, run_probing_frontier


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def plot_heatmap_mean(df: pd.DataFrame, value_col: str, title: str, out_path: str) -> None:
    """
    Heatmap over (p_explore, jitter) of mean(value_col) over seeds.
    """
    pivot = (
        df.groupby(["p_explore", "jitter"])[value_col]
        .mean()
        .unstack()
    )

    plt.figure(figsize=(7, 5))
    im = plt.imshow(
        pivot.values,
        origin="lower",
        aspect="auto",
        cmap="viridis",
    )
    plt.colorbar(im, label=f"Mean {value_col}")

    plt.xticks(range(len(pivot.columns)), [f"{x:.2f}" for x in pivot.columns])
    plt.yticks(range(len(pivot.index)), [f"{x:.2f}" for x in pivot.index])

    plt.xlabel("jitter")
    plt.ylabel("p_explore")
    plt.title(title)

    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def main() -> None:
    cfg = load_config("configs/mm_toy.yaml")

    # --- Base MMParams ---
    if "mm_params" not in cfg:
        raise KeyError("configs/mm_toy.yaml must define a top-level `mm_params:` section.")

    base_params = MMParams(**cfg["mm_params"])

    # --- Frontier sweep config ---
    frontier = cfg.get("frontier", {})
    p_grid = list(map(float, frontier.get("p_grid", [0.0, 0.05, 0.1, 0.2, 0.3])))
    jitter_grid = list(map(float, frontier.get("jitter_grid", [0.0, 0.02, 0.05, 0.1])))
    seeds = list(map(int, frontier.get("seeds", [0, 1, 2, 3, 4])))

    # --- Calibration settings (optional) ---
    calib = cfg.get("intensity_calibration", {})
    k_bounds = tuple(map(float, calib.get("k_bounds", [0.0, 5.0])))
    grid_size = int(calib.get("grid_size", 300))

    frontier_cfg = FrontierConfig(
        p_grid=p_grid,
        jitter_grid=jitter_grid,
        seeds=seeds,
        k_bounds=(k_bounds[0], k_bounds[1]),
        grid_size=grid_size,
    )

    # --- Run experiment ---
    df = run_probing_frontier(base_params, frontier_cfg)

    # --- Outputs ---
    Path("reports").mkdir(exist_ok=True)
    Path("reports/figures").mkdir(parents=True, exist_ok=True)

    out_csv = "reports/frontier_results.csv"
    df.to_csv(out_csv, index=False)
    print(f"Saved {out_csv}")

    # Heatmaps (assumes performance_summary provides these columns)
    plot_heatmap_mean(
        df,
        value_col="pnl_final",
        title="PnL final (mean over seeds)",
        out_path="reports/figures/frontier_pnl_heatmap.png",
    )

    if "inv_max_abs" in df.columns:
        plot_heatmap_mean(
            df,
            value_col="inv_max_abs",
            title="Max |inventory| (mean over seeds)",
            out_path="reports/figures/frontier_inv_heatmap.png",
        )

    plot_heatmap_mean(
        df,
        value_col="k_abs_error",
        title="|k_hat - k_true| (mean over seeds)",
        out_path="reports/figures/frontier_kerr_heatmap.png",
    )

    # Frontier scatter: identifiability vs PnL (mean over seeds)
    agg = df.groupby(["p_explore", "jitter"]).mean(numeric_only=True).reset_index()

    plt.figure(figsize=(7, 5))
    plt.scatter(agg["k_abs_error"].values, agg["pnl_final"].values, s=40)
    plt.xlabel("|k_hat - k_true|")
    plt.ylabel("PnL final")
    plt.title("PnL vs Identifiability (mean over seeds)")
    plt.tight_layout()
    plt.savefig("reports/figures/frontier_pnl_vs_kerr.png")
    plt.close()

    print("Saved reports/figures/frontier_*.png")


if __name__ == "__main__":
    main()


