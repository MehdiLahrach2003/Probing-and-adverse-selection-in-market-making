# Roadmap — Optimal Quoting

This project is developed as a research-grade sandbox for optimal market making.
The goal is to keep the code modular, testable, and reproducible, while gradually adding realism.

## A) Research narrative (paper-style)
- Formalize the toy microstructure model assumptions
- Define identifiability issues in intensity calibration
- Show how probing restores identifiability
- Connect information acquisition to PnL and inventory risk

## B) Experimental suite
- Baseline MM backtest (toy simulator)
- Intensity calibration diagnostics (MLE + profile likelihood)
- Probing experiment (bias without probing vs recovery with probing)
- Policy benchmark (baseline vs probing vs Avellaneda–Stoikov)
- Frontier sweep (information–PnL trade-off)
- Stress testing (sensitivity to sigma, spread, costs, etc.)

## C) Engineering quality
- Stable configs (YAML, consistent schema)
- Deterministic runs via seeds
- Unit tests for each submodule
- CI green on every commit

## D) Extensions (next milestones)
- Transaction costs & adverse selection
- Time-varying intensities / non-stationarity
- Risk aversion calibration and parameter sweeps
- Cleaner plotting utilities + report generation
- Add references and derivations for AS policy
