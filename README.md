# Probing and adverse selection in market making 

Research-oriented project on **optimal market making and quoting** using
**stochastic control**, **order-flow intensity models**, and **inventory risk management**.

The goal is to build a clean, reproducible framework to:
- model execution via intensity-based fills,
- design optimal quoting policies under inventory constraints,
- backtest and evaluate market making strategies in a realistic microstructure setting.

## Structure
- `src/optimal_quoting/` : core library (models, strategy, simulator)
- `configs/` : experiment configurations
- `scripts/` : runnable experiments
- `docs/` : research notes, derivations, methodology
- `tests/` : unit tests
- `reports/` : generated figures and metrics

## Roadmap
See `docs/ROADMAP.md`.
# Optimal Quoting  
### Stochastic Control and Market Microstructure

**Research-oriented Python project** on optimal market making and optimal quoting under market microstructure uncertainty.

This repository implements a **toy microstructure simulator**, **intensity calibration**, and **multiple quoting policies**, with a strong focus on **identifiability**, **controlled exploration (probing)**, and **risk–PnL trade-offs**.

---

## Why this project matters

This project is designed as a **research-grade sandbox for quantitative finance**.

It demonstrates:
- how **microstructure data limitations bias parameter estimation**,  
- how **controlled exploration (probing)** restores identifiability,  
- and how **information acquisition impacts both profitability and inventory risk**.

The emphasis is not only on performance, but on **understanding the structure of the problem**.

---

## Core components

- Microstructure simulator with stochastic fills  
- Intensity calibration via **maximum likelihood estimation**  
- Quoting strategies:
  - baseline quoting  
  - probing-enhanced quoting  
  - **Avellaneda–Stoikov** policy  
- Benchmarking framework:
  - PnL  
  - volatility  
  - inventory risk  
- Information–PnL frontier experiments

---

## Reproducibility

The project is **fully reproducible**.

- Experiments are executed via scripts located in `scripts/`
- Configurations are stored in `configs/` (YAML)
- Generated results (CSV files and figures) are written to `reports/`

> Results are intentionally **not committed** to version control to preserve reproducibility and clarity.

---

## Main experiments

- Avellaneda–Stoikov toy backtest  
- Policy benchmark: **baseline vs probing**  
- Information–PnL frontier via probing parameter sweeps  

---

## Repository structure

src/optimal_quoting/ core library
─ backtest simulation engine
─ calibration intensity estimation
─ strategy quoting policies
─ metrics performance measures
─ experiments probing & frontier studies

scripts/ reproducible experiment entry points
configs/ YAML experiment configurations
docs/ research notes and roadmap
tests/ unit and integration tests

---

## Scope and limitations

This is a **toy model**, designed for **clarity, experimentation, and research insight**, not for production trading.

Planned extensions include:
- transaction costs and adverse selection  
- risk aversion tuning  
- richer market impact models  
- links to theoretical stochastic control results  

---

## Author

**Mehdi Lahrach**  
M1 Applied Mathematics and Statistics — Quantitative Finance - PSL University