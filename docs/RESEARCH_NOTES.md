Research Notes — Probing and Identifiability in Optimal Quoting

Abstract
This document summarizes the research motivations and findings of the Optimal Quoting project. We study a stylized market making environment to highlight how limited price exploration leads to biased parameter estimation, and how controlled probing restores identifiability at the cost of altered trading performance.

Modeling framework
We consider a single-asset market making setup where a market maker posts bid and ask quotes around a mid-price process. Order arrivals are modeled via parametric stochastic intensities depending on the distance between quotes and the mid price. The simulator is intentionally simple in order to isolate microstructure and learning effects.

Calibration and identifiability
Model parameters are estimated via maximum likelihood on simulated order flow data. When quotes remain concentrated around a narrow range, the observed data provides insufficient information to reliably identify intensity parameters. This leads to systematic bias, particularly in the decay parameter governing sensitivity to quote distance.

Probing as controlled exploration
To address identifiability issues, we introduce a probing mechanism that occasionally widens or perturbs quotes. Probing increases coverage over the quote distance domain, improving parameter recovery. This comes at the cost of increased inventory risk and modified PnL dynamics, highlighting a clear exploration–exploitation trade-off.

Experimental results
We compare baseline quoting, probing-enhanced quoting, and Avellaneda–Stoikov policies. Experiments include benchmarking across seeds and sweeping probing parameters. Results are summarized through profitability, volatility, inventory statistics, and calibration accuracy, leading to an information–PnL frontier.

Limitations and future work
This project is a research sandbox rather than a production trading system. Natural extensions include transaction costs, adverse selection, risk aversion tuning, multi-asset settings, and links to continuous-time stochastic control theory.

Reproducibility
All experiments are reproducible using the scripts and configurations provided in the repository. Generated results are stored locally and intentionally excluded from version control.