# Strategy-Backtest-Bayesian-Network-Based-Price-Prediction-Trading-Strategy
This repo outlines a step-by-step Python implementation of the proposed Bayesian Network trading strategy.

It is crucial to note that the source text describes the strategy at a high level but omits several critical details required for an exact replication. Specifically, the report does not specify:
-The exact predictor variables (features) used as nodes in the network.
-The data discretization method for converting continuous price/volume data into discrete states.
-The precise trading rules that translate the network's probabilistic forecast into buy/sell signals.

Therefore, the following implementation completes the strategy by making reasonable, clearly stated assumptions for these missing components. The code targets the E-mini S&P 500 (ES) futures contract.

***Strategy Overview***

The core concept is to use a Bayesian Network, a type of probabilistic graphical model, to predict the direction of the intraday price movement (the spread between the opening and closing price). The network structure is learned from historical data using the K2 algorithm, a score-based greedy search method. The resulting probabilistic forecast is then used to generate daily trading decisions
