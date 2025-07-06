import pandas as pd
import numpy as np
import yfinance as yf
import quantstats as qs
from pgmpy.models import BayesianNetwork
from pgmpy.estimators import K2Score, HillClimbSearch, MaximumLikelihoodEstimator

# --- Step 1: Data Preparation ---
# For demonstration, we will use daily data and create plausible daily features.
print("Step 1: Fetching and Preparing Data...")
try:
    # Fetching daily data for a longer history for a more robust model
    data = yf.download('ES=F', start='2015-01-01', end='2024-01-01', interval='1d')
    if data.empty:
        raise ValueError("No data downloaded. Please check the ticker and date range.")
except Exception as e:
    print(f"Could not download data: {e}")
    # Creating a dummy dataframe if download fails
    data = pd.DataFrame(
        np.random.rand(2000, 5),
        columns=['Open', 'High', 'Low', 'Close', 'Volume'],
        index=pd.to_datetime(pd.date_range('2015-01-01', periods=2000))
    )
    data['Close'] = data['Open'] + (np.random.rand(2000) - 0.5)

# --- Feature Engineering (ASSUMPTION) ---
data['Overnight_Return'] = (data['Open'] / data['Close'].shift(1)) - 1
data['Prev_Day_Return'] = data['Close'].shift(1).pct_change()
data['Prev_Day_Range'] = (data['High'].shift(1) - data['Low'].shift(1)) / data['Close'].shift(1)

# --- Target Variable Definition ---
# The goal is to predict the spread between open and close [T30](1)
# We define the target as the sign of (Close - Open)
data['Target'] = np.where(data['Close'] > data['Open'], 'Up', 'Down')

data.dropna(inplace=True)

# --- Data Discretization (ASSUMPTION) ---
# The K2 algorithm requires discrete data. We use quantiles (e.g., 3 bins).
features_to_discretize = ['Overnight_Return', 'Prev_Day_Return', 'Prev_Day_Range']
discretized_data = pd.DataFrame()

for col in features_to_discretize:
    try:
        discretized_data[col] = pd.qcut(data[col], q=3, labels=['Low', 'Medium', 'High'], duplicates='drop')
    except ValueError:
        # Handle cases with low variance by using rank
        discretized_data[col] = pd.qcut(data[col].rank(method='first'), q=3, labels=['Low', 'Medium', 'High'])


discretized_data['Target'] = data['Target']

# Split data for training and testing
train_data = discretized_data.loc[:'2022-12-31']
test_data = discretized_data.loc['2023-01-01':]

print(f"Data prepared. Training set size: {len(train_data)}, Test set size: {len(test_data)}")

# --- Step 2: Bayesian Network Algorithm ---
# The report mentions using the K2 algorithm. `pgmpy`'s HillClimbSearch with a K2Score
# is a functionally equivalent greedy search algorithm.
print("\nStep 2: Building the Bayesian Network using a K2-based approach...")

scoring_method = K2Score(data=train_data)
hc = HillClimbSearch(data=train_data)

# Learn the network structure. The `Target` variable is what we want to predict.
# We assume other features can be its parents.
best_model_structure = hc.estimate(
    scoring_method=scoring_method,
    black_list=[(col, 'Target') for col in features_to_discretize] # Features can't be children of the target
)

# Create the Bayesian Network model with the learned structure
model = BayesianNetwork(best_model_structure.edges())

# Fit the model (learn Conditional Probability Distributions)
model.fit(data=train_data, estimator=MaximumLikelihoodEstimator)

print("Bayesian Network learned. Edges:")
print(model.edges())

# --- Step 3: Define Trading Rules (ASSUMPTION) ---
# The report does not specify trading rules. We define a clear set of rules.
# Rule: If the predicted probability of 'Up' is > threshold, go long at open, exit at close.
# If probability of 'Up' is < (1 - threshold), go short at open, exit at close.

PROB_THRESHOLD = 0.55 # Confidence threshold for taking a trade

def get_signal(model, daily_features):
    """
    Generates a trading signal based on the model's prediction.
    - +1: Long
    - -1: Short
    -  0: Neutral
    """
    try:
        # Prepare evidence for inference
        evidence = {col: daily_features[col] for col in features_to_discretize}
        
        # Perform probabilistic inference
        prediction = model.predict_probability(queries=['Target'], evidence=evidence)
        prob_up = prediction.values[prediction.state_names['Target'].index('Up')]
        
        if prob_up > PROB_THRESHOLD:
            return 1
        elif prob_up < (1 - PROB_THRESHOLD):
            return -1
        else:
            return 0
    except Exception as e:
        # If a specific state was not seen during training, it might fail.
        print(f"Could not generate signal for features {daily_features}: {e}")
        return 0

# --- Step 4: Event-Based Backtesting ---
print("\nStep 4: Running the backtest...")

test_set_full = data.loc[test_data.index].copy()
signals = []
for i in range(len(test_data)):
    daily_features = test_data.iloc[i]
    signal = get_signal(model, daily_features)
    signals.append(signal)

test_set_full['Signal'] = signals

# --- Backtest Assumptions (Slippage & Fees) ---
TRADING_COST_BPS = 0.00005 

# Calculate daily returns based on signal
# Trade is entered at Open and exited at Close of the same day.
test_set_full['Strategy_Return'] = np.where(
    test_set_full['Signal'] != 0,
    test_set_full['Signal'] * ((test_set_full['Close'] / test_set_full['Open']) - 1) - (2 * TRADING_COST_BPS), # Cost applied on entry and exit
    0
)

# --- Step 5: Performance Evaluation ---
print("\nStep 5: Evaluating strategy performance...")

strategy_returns = test_set_full['Strategy_Return']

# Calculate prediction accuracy, as mentioned in the report
actuals = test_data['Target']
predictions = np.where(test_set_full['Signal'] == 1, 'Up', np.where(test_set_full['Signal'] == -1, 'Down', 'Neutral'))
correct_predictions = (predictions == actuals) & (predictions != 'Neutral')
total_trades = (predictions != 'Neutral').sum()

if total_trades > 0:
    accuracy = correct_predictions.sum() / total_trades
    print(f"\nPrediction Accuracy on Trades: {accuracy:.2%}")
else:
    print("\nNo trades were executed.")

# Use quantstats for a comprehensive report
qs.extend_pandas()
strategy_returns.index = pd.to_datetime(strategy_returns.index)

print("\n--- Performance Report ---")
qs.reports.full(strategy_returns, benchmark='SPY')
