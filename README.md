# Oil-Price-Direction-Prediction-using-ML
Oil Price Direction Prediction using ML
# Feedforward Neural Network — WTI Crude Oil Price Direction Forecasting

This project implements a **Feedforward Neural Network (FFNN)** for binary classification of next-day WTI Crude Oil price direction (Up / Down).

The goal is to predict the direction of WTI Crude Oil prices and then utilize these predictions in two distinct investment strategies, comparing their performance against a market (buy-and-hold) baseline.

## Pipeline Overview
The forecasting pipeline involves several key steps:
1.  **Load Raw Data**: Fetch FRED-MD monthly macroeconomic data and WTI daily price data.
2.  **Feature Selection & Transformation**: Select 31 specific macro variables and apply t-code stationarity transformations (Level, First Difference, Log First Difference, Log Second Difference).
3.  **Lagging & Forward-Filling**: Apply a 1-month reporting lag to macroeconomic data and forward-fill values to align with daily WTI prices, ensuring no future leakage.
4.  **Feature Engineering**: Construct 30 lagged daily WTI price features and a binary target variable (1 if next-day price is Up, 0 otherwise).
5.  **Data Split**: Chronologically split the data into training, validation, and test sets (up to 2014, 2015-2017, 2018 onward, respectively) without shuffling.
6.  **Normalization**: Apply `StandardScaler` to features, fitting only on training data to prevent data leakage.
7.  **Cross-Validation**: Perform 5-fold `TimeSeriesSplit` cross-validation with expanding windows, fitting a fresh `StandardScaler` inside each fold to maintain data integrity.
8.  **Model Training**: Train the final FFNN on the full training set, using the validation set for early stopping.
9.  **Evaluation**: Evaluate the model's out-of-sample performance on the test set using various classification metrics (Accuracy, Precision, Recall, F1, ROC-AUC) and visualisations.

## FFNN Model Architecture
The Feedforward Neural Network is designed for binary classification with the following structure:
-   Input Layer (61 features: 30 lagged prices + 31 macro variables)
-   Dense(128, ReLU) → BatchNormalization → Dropout(0.3)
-   Dense(64, ReLU, L2 Regularization) → Dropout(0.2)
-   Dense(32, ReLU, L2 Regularization) → Dropout(0.2)
-   Output Layer: Dense(1, Sigmoid) for binary probability output.

**Training Configuration:**
-   **Loss Function**: Binary Cross-Entropy
-   **Optimizer**: Adam (Learning Rate: 0.001)
-   **Callbacks**: Early Stopping (patience=30, restores best weights), ReduceLROnPlateau (factor=0.5, patience=10).

## Investment Strategies
Two investment strategies are implemented and evaluated on the test set:

### Strategy 1: Simple Prediction-Based Trading
-   **Entry Condition**: Enter a position (take daily return) if the model predicts an 'Up' movement (`Prediction = 1`).
-   **Exit Condition**: No position taken (return is 0) if the model predicts 'Down' (`Prediction = 0`).
-   **Transaction Costs**: A fixed cost of `0.001` is applied only when a position is entered.

### Strategy 2: Probability Threshold-Based Trading
-   **Entry Condition**: Enter a position only if the model's predicted probability of an 'Up' movement (`Probability`) is greater than `0.52` (higher confidence filter).
-   **Exit Condition**: No position taken if `Probability` is not above `0.52`.
-   **Transaction Costs**: A fixed cost of `0.001` is applied only when `Probability` is greater than `0.6` (very high confidence).

## Final Metrics (Test Set Performance)
The performance of both strategies is compared against a market (buy-and-hold) baseline using Cumulative Return and Annualized Sharpe Ratio.

| Metric              | Baseline (Majority) | FFNN Model (Strategy 1) | FFNN Model (Strategy 2) |
| :------------------ | :------------------ | :---------------------- | :---------------------- |
| Accuracy            | 0.5902              | N/A                     | N/A                     |
| Precision (Down)    | 0.0000              | N/A                     | N/A                     |
| Recall (Down)       | 0.0000              | N/A                     | N/A                     |
| Precision (Up)      | 0.5902              | N/A                     | N/A                     |
| Recall (Up)         | 1.0000              | N/A                     | N/A                     |
| F1 (macro)          | 0.3711              | N/A                     | N/A                     |
| ROC-AUC             | N/A                 | N/A                     | N/A                     |
| **Cumulative Return** | **0.1213 (12.13%)** | **0.1848 (18.48%)**     | **0.2685 (26.85%)**     |
| **Sharpe Ratio**    | **0.7685**          | **1.4167**              | **1.9925**              |

**Interpretation:**
-   **Strategy 2** significantly outperforms both Strategy 1 and the Market, demonstrating the highest cumulative return and best risk-adjusted returns (Sharpe Ratio).
-   **Strategy 1** also beats the market baseline, showing the value of the FFNN's predictions even with a simpler decision rule.
-   The **Market (Buy-and-Hold)** serves as a baseline, indicating that the FFNN-driven strategies successfully identify profitable trading opportunities.

## Usage
This project is designed as a Google Colab notebook. To run it:
1.  Open the `.ipynb` file in Google Colab.
2.  Mount your Google Drive to `/content/drive/MyDrive` as the project expects raw data files (`FRED-MD_2024m12.csv` and `Crude Oil WTI Futures Historical Data (1).csv`) to be located in `VIP/` within your Google Drive.
3.  Run all cells sequentially.

