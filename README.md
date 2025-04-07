## Code for Stock Prediction

```python
import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
from sklearn.linear_model import Ridge, Lasso, LinearRegression, LogisticRegression
from sklearn.ensemble import (RandomForestRegressor, ExtraTreesRegressor, AdaBoostRegressor, 
                             RandomForestClassifier, ExtraTreesClassifier, AdaBoostClassifier)
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.neural_network import MLPRegressor, MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (mean_squared_error, mean_absolute_error, r2_score, 
                             mean_absolute_percentage_error, accuracy_score, f1_score, 
                             precision_score, recall_score)
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor, XGBClassifier

# Suppress warnings
warnings.filterwarnings("ignore")

# Download stock data
ticker = 'AAPL'
stock_data = yf.download(ticker, start="2010-01-01", end="2023-12-31", auto_adjust=True)

# Feature engineering for both regression and classification
stock_data['Date'] = stock_data.index
stock_data['Year'] = stock_data['Date'].dt.year
stock_data['Month'] = stock_data['Date'].dt.month
stock_data['Day'] = stock_data['Date'].dt.day
stock_data['Returns'] = stock_data['Close'].pct_change()
stock_data['Direction'] = (stock_data['Returns'] > 0).astype(int)  # 1 if price increased

# Create lag features for classification
for lag in range(1, 4):
    stock_data[f'Lag_{lag}'] = stock_data['Returns'].shift(lag)

# Drop missing values
stock_data.dropna(inplace=True)

# (Rest of the code...)
