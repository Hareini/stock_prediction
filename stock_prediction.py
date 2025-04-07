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

# Flatten MultiIndex columns if they exist
if isinstance(stock_data.columns, pd.MultiIndex):
    stock_data.columns = stock_data.columns.get_level_values(0)

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

# =================================================================
# REGRESSION MODELS (Predicting Close Price)
# =================================================================
X_reg = stock_data[['Year', 'Month', 'Day']]
y_reg = stock_data['Close']
X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(
    X_reg, y_reg, test_size=0.2, random_state=42, shuffle=False)

scaler_reg = StandardScaler()
X_train_reg_scaled = scaler_reg.fit_transform(X_train_reg)
X_test_reg_scaled = scaler_reg.transform(X_test_reg)

reg_models = {
    'Ridge Regression': Ridge(),
    'Lasso Regression': Lasso(max_iter=5000, alpha=0.1),
    'Linear Regression': LinearRegression(),
    'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
    'Extra Trees': ExtraTreesRegressor(n_estimators=100, random_state=42),
    'MLP (Neural Net)': MLPRegressor(max_iter=1000, hidden_layer_sizes=(100, 50), random_state=42),
    'AdaBoost': AdaBoostRegressor(random_state=42),
    'Decision Tree': DecisionTreeRegressor(random_state=42),
    'XGBoost': XGBRegressor(random_state=42)
}

reg_performance = []
for model_name, model in reg_models.items():
    try:
        model.fit(X_train_reg_scaled, y_train_reg)
        y_pred = model.predict(X_test_reg_scaled)
        
        reg_performance.append({
            'Model': model_name,
            'MSE': mean_squared_error(y_test_reg, y_pred),
            'RMSE': np.sqrt(mean_squared_error(y_test_reg, y_pred)),
            'R²': r2_score(y_test_reg, y_pred),
            'MAE': mean_absolute_error(y_test_reg, y_pred),
            'MAPE (%)': mean_absolute_percentage_error(y_test_reg, y_pred) * 100
        })
    except Exception as e:
        print(f"Regression error with {model_name}: {str(e)}")

reg_df = pd.DataFrame(reg_performance).sort_values(by='R²', ascending=False)

# =================================================================
# CLASSIFICATION MODELS (Predicting Price Direction)
# =================================================================
X_clf = stock_data[['Lag_1', 'Lag_2', 'Lag_3']]
y_clf = stock_data['Direction']
X_train_clf, X_test_clf, y_train_clf, y_test_clf = train_test_split(
    X_clf, y_clf, test_size=0.2, random_state=42, shuffle=False)

scaler_clf = StandardScaler()
X_train_clf_scaled = scaler_clf.fit_transform(X_train_clf)
X_test_clf_scaled = scaler_clf.transform(X_test_clf)

clf_models = {
    'Logistic Regression': LogisticRegression(max_iter=1000),
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
    'Extra Trees': ExtraTreesClassifier(n_estimators=100, random_state=42),
    'MLP (Neural Net)': MLPClassifier(max_iter=1000, hidden_layer_sizes=(50, 25), random_state=42),
    'AdaBoost': AdaBoostClassifier(random_state=42),
    'Decision Tree': DecisionTreeClassifier(random_state=42),
    'XGBoost': XGBClassifier(random_state=42)
}

clf_performance = []
for model_name, model in clf_models.items():
    try:
        model.fit(X_train_clf_scaled, y_train_clf)
        y_pred = model.predict(X_test_clf_scaled)
        
        clf_performance.append({
            'Model': model_name,
            'Accuracy': accuracy_score(y_test_clf, y_pred),
            'F1-Score': f1_score(y_test_clf, y_pred),
            'Precision': precision_score(y_test_clf, y_pred),
            'Recall': recall_score(y_test_clf, y_pred)
        })
    except Exception as e:
        print(f"Classification error with {model_name}: {str(e)}")

clf_df = pd.DataFrame(clf_performance).sort_values(by='Accuracy', ascending=False)

# =================================================================
# RESULTS
# =================================================================
print("\nREGRESSION RESULTS:")
print(reg_df)

print("\nCLASSIFICATION RESULTS:")
print(clf_df)

# Visualization: Actual vs Predicted Prices (Best Regression Model)
best_reg_model = reg_models[reg_df.iloc[0]['Model']]
y_pred_reg = best_reg_model.predict(X_test_reg_scaled)

plt.figure(figsize=(14, 7))
plt.plot(y_test_reg.index, y_test_reg, label='Actual Prices', color='blue', linewidth=2)
plt.plot(y_test_reg.index, y_pred_reg, label='Predicted Prices', color='red', linestyle='--', linewidth=1.5)
plt.xlabel('Date')
plt.ylabel('Stock Price ($)')
plt.title(f'Best Regression Model: {reg_df.iloc[0]["Model"]} (R²={reg_df.iloc[0]["R²"]:.3f})')
plt.legend()
plt.grid(True)
plt.show()
