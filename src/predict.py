import numpy as np

def predict_next_days_single_feature(model, x_input, scaler, time_step=60, days=7):
    """
    Predict stock prices for the next 7 days using a model trained on only the Close price.
    """
    try:
        # Array to store predictions
        predicted_prices = []
        
        # Make a copy of the input data to avoid modifying the original
        current_batch = x_input.copy()
        
        # Loop for each day we want to predict
        for i in range(days):
            # Make prediction for the current batch (shape will be [1, 1])
            current_pred = model.predict(current_batch, verbose=0)
            
            # Store the prediction
            predicted_prices.append(current_pred[0, 0])
            
            # Update the input sequence for the next prediction
            # Remove the oldest data point and add the new prediction
            new_point = np.array([[[current_pred[0, 0]]]])  # Shape: [1, 1, 1]
            current_batch = np.append(current_batch[:, 1:, :], new_point, axis=1)
        
        # Convert the scaled predictions back to original price scale
        predicted_prices_array = np.array(predicted_prices).reshape(-1, 1)
        unscaled_predictions = scaler.inverse_transform(predicted_prices_array)
        
        # Return as a flat array
        return unscaled_predictions.flatten()
    except Exception as e:
        print(f"Error predicting next days: {str(e)}")
        return None




import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from datetime import datetime, date, timedelta
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model

def download_stock_data(symbol: str):
    """
    Download stock data from Yahoo Finance from Jan 1, 2025 to today.
    """
    try:
        # Get today's date and the date two years ago
        end_date = datetime.today().strftime('%Y-%m-%d')
        start_date = '2025-01-01'

        # Ensure the 'data' directory exists, create if not
        directory = 'data'
        if not os.path.exists(directory):
            os.makedirs(directory)

        # Download the stock data
        stock_data = yf.download(symbol, start=start_date, end=end_date, auto_adjust=False)
        
        # Save the downloaded data to a CSV file in the 'data' directory
        file_path = os.path.join(directory, f'{symbol}_recent.csv')
        stock_data.to_csv(file_path)

        print(f"✅ Downloaded recent data saved to {file_path}")
        return file_path

    except Exception as e:
        print(f"Error downloading data: {str(e)}")
        return None

def prepare_data_for_single_feature_model(df, feature='Close', time_step=60):
    """
    Prepare data for prediction using only the Close price feature.
    This is compatible with models trained on a single feature.
    """
    try:
        # Extract only the Close prices as a numpy array
        data = df[feature].values.reshape(-1, 1)
        
        # Scale the data
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = scaler.fit_transform(data)
        
        # Get the latest sequence for prediction
        x_input = scaled_data[-time_step:].reshape(1, time_step, 1)
        
        return x_input, scaler, scaled_data
    except Exception as e:
        print(f"Error preparing data: {str(e)}")
        return None, None, None

def predict_next_days_single_feature(model, x_input, scaler, time_step=60, days=7):
    """
    Predict stock prices for the next 7 days using a model trained on only the Close price.
    """
    try:
        # Array to store predictions
        predicted_prices = []
        
        # Make a copy of the input data to avoid modifying the original
        current_batch = x_input.copy()
        
        # Loop for each day we want to predict
        for i in range(days):
            # Make prediction for the current batch (shape will be [1, 1])
            current_pred = model.predict(current_batch, verbose=0)
            
            # Store the prediction
            predicted_prices.append(current_pred[0, 0])
            
            # Update the input sequence for the next prediction
            # Remove the oldest data point and add the new prediction
            new_point = np.array([[[current_pred[0, 0]]]])  # Shape: [1, 1, 1]
            current_batch = np.append(current_batch[:, 1:, :], new_point, axis=1)
        
        # Convert the scaled predictions back to original price scale
        predicted_prices_array = np.array(predicted_prices).reshape(-1, 1)
        unscaled_predictions = scaler.inverse_transform(predicted_prices_array)
        
        # Return as a flat array
        return unscaled_predictions.flatten()
    except Exception as e:
        print(f"Error predicting next days: {str(e)}")
        return None

def suggest_buy_sell(predicted_prices, last_known_price):
    """
    Suggest buy/sell decisions based on predicted prices and the last known price.
    """
    print("\n📊 Stock Prediction Summary & Recommendation")
    print("────────────────────────────────────────────")
    print(f"{'Day':<6}{'Price ($)':<12}{'Change %':<10}{'Action':<10}")
    print("────────────────────────────────────────────")
    
    previous_price = last_known_price
    
    for i, price in enumerate(predicted_prices):
        change_pct = ((price - previous_price) / previous_price) * 100
        
        if change_pct > 1.0:  # More than 1% gain
            action = "Buy 📈"
        elif change_pct < -1.0:  # More than 1% loss
            action = "Sell 📉"
        else:
            action = "Hold 🤝"
            
        print(f"Day {i+1:<2}  ${price:<10.2f}  {change_pct:+.2f}%    {action:<10}")
        previous_price = price
        
    print("────────────────────────────────────────────")
    
    # Overall recommendation
    overall_change = ((predicted_prices[-1] - last_known_price) / last_known_price) * 100
    print(f"\n🔮 Overall 7-Day Outlook: {overall_change:+.2f}%")
    
    if overall_change > 3.0:
        print("💡 Recommendation: STRONG BUY - Significant positive trend predicted")
    elif overall_change > 1.0:
        print("💡 Recommendation: BUY - Moderate positive trend predicted")
    elif overall_change < -3.0:
        print("💡 Recommendation: STRONG SELL - Significant negative trend predicted")
    elif overall_change < -1.0:
        print("💡 Recommendation: SELL - Moderate negative trend predicted")
    else:
        print("💡 Recommendation: HOLD - No significant trend predicted")