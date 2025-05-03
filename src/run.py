from src.download_data import *
from src.preprocess import *
from src.predict import *
from src.visualize import *
from tensorflow.keras.models import load_model


def run_stock_prediction(ticker):
    """
    Main function to run the stock prediction process.
    Compatible with models trained on only the Close price.
    """
    try:
        print(f"🚀 Starting prediction process for {ticker}...")
        
        # Step 1: Download recent data
        data_path = download_stock_data(ticker)
        if not data_path:
            return
        recent_data_path = preprocess_data(data_path,ticker)

        recent_df = pd.read_csv(recent_data_path)
        if 'Date' not in recent_df.columns:
            recent_df.reset_index(inplace=True)
            recent_df.rename(columns={'index': 'Date'}, inplace=True)
        recent_df['Date'] = pd.to_datetime(recent_df['Date'])
        
        # Step 2: Load historical data
        historical_df = load_historical_data(ticker)
    
            
        historical_df['Date'] = pd.to_datetime(historical_df['Date'])
        print(f"✅ Loaded historical data from historical_path")
        
        # Step 3: Merge data
        # Ensure we have the required columns in both dataframes
        required_cols = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']
        
        # Check and fix column names in recent_df
        if 'Adj Close' in recent_df.columns and 'Close' not in recent_df.columns:
            recent_df['Close'] = recent_df['Adj Close']
        
        # Ensure all required columns exist
        for col in required_cols:
            if col not in recent_df.columns:
                print(f"❌ Column {col} missing from recent data")
                if col != 'Date':  # Date is handled separately
                    recent_df[col] = 0  # Add dummy column
        
        # Concatenate dataframes
        all_data = pd.concat([historical_df, recent_df])
        
        combine_path = f"combine_data/{ticker}.csv"
        all_data.to_csv(combine_path, index=False)

        # Remove duplicates and sort by date
        all_data = all_data.drop_duplicates(subset=['Date']).sort_values('Date').reset_index(drop=True)
        print(f"✅ Data merged successfully. Total records: {len(all_data)}")
        
        # directory = '../combine_data'
        # if not os.path.exists(directory):
        #     os.makedirs(directory)

        # file_path = os.path.join(f'combine_data/{ticker}.csv')
        # all_data.to_csv(file_path)
        print("✅ Combine file save Sucessfully")
        # Step 4: Prepare data for prediction (using only Close price)
        x_input, scaler, scaled_data = prepare_data_for_single_feature_model(all_data)
        if x_input is None:
            return
        
        # Step 5: Load model
        model_path = "model/{}_model.h5".format(ticker)
        if not os.path.exists(model_path):
            print(f"❌ Model not found at {model_path}")
            return
            
        model = load_model(model_path)
        print(f"✅ Loaded model from {model_path}")
        
        # Step 6: Make predictions using the single feature model
        predicted_prices = predict_next_days_single_feature(model, x_input, scaler)
        if predicted_prices is None:
            return
        
        # Step 7: Show results
        last_known_price = all_data['Close'].iloc[-1]
        print(f"\n✅ Last known closing price: ${last_known_price:.2f}")
        
        suggest_buy_sell(predicted_prices, last_known_price)
        
        # Step 8: Plot the results
        plot_predicted_prices(all_data, predicted_prices)
        
        return predicted_prices
        
    except Exception as e:
        print(f"❌ Error in stock prediction process: {str(e)}")
        import traceback
        traceback.print_exc()
        return None