import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import os

def preprocess_data(file_path,ticker):
    try:
        # Load the raw data
        df = pd.read_csv(file_path)

        # Drop first two/three rows if metadata is detected (e.g., ticker name)
        try:
            pd.to_datetime(df.iloc[2, 0])  # Try to parse third row as date
            df = df.iloc[2:].copy()
        except Exception:
            df = df.iloc[3:].copy()

        # Rename 'Price' to 'Date' if necessary
        if 'Price' in df.columns:
            df.rename(columns={'Price': 'Date'}, inplace=True)

        # Expected clean column list
        expected_cols = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']
        df = df[[col for col in expected_cols if col in df.columns]]

        # Convert 'Date' to datetime format
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')

        # Convert price-related columns to numeric
        for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
            df[col] = pd.to_numeric(df[col], errors='coerce')

        # Remove rows with missing or invalid values
        df.dropna(inplace=True)

        # Save cleaned data
        os.makedirs('data', exist_ok=True)
        clean_file_path = os.path.join('data', f'{ticker}_clean.csv')
        df.to_csv(clean_file_path, index=False)

        print(f"✅ Cleaned data saved to: {clean_file_path}")
        return clean_file_path

    except Exception as e:
        print(f"❌ Error in preprocessing: {e}")
        return None, None




def merge_data(historical_df, recent_df):
    """
    Merge historical and recent data into a single dataframe.
    """
    try:
        # Ensure Date column is in datetime format
        if 'Date' not in recent_df.columns:
            recent_df.reset_index(inplace=True)
            recent_df.rename(columns={'price': 'Date'}, inplace=True)
        
        recent_df['Date'] = pd.to_datetime(recent_df['Date'])
        
        # Concatenate the dataframes
        merged_df = pd.concat([historical_df, recent_df])
        
        # Remove duplicates if any
        merged_df = merged_df.drop_duplicates(subset=['Date']).sort_values('Date')
        
        print(f"✅ Data merged successfully. Total records: {len(merged_df)}")
        return merged_df
    except Exception as e:
        print(f"Error merging data: {str(e)}")
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