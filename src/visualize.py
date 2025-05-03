from datetime import datetime, date, timedelta
import matplotlib.pyplot as plt
import pandas as pd

# def plot_predicted_prices(historical_df, predicted_prices):
#     """
#     Plot historical prices and predicted prices.
#     """
#     # Get the last 30 days of historical data for context
#     last_30_days = historical_df[-30:]
    
#     # Generate dates for predictions
#     last_date = last_30_days['Date'].iloc[-1]
#     future_dates = [last_date + timedelta(days=i+1) for i in range(len(predicted_prices))]
    
#     # Create the plot
#     plt.figure(figsize=(12, 6))
    
#     # Plot historical data
#     plt.plot(last_30_days['Date'], last_30_days['Close'], 
#              label='Historical Prices', color='blue', linewidth=2)
    
#     # Plot predicted data
#     plt.plot(future_dates, predicted_prices, 
#              label='Predicted Prices', color='red', marker='o', 
#              linestyle='--', linewidth=2)
    
#     # Add vertical line to separate historical from predicted
#     plt.axvline(x=last_date, color='green', linestyle='-', alpha=0.5,
#                 label='Prediction Start')
    
#     # Title and labels
#     plt.title("Stock Price Prediction for Next 7 Days", fontsize=16, weight='bold')
#     plt.xlabel("Date", fontsize=12)
#     plt.ylabel("Price ($)", fontsize=12)
#     plt.grid(True, linestyle='--', alpha=0.7)
#     plt.legend()
    
#     # Format dates on x-axis
#     plt.gcf().autofmt_xdate()
    
#     # Show plot
#     plt.tight_layout()
#     plt.show()




def visualize_monthly_data(df, selected_month=None, selected_year=None):
    try:
        # Ensure Date is in datetime format
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')

        # Drop rows with invalid/missing dates
        df.dropna(subset=['Date'], inplace=True)

        # Add Month and Year columns
        df['Month'] = df['Date'].dt.month
        df['Year'] = df['Date'].dt.year

        # Default to current month/year if not provided
        if selected_month is None:
            selected_month = datetime.datetime.today().strftime('%B')
        if selected_year is None:
            selected_year = datetime.datetime.today().year

        # Convert month name to number
        months = ['January', 'February', 'March', 'April', 'May', 'June', 'July',
                  'August', 'September', 'October', 'November', 'December']
        month_number = months.index(selected_month) + 1

        # Filter the data
        filtered_df = df[(df['Year'] == selected_year) & (df['Month'] == month_number)]

        if not filtered_df.empty:
            # Plotting
            plt.figure(figsize=(10, 6))
            plt.plot(filtered_df['Date'], filtered_df['Close'], label=f'{selected_month} {selected_year}',
                     color='royalblue', marker='o', markersize=5)
            plt.title(f'📊 Closing Prices - {selected_month} {selected_year}', fontsize=16, color='darkblue')
            plt.xlabel('Date', fontsize=12)
            plt.ylabel('Price ($)', fontsize=12)
            plt.grid(True, linestyle='--', alpha=0.6)
            plt.xticks(rotation=45)
            plt.legend()
            plt.tight_layout()
            plt.show()

            # Display table
            display(filtered_df[['Date', 'Close']].reset_index(drop=True))
        else:
            print(f"⚠️ No data found for {selected_month} {selected_year}.")

    except Exception as e:
        print(f"❌ Visualization failed: {e}")



# src/visualize.py
import matplotlib.pyplot as plt
import streamlit as st

def plot_monthly_yearly(df, period="Month"):
    df['Date'] = pd.to_datetime(df['Date'])
    df.set_index('Date', inplace=True)

    if period == "Month":
        resampled = df['Close'].resample('M').mean()
    else:
        resampled = df['Close'].resample('Y').mean()

    fig, ax = plt.subplots()
    resampled.plot(ax=ax, title=f"Average Close Price per {period}")
    ax.set_ylabel("Price ($)")
    ax.set_xlabel(period)
    st.pyplot(fig)








import matplotlib.pyplot as plt
import streamlit as st

def plot_predicted_prices_streamlit(all_data, predicted_prices):
    plt.figure(figsize=(10, 4))
    plt.plot(all_data['Date'], all_data['Close'], label="Historical", color="blue")
    future_dates = pd.date_range(all_data['Date'].iloc[-1], periods=8, freq='D')[1:]
    plt.plot(future_dates, predicted_prices, label="Predicted", color="red", linestyle='--')
    plt.legend()
    plt.title("Historical vs Predicted Closing Prices")
    st.pyplot(plt)

def plot_month_year_data(df):
    plt.figure(figsize=(10, 4))
    plt.plot(df['Date'], df['Close'], marker='o', linestyle='-')
    plt.title("Stock Prices")
    plt.xlabel("Date")
    plt.ylabel("Close Price")
    plt.grid(True)
    st.pyplot(plt)

def show_buy_sell_recommendation(predicted_prices, last_price):
    st.subheader("📊 Stock Prediction Summary & Recommendation")

    rows = []
    for i, price in enumerate(predicted_prices):
        change_pct = ((price - last_price) / last_price) * 100
        if change_pct > 1:
            action = "Buy 🟢"
        elif change_pct < -1:
            action = "Sell 🔻"
        else:
            action = "Hold 🤝"
        rows.append((f"Day {i+1}", f"${price:.2f}", f"{change_pct:+.2f}%", action))

    df = pd.DataFrame(rows, columns=["Day", "Price ($)", "Change %", "Action"])
    st.dataframe(df)







