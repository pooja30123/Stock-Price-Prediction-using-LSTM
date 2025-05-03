import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import numpy as np
from src.download_data import load_historical_data
from datetime import datetime

st.set_page_config(page_title="Historical Stock Data", layout="wide")

# Load CSS
def load_css():
    with open('style.css', 'r') as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

load_css()

# App header
st.markdown("<h1 style='text-align: center'>📅 Historical Stock Data</h1>", unsafe_allow_html=True)

ticker = st.session_state.get('selected_ticker', None)

# Helper functions
def get_month_name(month_num):
    return datetime(2023, month_num, 1).strftime('%B')

def create_price_chart(df):
    """Create a simple line chart for stock price with volume as bars"""
    # Create figure
    fig = go.Figure()
    
    # Add price line
    fig.add_trace(
        go.Scatter(
            x=df['Date'],
            y=df['Close'],
            mode='lines',
            name="Price",
            line=dict(color='#3b82f6', width=2)
        )
    )
    
    # Add moving averages
    df['MA5'] = df['Close'].rolling(window=5).mean()
    df['MA20'] = df['Close'].rolling(window=20).mean()
    
    fig.add_trace(
        go.Scatter(
            x=df['Date'],
            y=df['MA5'],
            mode='lines',
            name="5-day MA",
            line=dict(color='#f59e0b', width=2, dash='dot')
        )
    )
    
    fig.add_trace(
        go.Scatter(
            x=df['Date'],
            y=df['MA20'],
            mode='lines',
            name="20-day MA",
            line=dict(color='#ef4444', width=2, dash='dot')
        )
    )
    
    # Update layout
    fig.update_layout(
        xaxis_title="Date",
        yaxis_title="Price (USD)",
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=0, r=0, t=10, b=0),
        height=400,
        template="plotly_white",
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)'
    )
    
    # Add range selector
    fig.update_layout(
        xaxis=dict(
            rangeselector=dict(
                buttons=list([
                    dict(count=7, label="1w", step="day", stepmode="backward"),
                    dict(count=1, label="1m", step="month", stepmode="backward"),
                    dict(count=3, label="3m", step="month", stepmode="backward"),
                    dict(step="all")
                ]),
                bgcolor="#e2e8f0",
                activecolor="#3b82f6"
            ),
            type="date"
        )
    )
    
    return fig

def calculate_stats(df):
    """Calculate key statistics for the selected period"""
    first_price = df['Open'].iloc[0]
    last_price = df['Close'].iloc[-1]
    highest_price = df['High'].max()
    lowest_price = df['Low'].min()
    avg_price = df['Close'].mean()
    price_change = last_price - first_price
    price_change_pct = (price_change / first_price) * 100
    daily_returns = df['Close'].pct_change().dropna()
    volatility = daily_returns.std() * 100  # Daily volatility
    
    return {
        'first_price': first_price,
        'last_price': last_price,
        'highest_price': highest_price,
        'lowest_price': lowest_price,
        'avg_price': avg_price,
        'price_change': price_change,
        'price_change_pct': price_change_pct,
        'volatility': volatility
    }

if ticker:
    st.markdown(f"""
    <div class="card">
        <h2>Historical Data for {ticker}</h2>
    </div>
    """, unsafe_allow_html=True)
    
    # Load historical data
    df = load_historical_data(ticker)
    if df is not None and not df.empty:
        df['Date'] = pd.to_datetime(df['Date'])
        df['Year'] = df['Date'].dt.year
        df['Month'] = df['Date'].dt.month
        
        # Filter options
        col1, col2 = st.columns(2)
        with col1:
            year = st.selectbox("Select Year", sorted(df['Year'].unique(), reverse=True))
        with col2:
            month = st.selectbox("Select Month", sorted(df[df['Year'] == year]['Month'].unique()), 
                               format_func=get_month_name)
        
        # Filter the data
        filtered_df = df[(df['Year'] == year) & (df['Month'] == month)]
        
        if not filtered_df.empty:
            # First show the chart
            st.markdown('<div class="card">', unsafe_allow_html=True)
            fig = create_price_chart(filtered_df)
            st.plotly_chart(fig, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Calculate statistics
            stats = calculate_stats(filtered_df)
            
            # Display statistics cards
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.subheader("📊 Key Statistics")
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.markdown(f"""
                <div class="stat-card">
                    <h4>Period Change</h4>
                    <p class="{'up-value' if stats['price_change'] >= 0 else 'down-value'}">
                        {stats['price_change_pct']:.2f}% 
                        ({'+' if stats['price_change'] >= 0 else ''}{stats['price_change']:.2f} USD)
                    </p>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                st.markdown(f"""
                <div class="stat-card">
                    <h4>Price Range</h4>
                    <p>
                        ${stats['lowest_price']:.2f} - ${stats['highest_price']:.2f}
                    </p>
                </div>
                """, unsafe_allow_html=True)
            
            with col3:
                st.markdown(f"""
                <div class="stat-card">
                    <h4>Average Price</h4>
                    <p>
                        ${stats['avg_price']:.2f}
                    </p>
                </div>
                """, unsafe_allow_html=True)
            
            with col4:
                st.markdown(f"""
                <div class="stat-card">
                    <h4>Daily Volatility</h4>
                    <p>
                        {stats['volatility']:.2f}%
                    </p>
                </div>
                """, unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

            # Show the table of filtered data
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.subheader(f"📅 Stock Data for {get_month_name(month)} {year}")
            display_df = filtered_df[['Date', 'Open', 'High', 'Low', 'Close', 'Volume']].copy()
            display_df['Date'] = display_df['Date'].dt.strftime('%Y-%m-%d')
            st.dataframe(display_df, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
        else:
            st.error("No data available for the selected month and year.")
    else:
        st.error("Historical data could not be loaded or is empty.")
else:
    st.warning("Please select a ticker from the main page.")

# Add a navigation menu
st.sidebar.markdown("### Navigation")
if st.sidebar.button("🏠 Home"):
    st.switch_page("streamlit_app.py")
if st.sidebar.button("🔮 Predictions"):
    st.switch_page("pages/1_Predict_Next_7_Days.py")

# Footer
st.markdown('<div class="footer">', unsafe_allow_html=True)
st.markdown("© 2025 Stock Prediction App. This is for educational purposes only.", unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)
            
