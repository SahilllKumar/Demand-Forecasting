# demand_forecast_app.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from prophet import Prophet
from prophet.plot import plot_plotly
from sklearn.metrics import mean_absolute_error, mean_squared_error
import io

# App configuration
st.set_page_config(page_title="Demand Forecasting", layout="wide")

# Main header
st.title("📈 AI-Powered Demand Forecasting")
st.markdown("Upload historical demand data to generate future predictions")

# Sidebar for inputs
with st.sidebar:
    st.header("⚙️ Configuration")
    uploaded_file = st.file_uploader("Upload CSV data", type=["csv"])
    
    st.subheader("Forecasting Parameters")
    periods = st.number_input("Forecast Period (days)", min_value=1, max_value=365, value=90)
    seasonality_mode = st.selectbox("Seasonality Mode", ["additive", "multiplicative"])
    confidence_interval = st.slider("Confidence Interval", 0.7, 0.99, 0.95)
    
    st.subheader("Advanced Options")
    daily_seasonality = st.checkbox("Daily Seasonality", value=False)
    weekly_seasonality = st.checkbox("Weekly Seasonality", value=True)
    yearly_seasonality = st.checkbox("Yearly Seasonality", value=True)
    
    st.markdown("---")
    st.info("Ensure your data has columns: 'ds' (date) and 'y' (demand)")

# Initialize session state
if "forecast" not in st.session_state:
    st.session_state.forecast = None

# Sample data for demonstration
@st.cache_data
def load_sample_data():
    dates = pd.date_range(start="2019-01-01", end="2023-12-31")
    demand = np.sin(np.arange(len(dates)) * 50 + 100) + np.random.normal(0, 10, len(dates))
    return pd.DataFrame({"ds": dates, "y": np.abs(demand)})

# Data processing
def process_data(df):
    df = df.copy()
    df['ds'] = pd.to_datetime(df['ds'])
    df = df.sort_values('ds')
    return df[['ds', 'y']].dropna()

# Modeling function
def create_forecast(df, periods, seasonality_mode, confidence_interval, 
                    daily_seasonality, weekly_seasonality, yearly_seasonality):
    model = Prophet(
        interval_width=confidence_interval,
        seasonality_mode=seasonality_mode,
        daily_seasonality=daily_seasonality,
        weekly_seasonality=weekly_seasonality,
        yearly_seasonality=yearly_seasonality
    )
    
    model.fit(df)
    future = model.make_future_dataframe(periods=periods)
    forecast = model.predict(future)
    
    # Calculate metrics
    merged = forecast[['ds', 'yhat']].merge(df, on='ds', how='inner')
    mae = mean_absolute_error(merged['y'], merged['yhat'])
    rmse = np.sqrt(mean_squared_error(merged['y'], merged['yhat']))
    
    return forecast, model, mae, rmse

# Main app logic
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
else:
    df = load_sample_data()
    st.info("Using sample data. Upload your own CSV to customize")

df_processed = process_data(df)

# Create tabs
tab1, tab2, tab3, tab4 = st.tabs(["📊 Data", "🔮 Forecast", "📈 Trends", "📥 Export"])

with tab1:
    st.subheader("Historical Demand Data")
    st.dataframe(df_processed, height=300)
    
    fig_raw = go.Figure()
    fig_raw.add_trace(go.Scatter(x=df_processed['ds'], y=df_processed['y'], 
                                name="Actual Demand", line=dict(color='royalblue')))
    fig_raw.update_layout(title="Historical Demand Pattern",
                         xaxis_title="Date",
                         yaxis_title="Demand")
    st.plotly_chart(fig_raw, use_container_width=True)

# Forecasting
with tab2:
    if st.button("Generate Forecast") or st.session_state.forecast is not None:
        with st.spinner("Training model and forecasting..."):
            forecast, model, mae, rmse = create_forecast(
                df_processed, periods, seasonality_mode, confidence_interval,
                daily_seasonality, weekly_seasonality, yearly_seasonality
            )
            st.session_state.forecast = forecast
            st.session_state.model = model
            st.session_state.mae = mae
            st.session_state.rmse = rmse
            
    if st.session_state.forecast is not None:
        st.success("Forecast generated successfully!")
        
        # Show metrics
        col1, col2 = st.columns(2)
        col1.metric("MAE (Accuracy)", f"{st.session_state.mae:.2f}")
        col2.metric("RMSE (Accuracy)", f"{st.session_state.rmse:.2f}")
        
        # Plot forecast
        fig_forecast = plot_plotly(st.session_state.model, st.session_state.forecast)
        fig_forecast.update_layout(title=f"Demand Forecast - Next {periods} Days",
                                  xaxis_title="Date",
                                  yaxis_title="Demand")
        st.plotly_chart(fig_forecast, use_container_width=True)
        
        # Show forecast components
        st.subheader("Forecast Components")
        fig_components = st.session_state.model.plot_components(st.session_state.forecast)
        st.pyplot(fig_components)

with tab3:
    if st.session_state.forecast is not None:
        st.subheader("Trend Analysis")
        
        # Weekly seasonality
        if weekly_seasonality:
            fig_weekly = go.Figure()
            fig_weekly.add_trace(go.Bar(
                x=['Sun', 'Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat'],
                y=st.session_state.forecast['weekly'].iloc[:7].values,
                name="Weekly Seasonality"
            ))
            fig_weekly.update_layout(title="Weekly Demand Pattern")
            st.plotly_chart(fig_weekly, use_container_width=True)
        
        # Yearly seasonality
        if yearly_seasonality:
            fig_yearly = go.Figure()
            fig_yearly.add_trace(go.Scatter(
                x=pd.date_range("2023-01-01", periods=365, freq='D'),
                y=st.session_state.forecast['yearly'].iloc[:365].values,
                name="Yearly Seasonality"
            ))
            fig_yearly.update_layout(title="Yearly Demand Pattern")
            st.plotly_chart(fig_yearly, use_container_width=True)

with tab4:
    if st.session_state.forecast is not None:
        st.subheader("Download Forecast Results")
        
        # Prepare data for download
        export_df = st.session_state.forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]
        export_df.columns = ['Date', 'Forecast', 'Lower Bound', 'Upper Bound']
        
        # Show preview
        st.dataframe(export_df.tail(periods))
        
        # Download buttons
        csv_buffer = io.StringIO()
        export_df.to_csv(csv_buffer, index=False)
        
        col1, col2 = st.columns(2)
        col1.download_button(
            label="Download CSV",
            data=csv_buffer.getvalue(),
            file_name="demand_forecast.csv",
            mime="text/csv"
        )
        
        col2.download_button(
            label="Download Full Report (PDF)",
            data=csv_buffer.getvalue(),
            file_name="forecast_report.pdf",
            mime="application/pdf",
            disabled=True  # Enable if report generation is implemented
        )

# Footer
st.markdown("---")
st.markdown("**Demand Forecasting App** • Powered by Prophet AI • [GitHub Repo](https://github.com/SahilllKumar/Demand-Forecasting)")
