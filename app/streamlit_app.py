import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import tensorflow as tf
from datetime import datetime, timedelta
import warnings
import time
import os
import sys
warnings.filterwarnings('ignore')

# Import your existing classes (adjust import path as needed)
# from your_training_file import StockLSTMPredictor, create_ml_ready_dataset

# Configure Streamlit page
st.set_page_config(
    page_title="NVIDIA Stock Prediction",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

class StreamlitDemo:
    def __init__(self, model_path='final_lstm_model.h5'):
        self.model_path = model_path
        self.model = None
        self.predictor = None
        
    def load_model(self):
        """Load the trained model"""
        try:
            self.model = tf.keras.models.load_model(self.model_path)
            return True
        except Exception as e:
            st.error(f"Error loading model: {str(e)}")
            return False
    
    def initialize_predictor(self):
        """Initialize your existing StockLSTMPredictor"""
        try:
            # Import and initialize your existing predictor
            from train_model import StockLSTMPredictor
            self.predictor = StockLSTMPredictor(
                sequence_length=90,
                prediction_horizon=1
            )
            self.predictor.model = self.model
            return True
        except Exception as e:
            st.error(f"Error initializing predictor: {str(e)}")
            return False

@st.cache_data(ttl=300)  # Cache for 5 minutes
def get_fresh_dataset(ticker="NVDA", company_name="NVIDIA Corporation", api_key=None, force_refresh=True):
    """Get fresh dataset using your existing function"""
    try:
        # Import your dataset creation function
        ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
        SRC_DIR = os.path.join(ROOT_DIR, 'src')
        sys.path.append(SRC_DIR)
        from preprocess_data import create_ml_ready_dataset
        
        # Get fresh data with all features
        df = create_ml_ready_dataset(
            ticker=ticker,
            company_name=company_name,
            api_key=api_key,
            start_date="2024-01-01",
            force_refresh=force_refresh
        )
        return df
    except Exception as e:
        st.error(f"Error creating dataset: {str(e)}")
        return None

@st.cache_data(ttl=300)
def prepare_prediction_data(df):
    """Use your existing prepare_data method to get the latest sequence"""
    try:
        # Import your predictor class
        ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
        SRC_DIR = os.path.join(ROOT_DIR, 'src')
        sys.path.append(SRC_DIR)
        from train_model import StockLSTMPredictor  # Adjust import path
        
        # Initialize predictor for data preparation
        temp_predictor = StockLSTMPredictor(sequence_length=90)
        
        # Use your existing prepare_data method but only to get data processing
        # We'll extract the latest sequence for prediction
        train_data, val_data, test_data = temp_predictor.prepare_data(
            df, 
            target_column='returns',
            test_size=0.01,  # Minimal test set since we want most data
            validation_size=0.01
        )
        
        # Get the scalers that were fitted
        scaler_features = temp_predictor.scaler_features
        scaler_target = temp_predictor.scaler_target
        feature_columns = temp_predictor.feature_columns
        
        # Get all processed data by combining train/val/test
        X_all = np.concatenate([train_data[0], val_data[0], test_data[0]], axis=0)
        y_all = np.concatenate([train_data[1], val_data[1], test_data[1]], axis=0)
        
        # Get the most recent sequence for prediction
        latest_sequence = X_all[-1:] if len(X_all) > 0 else None
        
        return {
            'latest_sequence': latest_sequence,
            'scaler_features': scaler_features,
            'scaler_target': scaler_target,
            'feature_columns': feature_columns,
            'processed_data': (X_all, y_all),
            'original_df': df
        }
        
    except Exception as e:
        st.error(f"Error preparing prediction data: {str(e)}")
        return None

def make_predictions(model, latest_sequence, n_predictions=10):
    """Make multiple step predictions"""
    if model is None or latest_sequence is None:
        return None
    
    predictions = []
    current_sequence = latest_sequence.copy()
    
    try:
        for i in range(n_predictions):
            # Predict next return
            pred_return = model.predict(current_sequence, verbose=0)[0, 0]
            predictions.append(pred_return)
            
            # For multi-step prediction, we'd need to update the sequence
            # This is simplified - in practice, you'd need to reconstruct the full feature set
            # For now, we'll just make single-step predictions
            
        return predictions
    except Exception as e:
        st.error(f"Error making predictions: {str(e)}")
        return None

def create_comprehensive_chart(df, predictions=None, processed_data=None, model=None):
    """Create comprehensive charts showing actual data and predictions"""
    
    # Create subplots
    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=(
            'NVIDIA Stock Price (Last 90 Days)', 
            'Daily Returns (Actual vs Historical Fit)'
        ),
        vertical_spacing=0.08,
        row_heights=[0.6, 0.4]
    )
    
    # Get recent data for display (last 90 days)
    recent_df = df.tail(90).copy()
    
    # 1. Price Chart
    fig.add_trace(
        go.Scatter(
            x=recent_df['Date'] if 'Date' in recent_df.columns else recent_df.index,
            y=recent_df['Close'] if 'Close' in recent_df.columns else recent_df.get('close', recent_df.get('price')),
            mode='lines',
            name='NVDA Price',
            line=dict(color='#1f77b4', width=2)
        ),
        row=1, col=1
    )
    
    # 2. Returns Chart - Actual vs Historical
    if 'returns' in recent_df.columns:
        returns_data = recent_df['returns'].dropna()
        dates = recent_df['Date'][1:] if 'Date' in recent_df.columns else recent_df.index[1:]
        
        # Actual returns
        colors = ['red' if x < 0 else 'green' for x in returns_data]
        fig.add_trace(
            go.Bar(
                x=dates,
                y=returns_data,
                name='Actual Returns',
                marker_color=colors,
                opacity=0.7
            ),
            row=2, col=1
        )
        
        # Add model fit if available
        if processed_data is not None:
            X_all, y_all = processed_data
            if len(y_all) > 0:
                # Get model predictions for historical data (last portion)
                try:
                    historical_pred = model.predict(X_all[-len(returns_data):], verbose=0).flatten()
                    fig.add_trace(
                        go.Scatter(
                            x=dates,
                            y=historical_pred,
                            mode='lines',
                            name='Model Fit',
                            line=dict(color='orange', width=2, dash='dot'),
                            opacity=0.8
                        ),
                        row=2, col=1
                    )
                except:
                    pass  # Skip if prediction fails
    
    # Update layout
    fig.update_layout(
        title='NVIDIA LSTM Stock Prediction Analysis',
        template='plotly_white',
        height=800,
        showlegend=True,
        hovermode='x unified'
    )
    
    # Update axes
    fig.update_xaxes(title_text="Date", row=3, col=1)
    fig.update_yaxes(title_text="Price ($)", row=1, col=1)
    fig.update_yaxes(title_text="Returns", row=2, col=1)
    fig.update_yaxes(title_text="Predicted Returns", row=3, col=1)
    
    return fig

def display_model_metrics(df, predictions, processed_data):
    """Display comprehensive model and prediction metrics"""
    
    col1, col2, col3, col4 = st.columns(4)
    
    # Current stock metrics
    current_price = df['Close'].iloc[-1] if 'Close' in df.columns else 0
    price_change = current_price - df['Close'].iloc[-2] if len(df) > 1 else 0
    price_change_pct = (price_change / df['Close'].iloc[-2]) * 100 if len(df) > 1 else 0
    current_return = df['returns'].iloc[-1] if 'returns' in df.columns and not pd.isna(df['returns'].iloc[-1]) else 0
    
    with col1:
        st.metric(
            "Current Price", 
            f"${current_price:.2f}",
            f"{price_change:+.2f} ({price_change_pct:+.2f}%)"
        )
    
    with col2:
        st.metric(
            "Latest Return",
            f"{current_return:.4f}",
            f"{current_return*100:.2f}%"
        )
    
    with col3:
        if predictions and len(predictions) > 0:
            next_pred = predictions[0]
            st.metric(
                "Next Day Prediction",
                f"{next_pred:.4f}",
                f"{next_pred*100:.2f}%"
            )

def main():
    """Main Streamlit app"""
    st.title("📈 NVIDIA LSTM Stock Prediction")
    st.markdown("*Real-time predictions using your trained LSTM model*")
    
    # Sidebar controls
    st.sidebar.header("⚙️ Settings")
    
    # API key input (if needed)
    import streamlit as st
    api_key = st.secrets['API_KEY'] if 'API_KEY' in st.secrets else None
    
    # Prediction settings
    n_predictions = 1
    force_refresh = st.sidebar.checkbox("Force data refresh", value=False, help="Force refresh of underlying data")
    show_technical_details = st.sidebar.checkbox("Show technical details", value=True)
    
    # Initialize demo
    demo = StreamlitDemo()
    
    # Load model
    with st.spinner("🔄 Loading trained model..."):
        if not demo.load_model():
            st.error("❌ Failed to load model. Make sure 'final_lstm_model.h5' is in the current directory.")
            st.info("""
            **Required files:**
            - `final_lstm_model.h5` - Your trained LSTM model
            - Make sure your training script is importable
            """)
            return
    
    # Get fresh dataset using your function
    with st.spinner("📊 Fetching latest data and creating features..."):
        df = get_fresh_dataset(
            ticker="NVDA",
            company_name="NVIDIA Corporation", 
            api_key=api_key if api_key else None,
            force_refresh=force_refresh
        )
        
        if df is None:
            st.error("❌ Failed to get dataset. Check your data creation function.")
            return
    
    # Prepare data for prediction using your existing pipeline
    with st.spinner("🔧 Preparing data for prediction..."):
        prediction_data = prepare_prediction_data(df)
        
        if prediction_data is None:
            st.error("❌ Failed to prepare prediction data.")
            return
    
    # Make predictions
    with st.spinner("🤖 Making predictions..."):
        predictions = make_predictions(
            demo.model, 
            prediction_data['latest_sequence'], 
            n_predictions
        )
    
    # Display metrics
    st.subheader("📊 Current Status")
    display_model_metrics(df, predictions, prediction_data.get('processed_data'))
    
    # Create and display comprehensive chart
    st.subheader("📈 Analysis & Predictions")
    fig = create_comprehensive_chart(
        df, 
        predictions, 
        prediction_data.get('processed_data'),
        demo.model
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Show technical details
    if show_technical_details:
        st.subheader("🔬 Technical Details")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.info(f"""
            **Dataset Info:**
            - Total records: {len(df):,}
            - Date range: {df['Date'].min().strftime('%Y-%m-%d')} to {df['Date'].max().strftime('%Y-%m-%d')}
            - Features: {len(prediction_data['feature_columns'])} technical indicators
            - Sequence length: 90 days
            - Last updated: {datetime.now().strftime('%H:%M:%S')}
            """)
        
        with col2:
            if predictions:
                st.success(f"""
                **Prediction Info:**
                - Predictions: {len(predictions)} days ahead
                - Model: Advanced LSTM
                - Target: Daily returns
                - Next prediction: {predictions[0]:.4f} ({predictions[0]*100:.2f}%)
                - Prediction range: {min(predictions):.4f} to {max(predictions):.4f}
                """)
        
        # Show feature columns
        with st.expander("📋 Feature Columns Used"):
            st.write(prediction_data['feature_columns'])
        
        # Show recent predictions vs actual (if available)
        if prediction_data.get('processed_data') is not None:
            with st.expander("🎯 Recent Model Performance"):
                X_all, y_all = prediction_data['processed_data']
                if len(X_all) > 10:
                    try:
                        recent_pred = demo.model.predict(X_all[-10:], verbose=0).flatten()
                        recent_actual = y_all[-10:].flatten()
                        
                        perf_df = pd.DataFrame({
                            'Actual': recent_actual,
                            'Predicted': recent_pred,
                            'Error': recent_actual - recent_pred
                        })
                        perf_df = perf_df.round(6)
                        st.dataframe(perf_df, use_container_width=True)
                        
                        mae = np.mean(np.abs(recent_actual - recent_pred))
                        st.metric("Recent MAE (10 days)", f"{mae:.6f}")
                        
                    except Exception as e:
                        st.write(f"Could not calculate recent performance: {e}")
    
    # Show recent data
    st.subheader("📅 Recent Data")
    display_cols = ['Date', 'Close', 'returns']
    if 'Volume' in df.columns:
        display_cols.append('Volume')
    
    recent_data = df[display_cols].tail(10).copy()
    if 'returns' in recent_data.columns:
        recent_data['returns'] = recent_data['returns'].round(6)
    if 'Close' in recent_data.columns:
        recent_data['Close'] = recent_data['Close'].round(2)
    
    st.dataframe(recent_data, use_container_width=True)
    
    # Control buttons
    col1, col2, col3 = st.columns([1, 1, 2])
    
    with col1:
        if st.button("🔄 Refresh Data"):
            st.cache_data.clear()
            st.rerun()
    
    # with col2:
    #     if st.button("💾 Download Predictions"):
    #         if predictions:
    #             pred_df = pd.DataFrame({
    #                 'Day': range(1, len(predictions) + 1),
    #                 'Predicted_Return': predictions,
    #                 'Predicted_Return_Pct': [p * 100 for p in predictions]
    #             })
    #             csv = pred_df.to_csv(index=False)
    #             st.download_button(
    #                 label="Download CSV",
    #                 data=csv,
    #                 file_name=f'nvda_predictions_{datetime.now().strftime("%Y%m%d_%H%M")}.csv',
    #                 mime='text/csv'
    #             )
    
    with col3:
        st.markdown("*Predictions are for educational purposes only. Not financial advice.*")

if __name__ == "__main__":
    main()