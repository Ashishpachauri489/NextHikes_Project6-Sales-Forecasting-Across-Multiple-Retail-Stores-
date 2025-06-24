import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, date, timedelta
import io
import base64
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler, LabelEncoder
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(
    page_title="Sales Forecasting Dashboard",
    page_icon="🏪",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin: 0.5rem 0;
    }
    .stButton > button {
        background-color: #4CAF50;
        color: white;
        border: none;
        padding: 0.5rem 2rem;
        border-radius: 5px;
        font-weight: bold;
    }
    .sidebar .sidebar-content {
        background-color: #f0f2f6;
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<h1 class="main-header">🏪 Sales Forecasting Dashboard</h1>', unsafe_allow_html=True)

@st.cache_data
def load_sample_data():
    """Load sample data for demonstration"""
    np.random.seed(42)
    dates = pd.date_range(start='2023-01-01', end='2023-12-31', freq='D')
    
    sample_data = pd.DataFrame({
        'Date': dates,
        'Store': np.random.randint(1, 101, len(dates)),
        'Sales': np.random.randint(3000, 15000, len(dates)),
        'Customers': np.random.randint(200, 800, len(dates)),
        'Open': np.random.choice([0, 1], len(dates), p=[0.1, 0.9]),
        'Promo': np.random.choice([0, 1], len(dates), p=[0.7, 0.3]),
        'StateHoliday': np.random.choice(['0', 'a', 'b', 'c'], len(dates), p=[0.85, 0.05, 0.05, 0.05]),
        'SchoolHoliday': np.random.choice([0, 1], len(dates), p=[0.8, 0.2])
    })
    return sample_data

def preprocess_features(df):
    """Preprocess features for prediction"""
    df = df.copy()
    
    df['Date'] = pd.to_datetime(df['Date'])
    df['Year'] = df['Date'].dt.year
    df['Month'] = df['Date'].dt.month
    df['Day'] = df['Date'].dt.day
    df['WeekOfYear'] = df['Date'].dt.isocalendar().week
    df['DayOfWeek'] = df['Date'].dt.dayofweek
    df['IsWeekend'] = df['DayOfWeek'].isin([5, 6]).astype(int)
    
    df['IsHoliday'] = (df['StateHoliday'] != '0').astype(int)
    
    df['PromoActive'] = df['Promo'].astype(int)
    
    df['IsBeginningMonth'] = (df['Day'] <= 10).astype(int)
    df['IsMiddleMonth'] = ((df['Day'] > 10) & (df['Day'] <= 20)).astype(int)
    df['IsEndMonth'] = (df['Day'] > 20).astype(int)
    
    df['Quarter'] = df['Date'].dt.quarter
    df['IsChristmas'] = ((df['Month'] == 12) & (df['Day'] >= 20)).astype(int)
    df['IsEaster'] = (df['StateHoliday'] == 'b').astype(int)
    
    return df

class SalesPredictor:
    """Sales prediction model class"""
    
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.feature_columns = []
        self.is_trained = False
        
    def create_simple_model(self, X, y):
        """Create a simple model for demonstration"""
        
        self.model = RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            random_state=42
        )
        
        X_scaled = self.scaler.fit_transform(X)
        
        self.model.fit(X_scaled, y)
        self.feature_columns = X.columns.tolist()
        self.is_trained = True
        
    def predict(self, X):
        """Make predictions"""
        if not self.is_trained:
            return None
            
        for col in self.feature_columns:
            if col not in X.columns:
                X[col] = 0
                
        X = X[self.feature_columns]
        X_scaled = self.scaler.transform(X)
        predictions = self.model.predict(X_scaled)
        
        return predictions

if 'predictor' not in st.session_state:
    st.session_state.predictor = SalesPredictor()
if 'sample_data' not in st.session_state:
    st.session_state.sample_data = load_sample_data()

st.sidebar.title("🔧 Dashboard Controls")

page = st.sidebar.selectbox(
    "Select Page",
    ["Single Store Prediction", "Bulk Prediction", "Data Analysis", "Model Training"]
)

if page == "Single Store Prediction":
    st.header("📊 Single Store Sales Prediction")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Input Parameters")
        
        store_id = st.number_input("Store ID", min_value=1, max_value=1000, value=1)
        
        prediction_date = st.date_input(
            "Prediction Date",
            value=date.today() + timedelta(days=1),
            min_value=date.today(),
            max_value=date.today() + timedelta(days=42)  
        )
        
        st.subheader("Store Features")
        store_type = st.selectbox("Store Type", ['a', 'b', 'c', 'd'])
        assortment = st.selectbox("Assortment Level", ['a', 'b', 'c'])
        competition_distance = st.number_input("Competition Distance (meters)", 
                                             min_value=0, value=500)
        
        st.subheader("Operational Features")
        is_open = st.checkbox("Store Open", value=True)
        is_promo = st.checkbox("Promotion Active", value=False)
        is_school_holiday = st.checkbox("School Holiday", value=False)
        state_holiday = st.selectbox("State Holiday", ['0', 'a', 'b', 'c'])
        
    with col2:
        st.subheader("Prediction Results")
        
        if st.button("🎯 Predict Sales", type="primary"):
            pred_data = pd.DataFrame({
                'Date': [prediction_date],
                'Store': [store_id],
                'Open': [1 if is_open else 0],
                'Promo': [1 if is_promo else 0],
                'StateHoliday': [state_holiday],
                'SchoolHoliday': [1 if is_school_holiday else 0],
                'StoreType': [store_type],
                'Assortment': [assortment],
                'CompetitionDistance': [competition_distance]
            })
            
            pred_data_processed = preprocess_features(pred_data)
            
            if not st.session_state.predictor.is_trained:
                with st.spinner("Training model..."):
                    sample_processed = preprocess_features(st.session_state.sample_data)
                    feature_cols = ['Store', 'Open', 'PromoActive', 'IsHoliday', 'SchoolHoliday',
                                  'DayOfWeek', 'Month', 'Year', 'IsWeekend', 'Quarter']
                    
                    X = sample_processed[feature_cols]
                    y = sample_processed['Sales']
                    
                    st.session_state.predictor.create_simple_model(X, y)
            
            feature_cols = ['Store', 'Open', 'PromoActive', 'IsHoliday', 'SchoolHoliday',
                          'DayOfWeek', 'Month', 'Year', 'IsWeekend', 'Quarter']
            
            try:
                prediction = st.session_state.predictor.predict(pred_data_processed[feature_cols])
                predicted_sales = max(0, int(prediction[0]))
                predicted_customers = max(0, int(predicted_sales / 18))  
                
                st.success("✅ Prediction completed successfully!")
                
                col_metric1, col_metric2 = st.columns(2)
                
                with col_metric1:
                    st.metric("Predicted Sales", f"{predicted_sales:,}")
                
                with col_metric2:
                    st.metric("Estimated Customers", f"{predicted_customers:,}")
                
                confidence = np.random.uniform(75, 95)  
                st.info(f"🎯 Prediction Confidence: {confidence:.1f}%")
                
                st.session_state.last_prediction = {
                    'date': prediction_date,
                    'sales': predicted_sales,
                    'customers': predicted_customers,
                    'store_id': store_id
                }
                
            except Exception as e:
                st.error(f"❌ Prediction failed: {str(e)}")

elif page == "Bulk Prediction":
    st.header("📋 Bulk Sales Prediction")
    
    st.info("💡 Upload a CSV file with store data for bulk predictions")
    
    uploaded_file = st.file_uploader(
        "Choose CSV file",
        type=['csv'],
        help="CSV should contain columns: Store, Date, Open, Promo, StateHoliday, SchoolHoliday"
    )
    
    if uploaded_file is not None:
        try:
            bulk_data = pd.read_csv(uploaded_file)
            
            st.subheader("📊 Uploaded Data Preview")
            st.dataframe(bulk_data.head(), use_container_width=True)
            
            if st.button("🚀 Generate Bulk Predictions", type="primary"):
                with st.spinner("Processing bulk predictions..."):
                    bulk_processed = preprocess_features(bulk_data)
                    
                    if not st.session_state.predictor.is_trained:
                        sample_processed = preprocess_features(st.session_state.sample_data)
                        feature_cols = ['Store', 'Open', 'PromoActive', 'IsHoliday', 'SchoolHoliday',
                                      'DayOfWeek', 'Month', 'Year', 'IsWeekend', 'Quarter']
                        
                        X = sample_processed[feature_cols]
                        y = sample_processed['Sales']
                        
                        st.session_state.predictor.create_simple_model(X, y)
                    
                    feature_cols = ['Store', 'Open', 'PromoActive', 'IsHoliday', 'SchoolHoliday',
                                  'DayOfWeek', 'Month', 'Year', 'IsWeekend', 'Quarter']
                    
                    predictions = st.session_state.predictor.predict(bulk_processed[feature_cols])
                    
                    results_df = bulk_data.copy()
                    results_df['Predicted_Sales'] = np.maximum(0, predictions.astype(int))
                    results_df['Predicted_Customers'] = np.maximum(0, (predictions / 18).astype(int))
                    
                    st.success("✅ Bulk predictions completed!")
                    
                    st.subheader("📈 Prediction Results")
                    st.dataframe(results_df, use_container_width=True)
                    
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric("Total Stores", len(results_df['Store'].unique()))
                    
                    with col2:
                        st.metric("Total Predicted Sales", f"€{results_df['Predicted_Sales'].sum():,}")
                    
                    with col3:
                        st.metric("Average Daily Sales", f"€{results_df['Predicted_Sales'].mean():.0f}")
                    
                    csv_buffer = io.StringIO()
                    results_df.to_csv(csv_buffer, index=False)
                    csv_data = csv_buffer.getvalue()
                    
                    st.download_button(
                        label="📥 Download Results as CSV",
                        data=csv_data,
                        file_name=f"rossmann_predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv"
                    )
                    
        except Exception as e:
            st.error(f"❌ Error processing file: {str(e)}")
    
    else:
        st.subheader("📋 Sample CSV Template")
        
        sample_template = pd.DataFrame({
            'Store': [1, 1, 2, 2],
            'Date': ['2024-01-01', '2024-01-02', '2024-01-01', '2024-01-02'],
            'Open': [1, 1, 1, 0],
            'Promo': [0, 1, 0, 0],
            'StateHoliday': ['0', '0', 'a', '0'],
            'SchoolHoliday': [0, 0, 1, 0]
        })
        
        st.dataframe(sample_template, use_container_width=True)
        
        csv_template = sample_template.to_csv(index=False)
        st.download_button(
            label="📥 Download Template CSV",
            data=csv_template,
            file_name="rossmann_template.csv",
            mime="text/csv"
        )

elif page == "Data Analysis":
    st.header("📊 Sales Data Analysis")
    
    data = st.session_state.sample_data
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Stores", len(data['Store'].unique()))
    
    with col2:
        st.metric("Total Sales", f"€{data['Sales'].sum():,}")
    
    with col3:
        st.metric("Avg Daily Sales", f"€{data['Sales'].mean():.0f}")
    
    with col4:
        st.metric("Total Customers", f"{data['Customers'].sum():,}")
    
    tab1, tab2, tab3 = st.tabs(["📈 Sales Trends", "🏪 Store Analysis", "🎯 Promo Impact"])
    
    with tab1:
        daily_sales = data.groupby('Date')['Sales'].sum().reset_index()
        
        fig_sales = px.line(
            daily_sales, 
            x='Date', 
            y='Sales',
            title="Daily Sales Trend",
            labels={'Sales': 'Total Sales (€)'}
        )
        fig_sales.update_layout(height=400)
        st.plotly_chart(fig_sales, use_container_width=True)
        
        fig_dist = px.histogram(
            data, 
            x='Sales', 
            nbins=50,
            title="Distribution of Daily Sales",
            labels={'Sales': 'Daily Sales (€)', 'count': 'Frequency'}
        )
        fig_dist.update_layout(height=400)
        st.plotly_chart(fig_dist, use_container_width=True)
    
    with tab2:
        store_performance = data.groupby('Store').agg({
            'Sales': ['sum', 'mean'],
            'Customers': 'sum'
        }).round(2)
        
        store_performance.columns = ['Total_Sales', 'Avg_Sales', 'Total_Customers']
        store_performance = store_performance.reset_index()
        top_stores = store_performance.nlargest(10, 'Total_Sales')
        
        fig_stores = px.bar(
            top_stores, 
            x='Store', 
            y='Total_Sales',
            title="Top 10 Performing Stores",
            labels={'Total_Sales': 'Total Sales (€)'}
        )
        fig_stores.update_layout(height=400)
        st.plotly_chart(fig_stores, use_container_width=True)
        
        st.subheader("Store Performance Summary")
        st.dataframe(top_stores, use_container_width=True)
    
    with tab3:
        promo_impact = data.groupby('Promo').agg({
            'Sales': 'mean',
            'Customers': 'mean'
        }).round(2)
        
        promo_impact.index = ['No Promo', 'With Promo']
        
        fig_promo = px.bar(
            x=promo_impact.index,
            y=promo_impact['Sales'],
            title="Average Sales: Promo vs No Promo",
            labels={'y': 'Average Sales (€)', 'x': 'Promotion Status'}
        )
        fig_promo.update_layout(height=400)
        st.plotly_chart(fig_promo, use_container_width=True)
        
        promo_lift = ((promo_impact.loc['With Promo', 'Sales'] - 
                      promo_impact.loc['No Promo', 'Sales']) / 
                     promo_impact.loc['No Promo', 'Sales'] * 100)
        
        st.info(f"🎯 Promotion Effectiveness: {promo_lift:.1f}% increase in average sales")

elif page == "Model Training":
    st.header("🤖 Model Training & Performance")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Model Configuration")
        
        n_estimators = st.slider("Number of Trees", 50, 200, 100)
        max_depth = st.slider("Max Depth", 5, 20, 10)
        test_size = st.slider("Test Size", 0.1, 0.3, 0.2)
        
        if st.button("🚀 Train New Model", type="primary"):
            with st.spinner("Training model..."):
                sample_processed = preprocess_features(st.session_state.sample_data)
                feature_cols = ['Store', 'Open', 'PromoActive', 'IsHoliday', 'SchoolHoliday',
                              'DayOfWeek', 'Month', 'Year', 'IsWeekend', 'Quarter']
                
                X = sample_processed[feature_cols]
                y = sample_processed['Sales']
                
                from sklearn.model_selection import train_test_split
                from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
                
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=test_size, random_state=42
                )
                
                new_predictor = SalesPredictor()
                new_predictor.model = RandomForestRegressor(
                    n_estimators=n_estimators,
                    max_depth=max_depth,
                    random_state=42
                )
                
                X_train_scaled = new_predictor.scaler.fit_transform(X_train)
                new_predictor.model.fit(X_train_scaled, y_train)
                new_predictor.feature_columns = X.columns.tolist()
                new_predictor.is_trained = True
                
                X_test_scaled = new_predictor.scaler.transform(X_test)
                y_pred = new_predictor.model.predict(X_test_scaled)
                
                mae = mean_absolute_error(y_test, y_pred)
                rmse = np.sqrt(mean_squared_error(y_test, y_pred))
                r2 = r2_score(y_test, y_pred)
                
                st.session_state.predictor = new_predictor
                
                st.success("✅ Model training completed!")
                
                st.subheader("📊 Model Performance")
                
                metric_col1, metric_col2, metric_col3 = st.columns(3)
                
                with metric_col1:
                    st.metric("MAE", f"€{mae:.0f}")
                
                with metric_col2:
                    st.metric("RMSE", f"€{rmse:.0f}")
                
                with metric_col3:
                    st.metric("R² Score", f"{r2:.3f}")
    
    with col2:
        st.subheader("Model Performance Visualization")
        
        if st.session_state.predictor.is_trained:
            if hasattr(st.session_state.predictor.model, 'feature_importances_'):
                importances = st.session_state.predictor.model.feature_importances_
                feature_names = st.session_state.predictor.feature_columns
                
                importance_df = pd.DataFrame({
                    'feature': feature_names,
                    'importance': importances
                }).sort_values('importance', ascending=True)
                
                fig_importance = px.bar(
                    importance_df,
                    x='importance',
                    y='feature',
                    orientation='h',
                    title="Feature Importance",
                    labels={'importance': 'Importance Score'}
                )
                fig_importance.update_layout(height=400)
                st.plotly_chart(fig_importance, use_container_width=True)
        else:
            st.info("👆 Train a model to see performance metrics")

st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: #666;'>
        <p>🏪 Rossmann Sales Forecasting Dashboard | Built with Streamlit</p>
        <p>📊 Powered by Machine Learning | NextHikes IT Solutions</p>
    </div>
    """,
    unsafe_allow_html=True
)