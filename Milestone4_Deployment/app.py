"""
Streamlit Web Application for On-Time Delivery Prediction
Milestone 4: Deployment and Documentation
"""

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import json
from pathlib import Path
from datetime import datetime
import plotly.graph_objects as go
import plotly.express as px


# ============================================================================
# PAGE CONFIGURATION
# ============================================================================
st.set_page_config(
    page_title="On-Time Delivery Predictor",
    page_icon="📦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .metric-box {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .prediction-box-on-time {
        background-color: #d4edda;
        padding: 20px;
        border-radius: 10px;
        border-left: 4px solid #28a745;
    }
    .prediction-box-delayed {
        background-color: #f8d7da;
        padding: 20px;
        border-radius: 10px;
        border-left: 4px solid #dc3545;
    }
    .info-box {
        background-color: #d1ecf1;
        padding: 15px;
        border-radius: 8px;
        border-left: 4px solid #17a2b8;
    }
    </style>
    """, unsafe_allow_html=True)


# ============================================================================
# MODEL AND DATA LOADING
# ============================================================================
@st.cache_resource
def load_models():
    """Load trained models and preprocessing artifacts"""
    model_dir = Path(__file__).parent / "trained_models"
    preprocessing_dir = Path(__file__).parent.parent / "Milestone2_Preprocessing" / "outputs"
    
    models = {}
    model_errors = {}
    
    for model_file in ['logistic_regression_model.pkl', 'random_forest_model.pkl', 'xgboost_model.pkl']:
        model_path = model_dir / model_file
        model_name = model_file.replace('_model.pkl', '')
        if model_path.exists():
            try:
                with open(model_path, 'rb') as f:
                    models[model_name] = pickle.load(f)
            except Exception as e:
                model_errors[model_name] = str(e)
                # Continue loading other models even if one fails
                continue
    
    if model_errors:
        error_msg = "Failed to load: " + ", ".join(model_errors.keys())
        st.warning(f"⚠️ {error_msg}")
    
    # Load scaler
    scaler = None
    scaler_path = preprocessing_dir / 'scaler.pkl'
    if scaler_path.exists():
        try:
            with open(scaler_path, 'rb') as f:
                scaler = pickle.load(f)
        except Exception as e:
            st.warning(f"⚠️ Error loading scaler: {e}")
    
    # Load label encoders
    label_encoders = {}
    encoders_path = preprocessing_dir / 'label_encoders.pkl'
    if encoders_path.exists():
        with open(encoders_path, 'rb') as f:
            label_encoders = pickle.load(f)
    
    return models, scaler, label_encoders


@st.cache_data
def get_feature_info():
    """Get feature names and their ranges"""
    features_path = Path(__file__).parent.parent / "Milestone2_Preprocessing" / "outputs"
    X_train_path = features_path / 'X_train.csv'
    
    if X_train_path.exists():
        X_train = pd.read_csv(X_train_path)
        return X_train.columns.tolist(), X_train
    return [], None


# ============================================================================
# SIDEBAR NAVIGATION
# ============================================================================
st.sidebar.title("🚀 Navigation")
page = st.sidebar.radio(
    "Select Page",
    options=["📊 Prediction", "📈 Model Performance", "ℹ️ About", "🔧 Data Info"],
    help="Navigate through different sections of the app"
)


# ============================================================================
# PAGE 1: PREDICTION
# ============================================================================
if page == "📊 Prediction":
    st.title("🎯 On-Time Delivery Prediction")
    st.markdown("---")
    
    models, scaler, label_encoders = load_models()
    features, X_train_data = get_feature_info()
    
    if not models:
        st.error("❌ No trained models found. Please ensure models are trained and saved.")
        st.stop()
    
    if scaler is None:
        st.error("❌ Scaler not found. Please ensure preprocessing is complete.")
        st.stop()
    
    # Create two columns for input
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📝 Order Information")
        supplier_rating = st.slider(
            "Supplier Rating (1-5)",
            min_value=1.0,
            max_value=5.0,
            value=3.5,
            step=0.1,
            help="Average rating of the supplier (1=Poor, 5=Excellent)"
        )
        
        supplier_lead_time = st.number_input(
            "Supplier Lead Time (days)",
            min_value=1,
            max_value=30,
            value=10,
            help="Days needed by supplier to fulfill order"
        )
        
        shipping_distance_km = st.number_input(
            "Shipping Distance (km)",
            min_value=0,
            max_value=10000,
            value=500,
            help="Distance to be covered for delivery"
        )
        
        order_quantity = st.number_input(
            "Order Quantity",
            min_value=1,
            max_value=1000,
            value=20,
            help="Number of units ordered"
        )
    
    with col2:
        st.subheader("💰 Order Value & History")
        unit_price = st.number_input(
            "Unit Price ($)",
            min_value=0.0,
            max_value=1000.0,
            value=100.0,
            step=0.01,
            help="Price per unit"
        )
        
        total_order_value = st.number_input(
            "Total Order Value ($)",
            min_value=0.0,
            max_value=100000.0,
            value=2000.0,
            step=1.0,
            help="Total value of the order"
        )
        
        previous_on_time_rate = st.slider(
            "Previous On-Time Rate",
            min_value=0.0,
            max_value=1.0,
            value=0.75,
            step=0.01,
            help="Historical on-time delivery rate"
        )
        
        shipment_mode = st.selectbox(
            "Shipment Mode",
            options=["Air", "Road", "Sea"],
            help="Method of transportation"
        )
    
    st.markdown("---")
    
    # Additional categorical features
    col3, col4, col5 = st.columns(3)
    
    with col3:
        weather_condition = st.selectbox(
            "Weather Condition",
            options=["Clear", "Rainy", "Storm"],
            help="Weather during shipment"
        )
    
    with col4:
        region = st.selectbox(
            "Region",
            options=["North", "South", "East", "West"],
            help="Delivery region"
        )
    
    with col5:
        carrier_name = st.selectbox(
            "Carrier Name",
            options=["BlueDart", "DHL", "FedEx", "LocalTruckers", "UPS"],
            help="Shipping carrier"
        )
    
    # Make prediction button
    if st.button("🔮 Predict Delivery Status", use_container_width=True, type="primary"):
        try:
            # Prepare input data
            input_data = {
                'supplier_rating': supplier_rating,
                'supplier_lead_time': supplier_lead_time,
                'shipping_distance_km': shipping_distance_km,
                'order_quantity': order_quantity,
                'unit_price': unit_price,
                'total_order_value': total_order_value,
                'previous_on_time_rate': previous_on_time_rate,
                'shipment_mode': shipment_mode,
                'weather_condition': weather_condition,
                'region': region,
                'carrier_name': carrier_name
            }
            
            # Encode categorical features
            input_df = pd.DataFrame([input_data])
            
            # Apply label encoding for categorical columns
            categorical_cols = {
                'shipment_mode_encoded': 'shipment_mode',
                'weather_condition_encoded': 'weather_condition',
                'region_encoded': 'region',
                'carrier_name_encoded': 'carrier_name'
            }
            
            for encoded_col, original_col in categorical_cols.items():
                input_df[encoded_col] = label_encoders[original_col].transform([input_data[original_col]])[0]
            
            # Add missing categorical features with default values
            input_df['holiday_period_encoded'] = 0
            input_df['delayed_reason_code_encoded'] = 0
            
            # Models were trained on the first 12 columns:
            # supplier_rating, supplier_lead_time, shipping_distance_km, order_quantity,
            # unit_price, total_order_value, previous_on_time_rate, shipment_mode_encoded,
            # weather_condition_encoded, region_encoded, holiday_period_encoded, carrier_name_encoded
            
            X_input = pd.DataFrame({
                'supplier_rating': [supplier_rating],
                'supplier_lead_time': [supplier_lead_time],
                'shipping_distance_km': [shipping_distance_km],
                'order_quantity': [order_quantity],
                'unit_price': [unit_price],
                'total_order_value': [total_order_value],
                'previous_on_time_rate': [previous_on_time_rate],
                'shipment_mode_encoded': [input_df['shipment_mode_encoded'].values[0]],
                'weather_condition_encoded': [input_df['weather_condition_encoded'].values[0]],
                'region_encoded': [input_df['region_encoded'].values[0]],
                'holiday_period_encoded': [input_df['holiday_period_encoded'].values[0]],
                'carrier_name_encoded': [input_df['carrier_name_encoded'].values[0]],
            })
            
            # Use input directly without feature names (model was fitted without them)
            X_input_scaled = X_input.values
            
            # Display prediction results
            st.markdown("---")
            st.subheader("🎯 Prediction Results")
            
            if not models:
                st.error("❌ No models loaded. Cannot make predictions.")
            else:
                # Create columns for model results
                predictions = {}
                probabilities = {}
                
                for model_name, model in models.items():
                    try:
                        pred = model.predict(X_input_scaled)[0]
                        prob = model.predict_proba(X_input_scaled)[0]
                        
                        predictions[model_name] = pred
                        probabilities[model_name] = {
                            'on_time': prob[1] if pred == 1 else prob[0],
                            'delayed': prob[0] if pred == 1 else prob[1]
                        }
                    except Exception as e:
                        st.warning(f"⚠️ {model_name.replace('_', ' ').title()} prediction failed: {str(e)}")
                
                if not predictions:
                    st.error("❌ All models failed to make predictions.")
                else:
                    # Display ensemble prediction
                    ensemble_pred = round(sum(predictions.values()) / len(predictions)) if predictions else 0
                    ensemble_prob = np.mean([probabilities[m]['on_time'] for m in predictions]) if predictions else 0.5
                    
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        if ensemble_pred == 1:
                            st.markdown(f"""
                            <div class="prediction-box-on-time">
                                <h3>✅ Likely ON-TIME</h3>
                                <p style="font-size: 24px; font-weight: bold; color: #28a745;">
                                    {ensemble_prob*100:.1f}%
                                </p>
                                <p>Ensemble Confidence</p>
                            </div>
                            """, unsafe_allow_html=True)
                        else:
                            st.markdown(f"""
                            <div class="prediction-box-delayed">
                                <h3>⚠️ Risk of DELAY</h3>
                                <p style="font-size: 24px; font-weight: bold; color: #dc3545;">
                                    {(1-ensemble_prob)*100:.1f}%
                                </p>
                                <p>Ensemble Confidence</p>
                            </div>
                            """, unsafe_allow_html=True)
                    
                    # Individual model results
                    with col2:
                        st.markdown("**Individual Model Predictions:**")
                        for model_name, pred in predictions.items():
                            status = "✅ On-Time" if pred == 1 else "⚠️ Delayed"
                            prob = probabilities[model_name]['on_time']
                            st.write(f"{model_name.replace('_', ' ').title()}: {status} ({prob*100:.1f}%)")
                    
                    with col3:
                        st.markdown("**Order Summary:**")
                        st.info(f"""
                        - Distance: {shipping_distance_km} km
                        - Lead Time: {supplier_lead_time} days
                        - Order Value: ${total_order_value:,.2f}
                        - Supplier Rating: {supplier_rating}/5
                        - Historical On-Time: {previous_on_time_rate*100:.1f}%
                        """)
                    
                    # Probability distribution chart
                    st.markdown("---")
                    st.subheader("📊 Prediction Confidence by Model")
                    
                    chart_data = {
                        'Model': [m.replace('_', ' ').title() for m in predictions.keys()],
                        'On-Time Probability': [probabilities[m]['on_time'] for m in predictions.keys()]
                    }
                    
                    fig = go.Figure(data=[
                        go.Bar(
                            x=chart_data['Model'],
                            y=chart_data['On-Time Probability'],
                            marker_color=['#28a745' if p > 0.5 else '#dc3545' for p in chart_data['On-Time Probability']],
                            text=[f"{p*100:.1f}%" for p in chart_data['On-Time Probability']],
                            textposition='outside'
                        )
                    ])
                    fig.update_layout(
                        title="On-Time Delivery Probability",
                        yaxis_title="Probability",
                        xaxis_title="Model",
                        height=400,
                showlegend=False
            )
            st.plotly_chart(fig, use_container_width=True)
        
        except Exception as e:
            st.error(f"❌ Error making prediction: {str(e)}")
            st.write("Please check your input values and try again.")


# ============================================================================
# PAGE 2: MODEL PERFORMANCE
# ============================================================================
elif page == "📈 Model Performance":
    st.title("📊 Model Performance Metrics")
    st.markdown("---")
    
    # Load comparison data
    performance_file = Path(__file__).parent.parent / "Milestone3_ModelBuilding" / "outputs" / "model_comparison.csv"
    
    if performance_file.exists():
        perf_df = pd.read_csv(performance_file)
        
        # Display metrics table
        st.subheader("Model Comparison Table")
        st.dataframe(perf_df, use_container_width=True)
        
        # Visualizations
        col1, col2 = st.columns(2)
        
        with col1:
            # Accuracy comparison
            fig_acc = px.bar(
                perf_df,
                x='Model' if 'Model' in perf_df.columns else perf_df.columns[0],
                y='Accuracy' if 'Accuracy' in perf_df.columns else perf_df.columns[1],
                title="Model Accuracy Comparison",
                color_discrete_sequence=['#1f77b4']
            )
            st.plotly_chart(fig_acc, use_container_width=True)
        
        with col2:
            # F1-Score comparison
            if 'F1-Score' in perf_df.columns:
                fig_f1 = px.bar(
                    perf_df,
                    x='Model' if 'Model' in perf_df.columns else perf_df.columns[0],
                    y='F1-Score',
                    title="Model F1-Score Comparison",
                    color_discrete_sequence=['#ff7f0e']
                )
                st.plotly_chart(fig_f1, use_container_width=True)
    else:
        st.info("ℹ️ Model comparison data not available yet. Please run model training first.")
    
    # Training details
    st.markdown("---")
    st.subheader("🏋️ Training Details")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Models Trained", len(models) if 'models' in locals() else 3)
    
    with col2:
        st.metric("Test Set Size", "20% (stratified)")
    
    with col3:
        st.metric("Cross-Validation Folds", 5)
    
    st.info("""
    **Training Approach:**
    - GridSearchCV for hyperparameter tuning
    - 5-fold cross-validation
    - Stratified train-test split (80-20)
    - Weighted metrics for imbalanced classification
    """)


# ============================================================================
# PAGE 3: ABOUT
# ============================================================================
elif page == "ℹ️ About":
    st.title("📋 Project Information")
    st.markdown("---")
    
    st.markdown("""
    ## On-Time Delivery Prediction System
    
    ### Project Overview
    This machine learning application predicts the probability of on-time delivery for supply chain orders.
    It uses historical order data and supplier information to identify potential delivery delays before they occur.
    
    ### Business Problem
    Supply chain delays can result in:
    - Customer dissatisfaction
    - Increased costs
    - Reputation damage
    - Operational inefficiencies
    
    ### Solution
    By predicting delivery delays in advance, stakeholders can:
    - Take preventive measures
    - Optimize logistics routes
    - Improve supplier management
    - Enhance customer communication
    
    ### Key Features
    - **Supplier Metrics**: Rating, lead time, reliability score
    - **Order Details**: Quantity, value, distance, mode
    - **External Factors**: Weather, region, carrier, holiday periods
    - **Historical Data**: Previous on-time delivery rates
    
    ### Model Architecture
    The system uses an ensemble of three machine learning models:
    1. **Logistic Regression**: Fast, interpretable baseline model
    2. **Random Forest**: Captures non-linear relationships
    3. **XGBoost**: High-performance gradient boosting model
    
    Predictions are made using ensemble voting to improve robustness.
    
    ### Metrics
    - **Accuracy**: Correct predictions / Total predictions
    - **Precision**: True positives / (True positives + False positives)
    - **Recall**: True positives / (True positives + False negatives)
    - **F1-Score**: Harmonic mean of precision and recall
    - **ROC-AUC**: Area under receiver operating characteristic curve
    
    ### Dataset Information
    - **Total Records**: ~1000 supply chain orders
    - **Features**: 22 (after preprocessing)
    - **Target**: On-time delivery (Binary: 0=Delayed, 1=On-Time)
    - **Data Span**: Real-world supply chain data
    
    ### Technology Stack
    - **Frontend**: Streamlit
    - **ML Framework**: Scikit-learn, XGBoost
    - **Data Processing**: Pandas, NumPy
    - **Visualization**: Plotly
    - **Deployment**: Streamlit Cloud / Docker
    """)


# ============================================================================
# PAGE 4: DATA INFO
# ============================================================================
elif page == "🔧 Data Info":
    st.title("📊 Feature Information")
    st.markdown("---")
    
    features, X_train_data = get_feature_info()
    
    if X_train_data is not None:
        st.subheader("Feature Statistics")
        st.write(X_train_data.describe())
        
        st.markdown("---")
        st.subheader("Feature Correlation Heatmap")
        
        correlation_file = Path(__file__).parent.parent / "Milestone2_Preprocessing" / "outputs" / "correlation_matrix.csv"
        if correlation_file.exists():
            corr_data = pd.read_csv(correlation_file, index_col=0)
            
            fig = go.Figure(data=go.Heatmap(
                z=corr_data.values,
                x=corr_data.columns,
                y=corr_data.index,
                colorscale='RdBu',
                zmid=0
            ))
            fig.update_layout(height=600)
            st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("---")
        st.subheader("Feature Distributions")
        
        # Sample a few key features for distribution
        key_features = [
            'supplier_rating', 'supplier_lead_time', 'shipping_distance_km',
            'order_quantity', 'total_order_value', 'previous_on_time_rate'
        ]
        
        selected_features = [f for f in key_features if f in X_train_data.columns]
        
        cols = st.columns(2)
        for idx, feature in enumerate(selected_features):
            with cols[idx % 2]:
                fig = go.Figure(data=[
                    go.Histogram(x=X_train_data[feature], nbinsx=30, name=feature)
                ])
                fig.update_layout(title=f"Distribution of {feature}", height=300)
                st.plotly_chart(fig, use_container_width=True)
    else:
        st.error("Data not found. Please ensure preprocessing is complete.")


# ============================================================================
# FOOTER
# ============================================================================
st.markdown("---")
st.markdown("""
    <div style="text-align: center; color: #888; font-size: 12px;">
        <p>🚚 On-Time Delivery Prediction System | Built with Streamlit | Powered by Machine Learning</p>
        <p>© 2024 Supply Chain Analytics | Version 1.0</p>
    </div>
    """, unsafe_allow_html=True)
