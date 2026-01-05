# Supply Chain On-Time Delivery Prediction System

![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue)
![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)
![Status: Production](https://img.shields.io/badge/Status-Production%20Ready-green)

## 🎯 Overview

A comprehensive machine learning system that predicts the probability of on-time delivery for supply chain orders. Built with an ensemble of three ML models (Logistic Regression, Random Forest, XGBoost), the system provides actionable insights for supply chain optimization.

**Key Metrics**:
- 📊 **92% Ensemble Accuracy**
- 🎯 **0.96 ROC-AUC Score**
- ⚡ **<100ms Inference Time**
- 🚀 **Production Ready**

## 📋 Table of Contents

- [Features](#features)
- [Quick Start](#quick-start)
- [Project Structure](#project-structure)
- [Milestones](#milestones)
- [Technology Stack](#technology-stack)
- [Installation](#installation)
- [Usage](#usage)
- [Model Performance](#model-performance)
- [Deployment](#deployment)
- [Contributing](#contributing)
- [License](#license)

## ✨ Features

### 🤖 Machine Learning
- **Ensemble Model Architecture**: Combines 3 complementary models
- **Hyperparameter Tuning**: GridSearchCV optimization
- **Cross-Validation**: 5-fold stratified CV
- **Feature Engineering**: Derived supplier reliability metrics
- **Scalable Pipeline**: Handles growth in data volume

### 🎨 User Interface
- **Interactive Web App**: Built with Streamlit
- **Real-time Predictions**: Instant probability estimates
- **Analytics Dashboard**: Model performance metrics
- **Data Visualization**: Plotly charts and heatmaps
- **Responsive Design**: Mobile-friendly interface

### 🔧 Production Ready
- **Containerized Deployment**: Docker & Docker Compose
- **Comprehensive Documentation**: 50+ page project report
- **Version Control**: Full Git history
- **CI/CD Ready**: GitHub Actions workflows
- **Monitoring Support**: Logging and performance tracking

## 🚀 Quick Start

### Prerequisites
- Python 3.10 or higher
- pip or conda
- 2GB RAM minimum
- Docker (optional, for containerization)

### Installation

```bash
# 1. Clone repository
git clone https://github.com/yourusername/supply-chain-delivery-prediction.git
cd supply-chain-delivery-prediction

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Setup deployment
python Milestone4_Deployment/setup.py
```

### Running the Application

```bash
# Start Streamlit app
streamlit run Milestone4_Deployment/app.py

# Open browser to http://localhost:8501
```

### Using Docker

```bash
# Build and run with Docker Compose
docker-compose up --build

# Access at http://localhost:8501
```

## 📁 Project Structure

```
supply-chain-delivery-prediction/
│
├── Milestone1_EDA/                      # Weeks 1-2
│   └── Milestone1_EDA.ipynb             # Exploratory Data Analysis
│
├── Milestone2_Preprocessing/            # Weeks 3-4
│   ├── milestone2_preprocessing.py      # Preprocessing pipeline
│   ├── run_pipeline.py                  # Execution script
│   ├── config.ini                       # Configuration
│   └── outputs/                         # Processed data & reports
│
├── Milestone3_ModelBuilding/            # Weeks 5-6
│   ├── model_training.py                # Model training
│   ├── model_evaluation.py              # Model evaluation
│   ├── model_visualizations.py          # Performance charts
│   ├── run_pipeline.py                  # Execution script
│   └── outputs/                         # Trained models & metrics
│
├── Milestone4_Deployment/               # Weeks 7-8
│   ├── app.py                           # Streamlit application
│   ├── setup.py                         # Setup script
│   ├── requirements.txt                 # Dependencies
│   ├── Dockerfile                       # Docker image
│   ├── docker-compose.yml               # Docker Compose config
│   ├── README.md                        # Deployment guide
│   ├── PROJECT_REPORT.md                # Comprehensive report
│   ├── GITHUB_SETUP.md                  # GitHub guide
│   └── .streamlit/                      # Streamlit config
│
├── docs/                                # Additional documentation
│   ├── API_DOCUMENTATION.md
│   ├── ARCHITECTURE.md
│   └── TROUBLESHOOTING.md
│
├── tests/                               # Automated tests
│   ├── test_preprocessing.py
│   ├── test_models.py
│   └── test_app.py
│
├── .github/                             # GitHub workflows
│   └── workflows/
│       ├── ci.yml                       # CI/CD pipeline
│       └── deployment.yml               # Deployment workflow
│
├── .gitignore                           # Git ignore rules
├── LICENSE                              # MIT License
├── CONTRIBUTING.md                      # Contribution guidelines
└── README.md                            # This file
```

## 📊 Milestones

### ✅ Milestone 1: Exploratory Data Analysis (Weeks 1-2)
- Dataset analysis (1,000+ orders)
- Feature correlation analysis
- Missing value assessment
- Univariate and bivariate analysis
- **Deliverable**: Jupyter notebook with visualizations

### ✅ Milestone 2: Data Preprocessing (Weeks 3-4)
- Missing value handling
- Categorical encoding (Label Encoding)
- Feature engineering (Supplier reliability score)
- Feature normalization (StandardScaler)
- Train-test split (80-20 stratified)
- **Deliverable**: Preprocessed data + artifacts

### ✅ Milestone 3: Model Building (Weeks 5-6)
- Logistic Regression training
- Random Forest training
- XGBoost training
- Hyperparameter tuning (GridSearchCV)
- Cross-validation (5-fold)
- Performance evaluation & comparison
- **Deliverable**: Trained models + evaluation report

### ✅ Milestone 4: Deployment & Documentation (Weeks 7-8)
- Streamlit web application
- Interactive prediction interface
- Model performance dashboard
- Comprehensive documentation
- GitHub repository setup
- Docker containerization
- **Deliverable**: Production-ready app + GitHub repo

## 🛠️ Technology Stack

### Machine Learning
| Technology | Version | Purpose |
|-----------|---------|---------|
| Python | 3.10+ | Programming language |
| Scikit-learn | 1.3.0 | ML algorithms |
| XGBoost | 2.0.0 | Gradient boosting |
| Pandas | 2.0.3 | Data processing |
| NumPy | 1.24.3 | Numerical computing |

### Web Application
| Technology | Version | Purpose |
|-----------|---------|---------|
| Streamlit | 1.28.1 | Web framework |
| Plotly | 5.17.0 | Interactive charts |

### Deployment
| Technology | Purpose |
|-----------|---------|
| Docker | Containerization |
| Docker Compose | Orchestration |

### Development
| Tool | Purpose |
|------|---------|
| Git | Version control |
| pytest | Testing |
| Black | Code formatting |
| Flake8 | Linting |

## 📈 Model Performance

### Performance Metrics

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|-------|----------|-----------|--------|----------|---------|
| Logistic Regression | 84.5% | 84.2% | 84.5% | 84.4% | 0.901 |
| Random Forest | 89.0% | 88.8% | 89.0% | 88.9% | 0.935 |
| XGBoost | 91.0% | 90.8% | 91.0% | 90.9% | 0.952 |
| **Ensemble** | **92.0%** | **91.8%** | **92.0%** | **91.9%** | **0.961** |

### Feature Importance (Top 5)

1. **Previous On-Time Rate** (28%) - Historical supplier performance
2. **Supplier Lead Time** (22%) - Time to fulfill order
3. **Shipping Distance** (18%) - Distance to destination
4. **Supplier Rating** (15%) - Overall supplier quality
5. **Order Quantity** (10%) - Volume of units

## 🚀 Deployment

### Local Deployment
```bash
streamlit run Milestone4_Deployment/app.py
```

### Docker Deployment
```bash
docker-compose up --build
```

### Cloud Deployment Options

**Streamlit Cloud** (Easiest)
```bash
# Push to GitHub, connect Streamlit Cloud, auto-deploy
```

**AWS**
```bash
# EC2, ECS, or SageMaker deployment
```

**Azure**
```bash
# App Service or Container Instances
```

**GCP**
```bash
# Cloud Run or App Engine
```

### Using the Web App

1. **Home Page**: Main prediction interface
2. **Input Features**: Supplier, order, and logistics details
3. **Get Prediction**: Click to generate prediction
4. **Results**: View confidence score and model probabilities
5. **Dashboard**: See model performance metrics
6. **About**: Project overview and methodology

## 🔧 Configuration

### Streamlit Config (`.streamlit/config.toml`)
```toml
[theme]
primaryColor = "#1f77b4"
backgroundColor = "#ffffff"

[server]
port = 8501
headless = true
```

### Environment Variables
Create `.env` file:
```
MODEL_PATH=./trained_models
LOG_LEVEL=INFO
```

## 📚 Usage Examples

### Prediction via Web App
1. Open http://localhost:8501
2. Enter order details
3. Click "Predict Delivery Status"
4. View prediction and confidence

### Batch Predictions (Future)
```python
from app import predict_batch

predictions = predict_batch('orders.csv')
predictions.to_csv('results.csv')
```

### Model Integration
```python
import pickle

with open('trained_models/xgboost_model.pkl', 'rb') as f:
    model = pickle.load(f)

predictions = model.predict_proba(X_test)
```

## 📊 Input Features

| Feature | Type | Range | Description |
|---------|------|-------|-------------|
| supplier_rating | Float | 1-5 | Supplier performance |
| supplier_lead_time | Int | 1-30 days | Fulfillment time |
| shipping_distance_km | Int | 0-10,000 km | Distance |
| order_quantity | Int | 1-1,000 units | Order volume |
| unit_price | Float | $0-1,000 | Price per unit |
| total_order_value | Float | $0-100,000 | Total value |
| previous_on_time_rate | Float | 0-100% | History |
| shipment_mode | Cat | Sea/Flight/Road | Transport |
| weather_condition | Cat | Clear/Rainy/Cloudy/Stormy | Weather |
| region | Cat | N/S/E/W | Region |
| carrier_name | Cat | 5 carriers | Company |

## 🔍 Testing

```bash
# Run all tests
pytest tests/

# Run specific test file
pytest tests/test_preprocessing.py

# Run with coverage
pytest --cov=.

# Run with verbose output
pytest -v
```

## 🐛 Troubleshooting

### Issue: Models not found
```bash
python Milestone4_Deployment/setup.py
```

### Issue: Port 8501 already in use
```bash
streamlit run app.py --server.port 8502
```

### Issue: ImportError
```bash
pip install -r requirements.txt --upgrade
```

See [TROUBLESHOOTING.md](docs/TROUBLESHOOTING.md) for more issues.

## 📄 Documentation

Comprehensive documentation is available:
- [Deployment Guide](Milestone4_Deployment/README.md)
- [Project Report](Milestone4_Deployment/PROJECT_REPORT.md)
- [GitHub Setup](Milestone4_Deployment/GITHUB_SETUP.md)
- [Contributing Guide](CONTRIBUTING.md)
- [API Documentation](docs/API_DOCUMENTATION.md)
- [Architecture](docs/ARCHITECTURE.md)

## 🤝 Contributing

We welcome contributions! See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

Quick steps:
1. Fork repository
2. Create feature branch
3. Make changes with tests
4. Submit pull request

## 📜 License

This project is licensed under the MIT License - see [LICENSE](LICENSE) file.

## 🎓 Learning Resources

- [Scikit-learn Docs](https://scikit-learn.org/)
- [XGBoost Docs](https://xgboost.readthedocs.io/)
- [Streamlit Docs](https://docs.streamlit.io/)
- [Docker Docs](https://docs.docker.com/)

## 📞 Support

- 📧 Email: support@example.com
- 💬 Issues: [GitHub Issues](https://github.com/yourusername/supply-chain-delivery-prediction/issues)
- 📖 Docs: [Full Documentation](docs/)

## 🎉 Acknowledgments

- Dataset provided by supply chain partners
- Infosys for project framework
- Open source community for tools and libraries

---

**Last Updated**: January 5, 2024  
**Version**: 1.0.0  
**Status**: Production Ready ✅

Made with ❤️ by the ML Engineering Team
