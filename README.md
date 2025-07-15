# 🌾 A-Multi-Modal-Deep-Learning-Framework-for-Crop-Price-Prediction
## 📌 Project Title
## A Multi-Modal Deep Learning Framework for Crop Price Prediction on Crop Price Prediction Dataset
# 📌 Introduction
Agriculture forms the backbone of many economies, yet the sector remains highly vulnerable to price volatility driven by seasonal shifts, market dynamics, and unpredictable climate conditions. These fluctuations severely impact farmers' incomes, policymaking, and supply chain planning.

To address this challenge, our project introduces a Hybrid Deep Learning Framework that accurately predicts crop prices using a combination of:

Bidirectional LSTM (BiLSTM): Captures temporal patterns in historical price data.

XGBoost Regressor: Models non-linear residuals missed by deep learning.

SHAP Analysis: Provides interpretability by explaining the model’s decisions feature-wise.

# 🎯 Why This Project?
To empower farmers and supply chain participants with data-driven price forecasts.

To mitigate risks posed by sudden price drops or surges.

To advance research in agri-tech machine learning solutions with a focus on interpretability and accuracy.

# 🔬 Core Features
Robust Preprocessing: Handles missing data, outliers, and categorical encoding.

Feature Engineering: Generates lags, rolling averages, and other temporal statistics.

Hybrid Modeling: Combines sequential and non-sequential learning for superior predictions.

Interpretability with SHAP: Transparent insights into feature contributions.

Performance Metrics: Evaluated using RMSE, MAE, and R² score.

# 📊 Dataset Source
We trained our model on the Crop Price Prediction Dataset from Kaggle, which contains detailed pricing records across different states, commodities, and periods.

# 📂 Dataset Link:
Crop Price Prediction Dataset on Kaggle : https://www.kaggle.com/datasets/santoshd3/crop-price-prediction

### Project Details :

# 📊 Problem Statement
Crop price volatility affects the livelihoods of millions of farmers, especially in countries like India where a majority depend on agriculture. Traditional models lack the capacity to capture temporal trends and complex interactions between multiple factors affecting crop prices.

Our project addresses this gap by building a hybrid deep learning framework that combines Bidirectional LSTM (BiLSTM) for capturing temporal dependencies and XGBoost for modeling residuals, ensuring both predictive accuracy and interpretability using SHAP analysis.

# 🎯 Objectives
Develop a hybrid architecture integrating BiLSTM (deep sequential learning) and XGBoost (nonlinear residual modeling).

Engineer robust features including lags, rolling statistics, and categorical encodings.

Evaluate the model using RMSE, MAE, and R² metrics.

Utilize SHAP (SHapley Additive exPlanations) for feature-level interpretability.

Provide a scalable solution adaptable to multiple crops and regions.

# 📚 Dataset
*Source: Kaggle - Crop Price Prediction Dataset*

*File Used: corn yield.csv*

## Features:

*State, District, Market, Commodity, Variety*

# Date of arrival

*Minimum, Maximum, Modal prices*

# Data underwent extensive preprocessing including:

*Removal of irrelevant columns*

*Outlier detection with z-score*

*Handling missing values*

*Encoding categorical variables via One-Hot Encoding*

*Scaling with MinMaxScaler*

# 🛠️ Technologies Used
Programming Language: *Python*

Deep Learning: *TensorFlow, Keras (for BiLSTM)*

Machine Learning: *XGBoost*

Interpretability: *SHAP*

Visualization: *Matplotlib, Seaborn*

Development Platform: *Google Colab*

# 🔍 Methodology
# 1. Data Preprocessing

Cleaned and standardized data.

Generated lag features (1-day, 7-day, 30-day).

Calculated rolling mean and standard deviation (7-day window).

Applied One-Hot Encoding for categorical data.

Normalized numerical features.

# 2. Model Architecture
graph TD
    A[Raw Crop Data] --> B[Preprocessing & Encoding]
    B --> C[Feature Engineering (Lags, Rolling Stats)]
    C --> D1[BiLSTM Model]
    C --> D2[XGBoost Regressor]
    D1 --> E1[LSTM Predictions]
    D2 --> E2[Residuals Predictions]
    E1 --> F[Hybrid Output = LSTM + XGBoost]
    E2 --> F
    F --> G[Performance Evaluation & SHAP Explainability]
# 3. Hybrid Modeling
BiLSTM captures sequential temporal trends.

XGBoost models residuals left by BiLSTM predictions.

Combined output enhances accuracy.

# 4. Model Evaluation
RMSE (Root Mean Squared Error): 0.0335

MAE (Mean Absolute Error): 0.0129

R² Score: 0.9272

# 5. Interpretability
Implemented SHAP to visualize feature importance.

SHAP beeswarm plots helped interpret the impact of features on model predictions.

# 📈 Results
Achieved a high R² Score of ~0.927, reflecting strong predictive power.

The hybrid model outperformed standalone BiLSTM and XGBoost models.

SHAP analysis enabled transparency, critical for stakeholders like farmers and policymakers.

# 🔮 Future Scope
Integrate external data sources such as:

Rainfall and climate data

Market demand/supply information

Deploy as an interactive Web Dashboard or API.

Extend support for multiple crops and regions.

# 🖥️ How to Run

*Clone the repository:*

git clone <repository-url>
cd <repository-folder>

*Install Dependencies:*

pip install numpy pandas tensorflow xgboost shap matplotlib seaborn scikit-learn category_encoders
Execute the Jupyter Notebook or run on Google Colab.

*Upload the dataset (corn yield.csv) when prompted.*

This repository aims to provide a scalable, transparent, and practical solution for predicting crop prices, contributing to more stable agricultural markets and better policy formulation.
