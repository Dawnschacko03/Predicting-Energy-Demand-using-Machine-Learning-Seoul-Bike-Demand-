# Predicting-Energy-Demand-using-Machine-Learning-Seoul-Bike-Demand-
This project focuses on predicting hourly energy/demand (bike rental count) using regression and machine learning models.  
The objective is to analyze temporal and environmental factors such as:

Hour of the day

Temperature

Humidity

Rainfall & Snowfall

Season

Holiday

Functioning day

and build predictive models to optimize resource planning and operational efficiency.
This project compares three regression models to evaluate performance differences between linear and non-linear approaches.

📊 Dataset

Dataset Used: Seoul Bike Sharing Demand Dataset
The dataset contains:

8,760 hourly records

Weather information

Seasonal data

Holiday indicators

Bike rental counts (Target variable)

##⚙️ Technologies Used

Python

Pandas

NumPy

Scikit-learn

XGBoost

Matplotlib

Seaborn

Models Implemented

##1️⃣ Linear Regression (Baseline Model)

Used as a benchmark model. Assumes linear relationship between features and target
Performance:

R² ≈ 0.53

High bias due to non-linearity in data

##2️⃣ Random Forest Regressor

Ensemble learning model. Handles non-linearity and feature interactions

Performance:

R² ≈ 0.92

Significant improvement over linear regression

##3️⃣ XGBoost Regressor (Best Performing Model)
Gradient boosting framework
Iteratively reduces residual errors
Performance:
R² ≈ 0.93 – 0.95
Lowest RMSE among all models

##🔍 Key Insights
Bike demand shows strong non-linear behavior.
Weather conditions significantly impact rentals.
Temperature and hour of the day are major predictors.
Ensemble models significantly outperform linear models.

##📊 Visualizations Included

Actual vs Predicted Scatter Plot
Time-Series Comparison Plot
Feature Importance Graph
