# Airline Price Prediction

A machine learning project that predicts airline ticket prices using various regression models and feature engineering techniques. This project was developed as part of the Predictive Analysis course (CSE3047) at VIT University.

## Table of Contents
- [Overview](#overview)
- [Dataset](#dataset)
- [Installation](#installation)
- [Project Structure](#project-structure)
- [Data Preprocessing](#data-preprocessing)
- [Feature Engineering](#feature-engineering)
- [Models & Results](#models--results)
- [Visualizations](#visualizations)
- [Usage](#usage)
- [Conclusion](#conclusion)
- [Contributors](#contributors)

## Overview

Airline ticket prices can vary dynamically and significantly for the same flight, even for nearby seats within the same cabin. This project addresses the challenge of predicting optimal ticket prices by analyzing various factors that influence pricing. The goal is to help customers determine the best time to buy tickets while understanding price patterns.

### Key Features
- Comprehensive data preprocessing and cleaning
- Feature extraction from datetime columns
- Multiple encoding techniques for categorical variables
- Outlier detection and handling
- Implementation of 6 different machine learning models
- Comparative analysis with visualizations

## Dataset

**Source:** [Kaggle - Flight Price Prediction Dataset](https://www.kaggle.com/code/anshigupta01/flight-price-prediction/data)

**Dataset Statistics:**
- Total Rows: 10,683
- Total Columns: 10

**Attributes:**
- `Airline` - Name of the airline company
- `Date_of_Journey` - Date of the journey
- `Source` - Origin city
- `Destination` - Destination city
- `Dep_Time` - Departure time
- `Arrival_Time` - Arrival time
- `Duration` - Flight duration
- `Total_Stops` - Number of stops between source and destination
- `Additional_Info` - Additional flight information
- `Price` - Ticket price (Target variable)

## Installation

### Prerequisites
```bash
Python 3.7+
```

### Required Libraries
```bash
pip install numpy pandas matplotlib seaborn scikit-learn
```

### Clone the Repository
```bash
git clone https://github.com/yourusername/airline-price-prediction.git
cd airline-price-prediction
```

## Project Structure

```
airline-price-prediction/
│
├── data/
│   └── PA_PRICE_PREDICTION_DATASET.csv
│
├── notebooks/
│   └── airline_price_prediction.ipynb
│
├── src/
│   ├── data_preprocessing.py
│   ├── feature_engineering.py
│   ├── model_training.py
│   └── visualization.py
│
├── models/
│   └── trained_models/
│
├── requirements.txt
├── README.md
└── presentation/
    └── AIRLINE_PRICE_PREDICTION.pptx
```

## Data Preprocessing

### 1. Missing Value Handling
- **Source & Destination:** Filled with mode (10 and 11 nulls respectively)
- **Total_Stops:** Dropped rows (3 nulls)
- **Airline & Dep_Time:** Dropped rows (1 null each)
- **Price:** Filled with mean (24 nulls)

### 2. Data Type Conversion
Converted object datatypes to datetime format for temporal features:
- `Date_of_Journey`
- `Dep_Time`
- `Arrival_Time`

### 3. Outlier Treatment
- Identified outliers in the `Price` feature using box plots
- Replaced values above 40,000 with median price

## Feature Engineering

### Temporal Feature Extraction

**From Date_of_Journey:**
- `journey_day` - Day of the month
- `journey_month` - Month of the year

**From Departure Time:**
- `Dep_Time_hour` - Hour of departure
- `Dep_Time_min` - Minute of departure

**From Arrival Time:**
- `Arrival_Time_hour` - Hour of arrival
- `Arrival_Time_min` - Minute of arrival

**From Duration:**
- `dur_hour` - Hours of flight duration
- `dur_min` - Minutes of flight duration

### Categorical Encoding

**One-Hot Encoding (Nominal Data):**
- `Airline` (12 categories)
- `Source` (5 categories)
- `Destination` (6 categories)

**Label Encoding (Ordinal Data):**
- `Total_Stops`: Mapped as {non-stop: 0, 1 stop: 1, 2 stops: 2, 3 stops: 3, 4 stops: 4}

## Models & Results

The dataset was split into 80% training and 20% testing data. Six regression models were implemented and evaluated:

| Model | R² Score | Performance |
|-------|----------|-------------|
| **Random Forest Regressor** | **84%** | ⭐ **Best Model** |
| Gradient Boosting Regressor | 77% | Good |
| Decision Tree Regressor | 69% | Moderate |
| K-Nearest Neighbors Regressor | 55% | Fair |
| Logistic Regression | 15% | Poor |
| Support Vector Regressor | Worst | Poor |

### Evaluation Metrics
- R² Score (Coefficient of Determination)
- Mean Absolute Error (MAE)
- Mean Squared Error (MSE)
- Root Mean Squared Error (RMSE)

### Feature Importance
Feature selection was performed using mutual information classification to identify the most impactful features for price prediction.

## Visualizations

The project includes comprehensive visualizations:

1. **Price Distribution by Airline** - Box plots showing price variations across different airlines
2. **Price vs Total Stops** - Relationship between number of stops and ticket price
3. **Price vs Source City** - Price variations based on departure city
4. **Price vs Destination City** - Price variations based on arrival city
5. **Arrival Time Impact** - Hexbin plot showing price correlation with arrival hours
6. **Outlier Detection** - Distribution plots and box plots for identifying anomalies

## Usage

### Basic Example

```python
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

# Load the dataset
df = pd.read_csv('data/PA_PRICE_PREDICTION_DATASET.csv')

# Perform preprocessing (see data_preprocessing.py)
# ...

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=123)

# Train the model
model = RandomForestRegressor()
model.fit(X_train, y_train)

# Make predictions
predictions = model.predict(X_test)

# Evaluate
from sklearn.metrics import r2_score
print(f"R² Score: {r2_score(y_test, predictions)}")
```

### Model Prediction Function

```python
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import numpy as np

def predict(ml_model):
    model = ml_model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    
    print(f"Training Score: {model.score(X_train, y_train)}")
    print(f"R² Score: {r2_score(y_test, predictions)}")
    print(f"MAE: {mean_absolute_error(y_test, predictions)}")
    print(f"MSE: {mean_squared_error(y_test, predictions)}")
    print(f"RMSE: {np.sqrt(mean_squared_error(y_test, predictions))}")
    
    return predictions
```

## Conclusion

This project successfully demonstrates the application of machine learning techniques to predict airline ticket prices. Key findings include:

- **Random Forest Regressor** achieved the highest accuracy (84%), making it the optimal model for this dataset
- Temporal features (journey date, departure/arrival times) significantly impact price predictions
- Number of stops and airline carrier are strong predictors of ticket prices
- Proper handling of categorical variables and outliers improved model performance

The project provides valuable insights for both customers seeking optimal ticket prices and airlines optimizing their pricing strategies.

## Future Enhancements

- Integration of real-time data for dynamic price predictions
- Implementation of deep learning models (LSTM, Neural Networks)
- Development of a web application for user-friendly price predictions
- Incorporation of additional features (seasonal trends, holidays, events)
- Hyperparameter tuning using Grid Search or Random Search

## Contributors

**Course:** Predictive Analysis (CSE3047)  
**Faculty:** Prof. N. Nalini  
**Institution:** VIT University

## License

This project is available for educational purposes. Please refer to the dataset source for data usage terms.

## Acknowledgments

- Dataset source: [Kaggle](https://www.kaggle.com/code/anshigupta01/flight-price-prediction/data)
- VIT University for academic support
- Prof. N. Nalini for guidance

---

**Note:** This is an academic project developed for learning purposes in the field of predictive analytics and machine learning.
