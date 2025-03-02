# Car Price Prediction System

This is a machine learning-based car price prediction system that uses the Random Forest algorithm to predict used car prices based on vehicle characteristics. The project includes a complete workflow for data analysis, model training, and web application deployment.

## Project Overview

This project predicts car prices by analyzing various features such as manufacturer, model, mileage, engine volume, etc. The entire system consists of:

1. Data analysis and preprocessing
2. Model training and optimization
3. FastAPI backend API service
4. Interactive web frontend interface

## Tech Stack

- **Data Analysis**: Pandas, NumPy, Matplotlib, Seaborn
- **Machine Learning**: Scikit-learn, Random Forest Regression
- **Backend**: FastAPI, Uvicorn
- **Frontend**: HTML, CSS, JavaScript, Bootstrap, Select2
- **Development Tools**: Jupyter Notebook, Python

## Project Structure


```
Car Price Prediction System/
├── app/                    # FastAPI Application
│   ├── __init__.py        
│   ├── main.py             # FastAPI Main Program
│   ├── models.py           # Data Model Definitions
│   ├── prediction.py       # Prediction Logic
│   └── utils.py            # Utility Functions
├── static/                 # Static Files
│   └── index.html          # Frontend Interface
├── logs/                   # Log Files
├── real-car-predict.ipynb  # Jupyter Notebook for Model Training and Analysis
├── car_price_prediction_components.pkl  # Saved Model and Processing Components
└── README.md               # Project Documentation
```


## Feature Description

The model considers the following features:

### Categorical Features
- Manufacturer
- Model
- Category
- Leather interior
- Fuel type
- Gear box type
- Drive wheels
- Doors
- Wheel
- Color

### Numerical Features
- Levy
- Prod. year
- Engine volume
- Cylinders
- Airbags
- Mileage

## Model Development Process

### 1. Data Analysis
- Exploratory Data Analysis (EDA)
- Feature Distribution Visualization
- Correlation Analysis
- Handling Missing Values

### 2. Model Selection
By comparing multiple regression algorithms (Linear Regression, Random Forest, Gradient Boosting, etc.), Random Forest was chosen due to:
- Its ability to capture nonlinear relationships between features
- Robustness against outliers and noise
- Feature importance analysis support
- Best performance on the validation set

### 3. Hyperparameter Tuning
A staged grid search strategy was used to optimize the model:
1. Baseline model evaluation
2. Tuning tree complexity parameters (`max_depth`, `min_samples_split`, etc.)
3. Tuning forest size and randomness (`n_estimators`, `max_features`, etc.)
4. Integrating best parameters and adding regularization
5. Feature importance analysis and selection

### 4. Model Evaluation
- **Evaluation Metrics**: R² and MSE
- **Cross-validation**: Ensuring model stability
- **Overfitting Analysis**: Comparing training and validation performance to prevent overfitting

## API Endpoints

### Main Endpoints
- `GET /` - Provides the frontend interface
- `POST /predict` - Predicts car price based on vehicle features
- `GET /feature-values` - Retrieves valid values for categorical features
- `GET /health` - Health check endpoint
- `GET /model-info` - Retrieves model information

### Example Request (Prediction)
```json
{
  "Manufacturer": "TOYOTA",
  "Model": "Camry",
  "Category": "Sedan",
  "Leather_interior": "Yes",
  "Fuel_type": "Petrol",
  "Gear_box_type": "Automatic",
  "Drive_wheels": "Front",
  "Doors": "04-May",
  "Wheel": "Left wheel",
  "Color": "Black",
  "Levy": 1200,
  "Prod_year": 2018,
  "Engine_volume": 2.5,
  "Cylinders": 4,
  "Airbags": 8,
  "Mileage": 25000
}
```

### Example Response
```json
{
  "predicted_price": 15800.0,
  "confidence_interval": {
    "lower_bound": 14200.0,
    "upper_bound": 17400.0
  },
  "feature_importance": {
    "Prod_year": 0.3421,
    "Engine_volume": 0.1843,
    "Mileage": 0.1276,
    "Manufacturer": 0.0982,
    "Model": 0.0731
  }
}
```
## Frontend Interface

The frontend provides a user-friendly form for entering car features and obtaining price predictions:

- All categorical features use dropdown selection.
- Search functionality is automatically enabled when options exceed 10 choices.
- A visually appealing result display, including predicted price, confidence interval, and feature importance.
- Responsive design, supporting various devices.

## Deployment Guide

### Environment Requirements
- Python 3.8+
- Dependencies: fastapi, uvicorn, scikit-learn, pandas, numpy

### Installation Steps
1. Clone or download the repository.
2. Install dependencies: `pip install -r requirements.txt`
3. Run the server: `python -m app.main`
4. Access in the browser: `http://localhost:8000`

## Model Advantages

Reasons why Random Forest performed well in this project:

1. **Capturing Nonlinear Relationships**: Car price relationships with features (e.g., mileage, engine volume) are often nonlinear.
2. **Strong Robustness**: As an ensemble method, Random Forest is resistant to noise and outliers.
3. **Feature Importance**: Naturally supports feature importance analysis, helping to understand which variables contribute most to price prediction.
4. **Adaptability to High-Dimensional Data**: Effectively handles multiple numerical and categorical features.

## Future Improvements

1. Implement more advanced feature engineering techniques.
2. Experiment with more advanced models (e.g., XGBoost, LightGBM).
3. Add model interpretability components (e.g., SHAP value analysis).
4. Optimize the frontend user experience.
5. Implement periodic model retraining mechanisms.

## Contributions

Contributions and improvement suggestions are welcome!

## License

MIT