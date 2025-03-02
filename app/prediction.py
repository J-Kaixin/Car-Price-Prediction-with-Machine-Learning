import pandas as pd
import numpy as np
from app.models import CarFeatures, PredictionResponse
from app.utils import logger

def predict_car_price(car: CarFeatures, components: dict) -> PredictionResponse:
    # Convert input data to DataFrame
    input_data = car.dict()
    
    # Handle inconsistency in specific field names
    if 'Prod_year' in input_data and 'Prod. year' in components['model_features']:
        input_data['Prod. year'] = input_data.pop('Prod_year')
    
    input_df = pd.DataFrame([input_data])
    
    # Convert column names by removing underscores
    input_df.columns = [col.replace('_', ' ') for col in input_df.columns]
    
    # Log input data for debugging
    logger.info(f"Input data: {input_df.to_dict()}")

    # Handle missing values in numerical features
    for col in components['numerical_cols']:
        if col in input_df.columns:
            input_df[col] = pd.to_numeric(input_df[col], errors='coerce')
            if pd.isna(input_df[col]).any():
                input_df[col].fillna(components['numerical_impute_values'][col], inplace=True)

    # Handle missing values and standardization in categorical features
    for col in components['categorical_cols']:
        if col in input_df.columns:
            # Ensure data is of string type
            input_df[col] = input_df[col].astype(str)
            
            # Convert to lowercase and strip whitespace to match training processing
            input_df[col] = input_df[col].str.lower().str.strip()
            
            # Handle missing values
            if pd.isna(input_df[col]).any():
                input_df[col].fillna(components['categorical_impute_values'][col], inplace=True)
            
            logger.info(f"Processed {col}: {input_df[col].tolist()}")

    # Apply label encoding - critical fix
    for col in components['categorical_cols']:
        if col in input_df.columns:
            encoder = components['label_encoders'][col]
            
            # Find the most common category in training data as a fallback
            default_value = encoder.classes_[0] if len(encoder.classes_) > 0 else "unknown"
            
            # Create a new column ensuring values exist in encoder categories
            safe_values = []
            for idx, val in enumerate(input_df[col]):
                if val not in encoder.classes_:
                    logger.warning(f"Feature {col} contains value '{val}' not seen in training data, replacing with '{default_value}'")
                    safe_values.append(default_value)
                else:
                    safe_values.append(val)
            
            # Replace original column with safe values
            input_df[col] = safe_values
            
            try:
                # Apply encoder transformation
                input_df[col] = encoder.transform(input_df[col])
            except ValueError as e:
                logger.error(f"Encoding error in feature {col}: {str(e)}")
                raise ValueError(f"Feature {col} contains unknown labels: {input_df[col].tolist()}")

    # Apply scaling to numerical features
    cols_to_scale = [col for col in components['numerical_cols'] if col in input_df.columns and col != 'Price']
    if cols_to_scale:
        input_df[cols_to_scale] = components['scaler'].transform(input_df[cols_to_scale])

    # Ensure all required features exist
    missing_features = [feat for feat in components['model_features'] if feat not in input_df.columns]
    if missing_features:
        logger.error(f"Missing the following features: {missing_features}")
        raise ValueError(f"Input data is missing the following required features: {missing_features}")

    # Prepare features, ensuring feature order matches training data
    features = input_df[components['model_features']]
    
    # Apply feature selection (if available)
    if components.get('selected_indices') is not None:
        features = features.iloc[:, components['selected_indices']]

    # Predict price
    predicted_price = components['model'].predict(features)[0]
    logger.info(f"Predicted price: {predicted_price}")

    # Compute confidence interval (if model supports it)
    confidence_interval = None
    if hasattr(components['model'], 'estimators_'):
        predictions = [tree.predict(features)[0] for tree in components['model'].estimators_]
        confidence_interval = {
            "lower_bound": float(np.percentile(predictions, 2.5)),
            "upper_bound": float(np.percentile(predictions, 97.5))
        }

    # Compute feature importance (if model supports it)
    feature_importance = None
    if hasattr(components['model'], 'feature_importances_'):
        importances = components['model'].feature_importances_
        feature_names = features.columns
        importance_dict = {name: float(imp) for name, imp in zip(feature_names, importances)}
        feature_importance = dict(sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)[:5])

    return {
        "predicted_price": float(predicted_price),
        "confidence_interval": confidence_interval,
        "feature_importance": feature_importance
    }

# Add a helper function to get valid feature values for the frontend
def get_valid_feature_values(components):
    """Retrieve valid values for each categorical feature"""
    valid_values = {}
    for col in components['categorical_cols']:
        print(col)
        encoder = components['label_encoders'][col]
        valid_values[col] = encoder.classes_.tolist()
    return valid_values
