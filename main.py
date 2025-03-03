from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import uvicorn
import pickle
import os
from app.models import CarFeatures, PredictionResponse
from app.prediction import predict_car_price, get_valid_feature_values
from app.utils import logger

# Load model components
model_path = "/opt/render/project/src/static/car_price_prediction_components.pkl"

with open(model_path, 'rb') as f:
    components = pickle.load(f)

# Create FastAPI application
app = FastAPI(
    title="Car Price Prediction API",
    description="An API to predict car prices based on car features",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins; specify domains in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Create a directory for static files
static_folder = os.path.join(os.path.dirname(os.path.dirname(__file__)), "static")
os.makedirs(static_folder, exist_ok=True)

# Mount static file service
app.mount("/static", StaticFiles(directory=static_folder), name="static")

@app.get("/")
async def root():
    """Serve the frontend page"""
    index_path = os.path.join(static_folder, "index.html")
    return FileResponse(index_path)

@app.post("/predict", response_model=PredictionResponse)
async def predict_price(car: CarFeatures):
    """Predict car price based on car features"""
    try:
        result = predict_car_price(car, components)
        return result
    except Exception as e:
        logger.error(f"Error occurred during prediction: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy"}

@app.get("/model-info")
async def model_info():
    """Retrieve model information"""
    model_type = type(components['model']).__name__
    return {
        "model_type": model_type,
        "features": {
            "numerical": components['numerical_cols'],
            "categorical": components['categorical_cols']
        },
        "feature_selection": "enabled" if components.get('selected_indices') is not None else "disabled",
        "version": "1.0.0"
    }

@app.get("/feature-values")
async def feature_values():
    """Retrieve valid values for each feature"""
    try:
        values = get_valid_feature_values(components)
        return values
    except Exception as e:
        logger.error(f"Error occurred while retrieving feature values: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Feature values retrieval error: {str(e)}")

if __name__ == "__main__":
    # Get port number, prioritize environment variable
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("app.main:app", host="0.0.0.0", port=port)
