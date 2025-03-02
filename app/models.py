from pydantic import BaseModel, Field
from typing import Dict, Optional

class CarFeatures(BaseModel):
    """Car features input model"""
    Manufacturer: str = Field(..., description="Manufacturer")
    Model: str = Field(..., description="Model")
    Category: str = Field(..., description="Category")
    Leather_interior: str = Field(..., description="Leather interior (Yes/No)")
    Fuel_type: str = Field(..., description="Fuel type")
    Gear_box_type: str = Field(..., description="Gearbox type")
    Drive_wheels: str = Field(..., description="Drive wheels")
    Doors: str = Field(..., description="Number of doors")  # Changed to string type since original data contains formats like "04-May"
    Wheel: str = Field(..., description="Steering wheel position (Left/Right)")
    Color: str = Field(..., description="Color")
    Levy: Optional[float] = Field(None, description="Tax levy")
    Prod_year: Optional[int] = Field(None, description="Production year")  # Converted to 'Prod. year' in prediction logic
    Engine_volume: Optional[float] = Field(None, description="Engine volume")
    Cylinders: Optional[int] = Field(None, description="Number of cylinders")
    Airbags: Optional[int] = Field(None, description="Number of airbags")
    Mileage: Optional[float] = Field(None, description="Mileage")
    
    class Config:
        schema_extra = {
            "example": {
                "Manufacturer": "TOYOTA",
                "Model": "Camry",
                "Category": "Sedan",
                "Leather_interior": "Yes",
                "Fuel_type": "Petrol",
                "Gear_box_type": "Automatic",
                "Drive_wheels": "Front",
                "Doors": "04-May",  # Use format consistent with training data
                "Wheel": "Left wheel",  # Use format consistent with training data
                "Color": "Black",
                "Levy": 1200,
                "Prod_year": 2018,
                "Engine_volume": 2.5,
                "Cylinders": 4,
                "Airbags": 8,
                "Mileage": 25000
            }
        }

class PredictionResponse(BaseModel):
    """Prediction response model"""
    predicted_price: float = Field(..., description="Predicted price")
    confidence_interval: Optional[Dict[str, float]] = Field(None, description="Prediction confidence interval")
    feature_importance: Optional[Dict[str, float]] = Field(None, description="Feature importance")
