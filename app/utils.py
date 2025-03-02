import logging
import os

def setup_logger():
    """Set up the logger"""
    logger = logging.getLogger("car_price_api")
    
    # If handlers already exist, do not add again
    if logger.handlers:
        return logger
        
    logger.setLevel(logging.INFO)
    
    # Create console handler
    console_handler = logging.StreamHandler()
    console_formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)
    
    # Optional: Add a file handler to log messages to a file
    log_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'logs')
    os.makedirs(log_dir, exist_ok=True)
    file_handler = logging.FileHandler(os.path.join(log_dir, 'car_price_api.log'))
    file_formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)
    
    return logger

# Create a global logger instance
logger = setup_logger()
