import os
import sys
import joblib
import logging
import pandas as pd
import numpy as np
import json
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))
from model_inference import ModelInference
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'utils'))
from config import get_model_config, get_inference_config

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


inference = ModelInference(model_path='artifacts/models/telco_churn_analysis.joblib')

def streaming_inference(inference_instance, data):
    inference_instance.load_encoders('artifacts/encode')
    # inference_instance.load_scaler('artifacts/models/scaler.joblib')
    pred = inference_instance.predict(data)
    
    return pred
    
if __name__ == "__main__":
    data = {
        "customerID": "7469-LKBCI",
        "gender": "Male",
        "SeniorCitizen": 0,
        "Partner": "No",
        "Dependents": "No",
        "tenure": 16,
        "PhoneService": "Yes",
        "MultipleLines": "No",
        "InternetService": "No",
        "OnlineSecurity": "No internet service",
        "OnlineBackup": "No internet service",
        "DeviceProtection": "No internet service",
        "TechSupport": "No internet service",
        "StreamingTV": "No internet service",
        "StreamingMovies": "No internet service",
        "Contract": "Two year",
        "PaperlessBilling": "No",
        "PaymentMethod": "Credit card (automatic)",
        "MonthlyCharges": 18.95,
        "TotalCharges": 326.8,
    }
    pred = streaming_inference(inference, data)
    print(pred)