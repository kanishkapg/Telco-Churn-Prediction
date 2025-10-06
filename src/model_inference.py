import os
import sys
import json
import joblib
import logging
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Tuple, Optional, Union
from sklearn.base import BaseEstimator
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'utils'))
from feature_binning import FeatureBinningStrategy, CustomBinningStrategy
from feature_encoding import FeatureEncodingStrategy, NominalEncodingStrategy, OrdinalEncodingStrategy, BinaryEncodingStrategy
from feature_scaling import FeatureScalingStrategy, MinMaxScalingStrategy
from config import (get_binning_config, get_encoding_config, get_columns, get_scaling_config)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class ModelInference:
    # Here we use the trained model previously
    def __init__(self, model_path):
        self.model_path = model_path
        self.load_model()
        self.encoders = {}
        self.binning_config = get_binning_config()
        self.encoding_config = get_encoding_config()
        self.columns_config = get_columns()
        self.scaling_config = get_scaling_config()
        self.scaler = None
        
    def load_model(self):
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model file not found at {self.model_path}")
        self.model = joblib.load(self.model_path)
        logger.info(f"Model loaded from {self.model_path}")
        
    def load_encoders(self, encoders_dir):
        for file in os.listdir(encoders_dir):
            if file.endswith('_encoder.joblib'):
                feature_name = file.replace('_encoder.joblib', '')
                self.encoders[feature_name] = joblib.load(os.path.join(encoders_dir, file))
        logger.info(f"Encoders loaded from {encoders_dir}")

    # def load_scaler(self, scaler_path):
    #     if os.path.exists(scaler_path):
    #         self.scaler = joblib.load(scaler_path)
    #         logger.info(f"Scaler loaded from {scaler_path}")
    #     else:
    #         logger.warning(f"Scaler file not found at {scaler_path}. Scaling will not be applied.")

        
    def preprocess_input(self, data):
        df = pd.DataFrame([data])

        # Type Coercion
        df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')

        # Handle Missing Values
        df['TotalCharges'] = df['TotalCharges'].fillna(0)

        # Feature Engineering
        df['MultipleLines'] = df['MultipleLines'].replace({'No phone service': 'No'})
        internet_service_columns = self.columns_config['internet_service_columns']
        for col in internet_service_columns:
            df[col] = df[col].replace({'No internet service': 'No'})
        
        df['Add-ons'] = df[internet_service_columns].apply(lambda row: (row == 'Yes').sum(), axis=1)

        # Feature Binning
        binning = CustomBinningStrategy(self.binning_config['tenure_bins'])
        df = binning.bin_feature(df, 'tenure')

        # Feature Encoding
        df['InternetServices'] = df['InternetService'].map({
            'DSL': 'DSL',
            'Fiber optic': 'Fiber optic',
            'No': 'No Service'
        })

        # Apply nominal encoding using pre-trained encoders
        for column in self.encoding_config['nominal_columns']:
            if column in df.columns and column in self.encoders:
                encoder = self.encoders[column]
                encoded = encoder.transform(df[[column]])
                
                categories = encoder.categories_[0]
                encoded_cols = [f"{column}_{str(cat).replace(' ', '_')}" for cat in categories]
                encoded_df = pd.DataFrame(encoded, columns=encoded_cols, index=df.index)
                
                df = df.drop(column, axis=1)
                df = pd.concat([df, encoded_df], axis=1)
        
        # Apply ordinal and binary encoding
        ordinal_encoder = OrdinalEncodingStrategy(self.encoding_config['ordinal_mappings'])
        binary_encoder = BinaryEncodingStrategy(self.encoding_config['binary_columns'])
        
        df = ordinal_encoder.encode(df)
        df = binary_encoder.encode(df)

        # Feature Scaling
        if self.scaler:
            minmax_scaler = MinMaxScalingStrategy()
            df = minmax_scaler.scale(df, self.scaling_config['columns_to_scale'], scaler=self.scaler)

        # Post Processing
        df = df.drop(columns=self.columns_config['drop_columns'], axis=1)
        
        # Align columns with model's expected input
        model_features = self.model.feature_names_in_
        df = df.reindex(columns=model_features, fill_value=0)

        return df
    
    def predict(self, data):
        pp_data = self.preprocess_input(data)
        y_pred = self.model.predict(pp_data)
        y_proba = float(self.model.predict_proba(pp_data)[:, 1])
        y_pred = 'Churn' if y_pred == 1 else 'Retain'
        y_proba = round(y_proba * 100, 2)
        
        return {
            "Status": y_pred,
            "Confidence": f"{y_proba} %"
        }

