import logging
import joblib
import os
import json
import pandas as pd
from enum import Enum
from typing import Dict, List
from abc import ABC, abstractmethod
from sklearn.preprocessing import OneHotEncoder

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class FeatureEncodingStrategy(ABC):
    @abstractmethod
    def encode(self, df: pd.DataFrame) -> pd.DataFrame:
        pass
    
class VariableType(str, Enum):
    NOMINAL = 'nominal'
    ORDINAL = 'ordinal'
    
class NominalEncodingStrategy(FeatureEncodingStrategy):
    def __init__(self, nominal_columns):
        self.nominal_columns = nominal_columns
        self.encoders = {}
        os.makedirs('artifacts/encode', exist_ok=True)
        
    def encode(self, df):
        for column in self.nominal_columns:
            if column not in df.columns:
                logging.warning(f'Column {column} not found in DataFrame. Skipping encoding for this column.')
                continue
            
            encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore', dtype=int)
            encoded = encoder.fit_transform(df[[column]])
            
            categories = encoder.categories_[0]
            encoded_cols = [f"{column}_{str(cat).replace(' ', '_')}" for cat in categories]
            encoded_df = pd.DataFrame(encoded, columns=encoded_cols, index=df.index)
            
            df = df.drop(column, axis=1)
            df = pd.concat([df, encoded_df], axis=1)
            
            self.encoders[column] = encoder
            encoder_path = os.path.join('artifacts/encode', f'{column}_encoder.joblib')
            joblib.dump(encoder, encoder_path)
            
            # Optionally save categories as JSON for easy inspection
            categories_dict = {str(cat): i for i, cat in enumerate(categories)}
            json_path = os.path.join('artifacts/encode', f'{column}_categories.json')
            with open(json_path, 'w') as f:
                json.dump(categories_dict, f)
                
        return df

    def get_encoders(self):
        return self.encoders


class OrdinalEncodingStrategy(FeatureEncodingStrategy):
    def __init__(self, ordinal_mappings):
        self.ordinal_mappings = ordinal_mappings
        
    def encode(self, df):
        for column, mapping in self.ordinal_mappings.items():
            df[column] = df[column].map(mapping)
            logging.info(f'Encoded ordinal variable {column} with {len(mapping)} categories')
        return df
    

if __name__ == "__main__":
    # Example usage
    df = pd.DataFrame({
        'color': ['red', 'blue', 'green', 'blue'],
        'size': ['S', 'M', 'L', 'XL'],
        'tenure': [5, 15, 30, 45]
    })
    
    nominal_columns = ['color', 'size']
    ordinal_mappings = {
        'tenure': {
            'New': 0,
            'Established': 1,
            'Loyal': 2
        }
    }
    
    nominal_encoder = NominalEncodingStrategy(nominal_columns)
    df = nominal_encoder.encode(df)
    
    ordinal_encoder = OrdinalEncodingStrategy(ordinal_mappings)
    df['tenure'] = pd.cut(df['tenure'], bins=[-1, 24, 48, float('inf')], labels=['New', 'Established', 'Loyal'])
    df = ordinal_encoder.encode(df)
    
    print(df)