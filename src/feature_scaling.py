import logging
import pandas as pd
from abc import ABC, abstractmethod
from enum import Enum
from typing import List
from sklearn.preprocessing import MinMaxScaler

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class FeatureScalingStrategy(ABC):
    @abstractmethod
    def scale(self, df: pd.DataFrame, columns_to_scale: List[str]) -> pd.DataFrame:
        pass
    
class ScalingType(str, Enum):
    MINMAX = 'minmax'
    STANDARD = 'standard'
    
class MinMaxScalingStrategy(FeatureScalingStrategy):
    def __init__(self):
        self.scaler = MinMaxScaler()
        self.fitted = False
        
    def scale(self, df, columns_to_scale):
        df[columns_to_scale] = self.scaler.fit_transform(df[columns_to_scale])
        self.fitted = True
        logging.info(f'Scaled columns: {columns_to_scale} using MinMaxScaler')
        return df
    
    def get_scaler(self):
        return self.scaler


if __name__ == "__main__":
    # Example usage
    data = {
        'feature1': [10, 20, 30, 40, 50],
        'feature2': [100, 200, 300, 400, 500],
        'feature3': [5, 15, 25, 35, 45]
    }
    df = pd.DataFrame(data)
    logging.info("Original DataFrame:")
    logging.info(df)
    
    columns_to_scale = ['feature1', 'feature2']
    scaler = MinMaxScalingStrategy()
    scaled_df = scaler.scale(df, columns_to_scale)
    
    logging.info("Scaled DataFrame:")
    logging.info(scaled_df)