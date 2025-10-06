import logging
import pandas as pd
from abc import ABC, abstractmethod

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class OutlierDetectionStrategy(ABC):
    @abstractmethod
    def detect_outliers(self, df: pd.DataFrame) -> pd.DataFrame:
        pass
    
class IQROutlierDetection(OutlierDetectionStrategy):
    def detect_outliers(self, df, columns):
        outliers = pd.DataFrame(False, index=df.index, columns=columns)
        for col in columns:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            outliers[col] = (df[col] < lower_bound) | (df[col] > upper_bound)
        logging.info(f"Detected outliers using IQR method in columns: {columns}")
        return outliers
    
class OutlierDetector:
    def __init__(self, strategy):
        self._strategy = strategy

    def detect_outliers(self, df, selected_columns):
        return self._strategy.detect_outliers(df, selected_columns)
    
    def handle_outliers(self, df, selected_columns, method='drop'):
        outliers = self.detect_outliers(df, selected_columns)
        outlier_count = outliers.sum(axis=1)
        rows_to_remove = outlier_count >= 2
        return df[~rows_to_remove]
        

if __name__ == "__main__":
    # Example usage
    data = {
        'A': [1, 2, 3, 4, 100],
        'B': [5, 6, 7, 8, -50],
        'C': [10, 11, 12, 13, 14]
    }
    df = pd.DataFrame(data)
    logging.info(f"Original DataFrame:\n{df}")

    outlier_detector = OutlierDetector(strategy=IQROutlierDetection())
    cleaned_df = outlier_detector.handle_outliers(df, ['A', 'B'])
    logging.info(f"DataFrame after outlier removal:\n{cleaned_df}")