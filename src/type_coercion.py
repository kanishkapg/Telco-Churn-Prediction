from abc import ABC, abstractmethod
import pandas as pd
import logging
from typing import Optional, Dict

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class DataTransformer(ABC):
    @abstractmethod
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply the transformation to the DataFrame.
        """
        pass

class TypeCoercionStrategy(DataTransformer):
    def __init__(
        self, 
        column_types: Dict[str, str],
        errors: str = 'coerce'  # For pd.to_numeric or similar
    ):
        self.column_types = column_types
        self.errors = errors
        logging.info(f"Initialized TypeCoercionStrategy with column types: {self.column_types}, errors: {self.errors}")

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        df_coerced = df.copy()
        for col, target_type in self.column_types.items():
            if col not in df_coerced.columns:
                logging.warning(f"Column '{col}' specified for coercion but not found in DF")
                continue

            original_na_count = df_coerced[col].isnull().sum()
            
            if target_type == 'numeric':
                df_coerced[col] = pd.to_numeric(df_coerced[col], errors=self.errors)
            elif target_type == 'datetime':
                df_coerced[col] = pd.to_datetime(df_coerced[col], errors=self.errors)
            else:
                logging.warning(f"Unknown target type '{target_type}' for column '{col}'")
                
            new_na_count = df_coerced[col].isnull().sum() - original_na_count
            if new_na_count > 0:
                logging.info(f"Coerced '{col}' to {target_type}; introduced {new_na_count} new NaNs from invalid values")
            if df_coerced[col].isnull().all():
                logging.warning(f"After coercion, '{col}' is entirely NaN; consider dropping or custom handling")
                
        return df_coerced
        
if __name__ == "__main__":
    # Example usage
    sample_df = pd.DataFrame({
        'num_col': ['1', '2', 'three', '4.0', ''],
        'date_col': ['2020-01-01', 'not_a_date', '2020-03-15', None, '2020-05-20'],
        'other_col': ['a', 'b', 'c', 'd', 'e']
    })

    strategy = TypeCoercionStrategy(column_types={
        'num_col': 'numeric'
    })
    coerced_df = strategy.transform(sample_df)
    print(coerced_df)