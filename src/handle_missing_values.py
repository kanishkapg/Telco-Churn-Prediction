import os
from urllib import response
import pandas as pd
import numpy as np
import logging
from enum import Enum
from typing import Optional
from dotenv import load_dotenv
from pydantic import BaseModel
from abc import ABC, abstractmethod

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
load_dotenv()

class MissingValueHandlingStrategy(ABC):
    @abstractmethod
    def handle(self, df: pd.DataFrame) -> pd.DataFrame:
        pass

class DropMissingValuesStrategy(MissingValueHandlingStrategy):
    def __init__(self, critical_columns=[]):
        self.critical_columns = critical_columns
        logging.info(
            f"Critical columns for missing value handling: {self.critical_columns}"
        )
    
    def handle(self, df: pd.DataFrame) -> pd.DataFrame:
        df_cleaned = df.dropna(subset=self.critical_columns)
        n_dropped = len(df) - len(df_cleaned)
        logging.info(f"Dropped {n_dropped} rows with missing values in critical columns")
        return df_cleaned

    
class FillMethod(Enum):
    MEAN = 'mean'
    MEDIAN = 'median'
    MODE = 'mode'
    CONSTANT = 'constant'
    CUSTOM = 'custom'
    
class FillMissingValuesStrategy(MissingValueHandlingStrategy):
    def __init__(
        self,
        fill_method: FillMethod,
        columns: Optional[list] = None, # If None, apply to all columns with missing values
        constant_value: Optional[any] = None, # Required for CONSTANT
        custom_imputer: Optional[callable] = None # Required for CUSTOM: func that takes pd.Series and returns fill value
    ):
        self.fill_method = fill_method
        self.columns = columns
        self.constant_value = constant_value
        self.custom_imputer = custom_imputer
        self._validate_params()
        logging.info(
            f"Initialized FillMissingValuesStrategy with method: {self.fill_method}, columns: {self.columns}"
        )


    def _validate_params(self):
        if self.fill_method == FillMethod.CONSTANT and self.constant_value is None:
            raise ValueError("constant_value must be provided for CONSTANT fill method")
        if self.fill_method == FillMethod.CUSTOM and self.custom_imputer is None:
            raise ValueError("custom_imputer must be provided for CUSTOM fill method")
        
    def handle(self, df: pd.DataFrame) -> pd.DataFrame:
        cols_to_fill = self.columns or df.columns[df.isnull().any()].tolist()
        if not cols_to_fill:
            logging.info("No missing values to fill")
            return df
        
        df_filled = df.copy()
        for col in cols_to_fill:
            if self.fill_method == FillMethod.MEAN:
                if pd.api.types.is_numeric_dtype(df[col]):
                    fill_value = df_filled[col].mean()
                else:
                    logging.warning(f"Column '{col}' is not numeric. Skipping MEAN fill.")
                    continue
                
            elif self.fill_method == FillMethod.MEDIAN:
                if pd.api.types.is_numeric_dtype(df[col]):
                    fill_value = df_filled[col].median()
                else:
                    logging.warning(f"Column '{col}' is not numeric. Skipping MEDIAN fill.")
                    continue
                
            elif self.fill_method == FillMethod.MODE:
                fill_value = df_filled[col].mode()[0]
                
            elif self.fill_method == FillMethod.CONSTANT:
                fill_value = self.constant_value
                
            elif self.fill_method == FillMethod.CUSTOM:
                fill_value = self.custom_imputer(df_filled[col])
            else:
                raise ValueError(f"Unsupported fill method: {self.fill_method}")
            
            n_missing_before = df_filled[col].isnull().sum()
            df_filled[col].fillna(fill_value, inplace=True)
            n_missing_after = df_filled[col].isnull().sum()
            logging.info(
                f"Filled {n_missing_before - n_missing_after} missing values in column '{col}' using method '{self.fill_method}'"
            )
            
        return df_filled

if __name__ == "__main__":
    sample_df = pd.DataFrame({
        'num_col': [1, 2, None, 4],
        'cat_col': ['a', 'b', None, 'b'],
        'other': [5, None, 7, 8]
    })

    # Mean fill on all columns (skips non-num for mean)
    strategy = FillMissingValuesStrategy(fill_method=FillMethod.MEAN)
    filled_df = strategy.handle(sample_df)
    print(filled_df)

    # Mode fill on specific columns
    strategy_mode = FillMissingValuesStrategy(fill_method=FillMethod.MODE, columns=['cat_col'])
    filled_df_mode = strategy_mode.handle(sample_df)
    print(filled_df_mode)
    
    strategy_const = FillMissingValuesStrategy(fill_method=FillMethod.CONSTANT, columns=['other'], constant_value=13)
    filled_df_const = strategy_const.handle(sample_df)
    print(filled_df_const)