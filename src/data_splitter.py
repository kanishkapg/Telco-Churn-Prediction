import logging
import pandas as pd
from abc import ABC, abstractmethod
from enum import Enum
from typing import Tuple
from sklearn.model_selection import train_test_split

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class DataSplittingStrategy(ABC):
    @abstractmethod
    def split_data(self, df: pd.DataFrame, target_column: str) -> Tuple[
                                                                            pd.DataFrame,
                                                                            pd.DataFrame, 
                                                                            pd.Series,
                                                                            pd.Series
                                                                        ]:
        pass
    
class SplitType(str, Enum):
    SIMPLE = "simple"
    STRATIFIED = "stratified"
    
class SimpleTrainTestSplitStrategy(DataSplittingStrategy):
    def __init__(self, test_size = 0.2):
        self.test_size = test_size
        
    def split_data(self, df, target_column):
        y = df[target_column]
        X = df.drop(columns=[target_column])

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=self.test_size)

        return X_train, X_test, y_train, y_test
