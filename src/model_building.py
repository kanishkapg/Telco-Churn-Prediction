from typing import Dict, Any
from abc import ABC, abstractmethod
import joblib #provide better security for your artifacts than .pkl files
import os
from datetime import datetime
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

class BaseModelBuilder(ABC):
    """
        The double asterisk ** before kwargs is used to capture keyword arguments 
        into a dictionary.
        
        > *args captures positional arguments as a tuple.
        > **kwargs captures keyword arguments as a dictionary.
    """
    def __init__(self, model_name:str, **kwargs): #**kwargs: keyword_arguments
        self.model_name = model_name
        self.model = None
        self.model_params = kwargs
        
    @abstractmethod
    def build_model(self):
        pass
    
    def save_model(self, filepath):
        if self.model is None:
            raise ValueError("No model to save, Build the model first.")
        joblib.dump(self.model, filepath)

    def load_model(self, filepath):
        if not os.path.exists(filepath):
            raise ValueError("Model file does not exist.")
        self.model = joblib.load(filepath)
        
        
class LogisticRegressionModelBuilder(BaseModelBuilder):
    def __init__(self, **kwargs):
        default_params = {
            'C': 1.0,
            'max_iter': 100
        }
        default_params.update(kwargs)
        super().__init__('LogisticRegression', **default_params)
        
    def build_model(self):
        self.model = LogisticRegression(**self.model_params)
        return self.model
        
        
class RandomForestModelBuilder(BaseModelBuilder):
    def __init__(self, **kwargs):
        default_params = {
            'n_estimators': 100,
            'max_depth': 10,
            'min_samples_split': 2,
            'min_samples_leaf': 1,
            'random_state': 42
        }
        default_params.update(kwargs)
        super().__init__('RandomForest', **default_params)
        
    def build_model(self):
        self.model = RandomForestClassifier(**self.model_params)
        return self.model


class XGBoostModelBuilder(BaseModelBuilder):
    def __init__(self, **kwargs):
        default_params = {
            'n_estimators': 100,
            'max_depth': 10,
            'random_state': 42
        }
        default_params.update(kwargs)
        super().__init__('XGBoost', **default_params)

    def build_model(self):
        self.model = XGBClassifier(**self.model_params)
        return self.model
    
# rf = RandomForestModelBuilder()
# rf_model = rf.build_model()
# print(rf_model)

# xgb = XGBoostModelBuilder()
# xgb_model = xgb.build_model()
# print(xgb_model)

if __name__ == "__main__":
    lr = LogisticRegressionModelBuilder()
    lr_model = lr.build_model()
    print(lr_model)

    rf = RandomForestModelBuilder()
    rf_model = rf.build_model()
    print(rf_model)

    xgb = XGBoostModelBuilder()
    xgb_model = xgb.build_model()
    print(xgb_model)