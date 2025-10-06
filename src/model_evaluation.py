import os
import sys
import joblib
import logging
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")
from typing import Dict, Any, Tuple, Optional, Union
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'utils'))

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class ModelEvaluator:
    def __init__(self, model_name: str, model):
        self.model_name = model_name
        self.model = model
        self.evaluation_results = {}
        
    def evaluate(self, X_test, y_test):
        y_pred = self.model.predict(X_test)
        
        self.evaluation_results['accuracy'] = accuracy_score(y_test, y_pred)
        self.evaluation_results['precision'] = precision_score(y_test, y_pred)
        self.evaluation_results['recall'] = recall_score(y_test, y_pred)
        self.evaluation_results['f1_score'] = f1_score(y_test, y_pred)
        
        return self.evaluation_results