import os
import sys
import joblib
import logging
import pandas as pd
from typing import Dict, Any, Tuple, Optional

from data_pipeline import data_pipeline
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))
from model_training import ModelTrainer
from model_evaluation import ModelEvaluator
from model_building import RandomForestModelBuilder, XGBoostModelBuilder
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'utils'))
from config import get_model_config, get_data_paths


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def training_pipeline(
    data_path: str = 'data/raw/Telco-Customer-Churn.csv', # In case we don't have artifacts we will trigger our data pipeline
    model_params: Optional[Dict[str, Any]] = None,
    test_size: float = 0.2,
    model_path: str = 'artifacts/models/telco_churn_analysis.joblib'
):
    data_paths = get_data_paths()

    if (not os.path.exists(data_paths['X_train'])) or \
        (not os.path.exists(data_paths['y_train'])) or \
        (not os.path.exists(data_paths['X_test'])) or \
        (not os.path.exists(data_paths['y_test'])):

            data_pipeline()
    else:
        print("Loading Data Artifacts from Data Pipeline...")

    X_train = pd.read_csv(get_data_paths()['X_train'])
    y_train = pd.read_csv(get_data_paths()['y_train']).values.ravel() # Convert to 1D array
    X_test = pd.read_csv(get_data_paths()['X_test'])
    y_test = pd.read_csv(get_data_paths()['y_test']).values.ravel() # Convert to 1D array
    
    os.makedirs(data_paths['model_artifacts_dir'], exist_ok=True)

    model_builder = XGBoostModelBuilder(**model_params)
    model = model_builder.build_model()
    
    trainer = ModelTrainer()
    model, _ = trainer.train(
        model=model,
        X_train=X_train,
        y_train=y_train
    )
    trainer.save_model(model, model_path)
    
    evaluator = ModelEvaluator('XGBoost', model)
    evaluation_results = evaluator.evaluate(X_test, y_test)
    print(evaluation_results)
    

if __name__ == "__main__":
    model_config = get_model_config()
    model_params = model_config.get('model_params')
    training_pipeline(model_params=model_params)
