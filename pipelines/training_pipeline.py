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
from model_building import (
    LogisticRegressionModelBuilder, 
    RandomForestModelBuilder, 
    XGBoostModelBuilder,
    CatBoostModelBuilder
)
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'utils'))
from config import get_model_config, get_data_paths


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def training_pipeline(
    model_type: str = 'xgboost',
    data_path: str = 'data/raw/Telco-Customer-Churn.csv', # In case we don't have artifacts we will trigger our data pipeline
    model_params: Optional[Dict[str, Any]] = None,
    test_size: float = 0.2,
    model_path: Optional[str] = None
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

    # Model selection based on model_type
    model_builders = {
        'logistic_regression': LogisticRegressionModelBuilder,
        'random_forest': RandomForestModelBuilder,
        'xgboost': XGBoostModelBuilder,
        'catboost': CatBoostModelBuilder
    }
    
    if model_type.lower() not in model_builders:
        raise ValueError(f"Unsupported model type: {model_type}. Supported types: {list(model_builders.keys())}")
    
    # Set default model path if not provided
    if model_path is None:
        model_path = f'artifacts/models/{model_type}_telco_churn_analysis.joblib'
    
    # Get model parameters from config or use provided ones
    if model_params is None:
        model_config = get_model_config()
        model_params = model_config.get('model_types', {}).get(model_type, {})
    
    # Convert list parameters to single values (take first value from lists)
    processed_params = {}
    for key, value in model_params.items():
        if isinstance(value, list) and len(value) > 0:
            processed_params[key] = value[0]  # Take first value from list
        else:
            processed_params[key] = value
    
    logger.info(f"Training {model_type} model with parameters: {processed_params}")
    
    # Build and train the model
    model_builder_class = model_builders[model_type.lower()]
    model_builder = model_builder_class(**processed_params)
    model = model_builder.build_model()
    
    trainer = ModelTrainer()
    model, train_score = trainer.train(
        model=model,
        X_train=X_train,
        y_train=y_train
    )
    
    logger.info(f"Training score: {train_score}")
    trainer.save_model(model, model_path)
    logger.info(f"Model saved to: {model_path}")
    
    # Evaluate the model
    evaluator = ModelEvaluator(model_type, model)
    evaluation_results = evaluator.evaluate(X_test, y_test)
    logger.info(f"Evaluation results: {evaluation_results}")
    
    return model, evaluation_results
    

def train_multiple_models(model_types: list = None):
    """Train multiple models and compare their performance"""
    if model_types is None:
        model_types = ['logistic_regression', 'random_forest', 'xgboost', 'catboost']
    
    results = {}
    
    for model_type in model_types:
        logger.info(f"Starting training for {model_type}")
        try:
            model, evaluation_results = training_pipeline(model_type=model_type)
            results[model_type] = evaluation_results
            logger.info(f"Completed training for {model_type}")
        except Exception as e:
            logger.error(f"Failed to train {model_type}: {str(e)}")
            results[model_type] = None
    
    # Print comparison results
    print("\n" + "="*50)
    print("MODEL COMPARISON RESULTS")
    print("="*50)
    
    for model_type, metrics in results.items():
        if metrics is not None:
            print(f"\n{model_type.upper()}:")
            for metric, value in metrics.items():
                print(f"  {metric}: {value:.4f}")
        else:
            print(f"\n{model_type.upper()}: FAILED")
    
    return results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Train machine learning models for telco churn prediction')
    parser.add_argument('--model_type', type=str, default='xgboost', 
                       choices=['logistic_regression', 'random_forest', 'xgboost', 'catboost', 'all'],
                       help='Type of model to train')
    parser.add_argument('--compare_all', action='store_true', 
                       help='Train and compare all available models')
    
    args = parser.parse_args()
    
    if args.compare_all or args.model_type == 'all':
        # Train all models and compare
        train_multiple_models()
    else:
        # Train single model
        model_config = get_model_config()
        model_params = model_config.get('model_types', {}).get(args.model_type, {})
        training_pipeline(model_type=args.model_type, model_params=model_params)
