import logging
import pandas as pd
import os
from typing import Dict
import numpy as np
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))
from data_ingestion import DataIngestorCSV
from handle_missing_values import DropMissingValuesStrategy, FillMissingValuesStrategy, MissingValueHandlingStrategy, FillMethod
from outlier_detection import OutlierDetector, IQROutlierDetection
from type_coercion import DataTransformer, TypeCoercionStrategy
from feature_engineering import CategoricalReplacementStrategy, AggregationFeatureStrategy
from feature_binning import FeatureBinningStrategy, CustomBinningStrategy
from feature_encoding import FeatureEncodingStrategy, NominalEncodingStrategy, OrdinalEncodingStrategy, BinaryEncodingStrategy
from feature_scaling import FeatureScalingStrategy, MinMaxScalingStrategy
from data_splitter import DataSplittingStrategy, SimpleTrainTestSplitStrategy
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'utils'))
from config import (get_data_paths, get_columns, get_missing_values_config, get_outlier_config,
get_binning_config, get_encoding_config, get_engineering_config, get_scaling_config, get_splitting_config)


def data_pipeline(
                    data_path: str='data/raw/Telco-Customer-Churn.csv',
                    target_column: str='Churn',
                    test_size: float=0.2,
                    force_rebuild: bool=False
                ) -> Dict[str, np.ndarray]:
    
    data_paths = get_data_paths()
    columns = get_columns()
    outlier_config = get_outlier_config()
    engineering_config = get_engineering_config()
    binning_config = get_binning_config()
    encoding_config = get_encoding_config()
    scaling_config = get_scaling_config()
    splitting_config = get_splitting_config()
    
    
    print('=======Step 1 : Data Ingestion=======')
    artifacts_dir = os.path.join(os.path.dirname(__file__), '..', data_paths['data_artifacts_dir'])
    X_train_path = os.path.join(artifacts_dir, 'X_train.csv')
    y_train_path = os.path.join(artifacts_dir, 'y_train.csv')
    X_test_path = os.path.join(artifacts_dir, 'X_test.csv')
    y_test_path = os.path.join(artifacts_dir, 'y_test.csv')
    
    os.makedirs(data_paths['data_artifacts_dir'], exist_ok=True)
    
    if os.path.exists(X_train_path) and \
        os.path.exists(y_train_path) and \
        os.path.exists(X_test_path) and \
        os.path.exists(y_test_path):
            
            X_train = pd.read_csv(X_train_path)
            X_test = pd.read_csv(X_test_path)
            y_train = pd.read_csv(y_train_path)
            y_test = pd.read_csv(y_test_path)
            
    ingestor = DataIngestorCSV()
    df = ingestor.ingest(data_path)
    print(f"loaded data shape {df.shape}")
    
    print('\n=======Step 2 : Type Coercion=======')
    type_coercion_strategy = TypeCoercionStrategy(
        column_types={
            'TotalCharges': 'numeric'
        }
    )
    df = type_coercion_strategy.transform(df)
    print(f"Data shape after type coercion :  {df.shape}")


    print('\n=======Step 3 : Handle Missing Values=======')
    totalcharges_handler = FillMissingValuesStrategy(
        fill_method=FillMethod.CONSTANT,
        columns=['TotalCharges'],
        constant_value=0
    )
    
    df = totalcharges_handler.handle(df)
    print(f"Data shape after missing value handling :  {df.shape}")


    print('\n=======Step 4 : Handle Outliers=======')
    outlier_detector = OutlierDetector(strategy=IQROutlierDetection())
    df = outlier_detector.handle_outliers(df, columns['numeric_columns'])
    print(f"Data shape after outlier removal :  {df.shape}")


    print('\n=======Step 5 : Feature Engineering=======')
    category_replacer = CategoricalReplacementStrategy(
        replacements= {
            'MultipleLines': {'No phone service': 'No'},
            **{col: {'No internet service': 'No'} for col in columns['internet_service_columns']}
        }
    )
    feature_aggregator = AggregationFeatureStrategy(
        aggregation_configs= [
            {
                'new_col': 'Add-ons',
                'source_cols': columns['internet_service_columns'],
                'method': 'count_yes',
                'value_to_count': 'Yes'
            }
        ]
    )
    
    df = category_replacer.transform(df)
    df = feature_aggregator.transform(df)
    print(f"Data shape after feature engineering :  {df.shape}")


    print('\n=======Step 6 : Feature Binning=======')
    binning = CustomBinningStrategy(binning_config['tenure_bins'])
    df = binning.bin_feature(df, 'tenure')
    print(f'Data after feature binning : {df.head()}')
    

    print('\n=======Step 7 : Feature Encoding=======')
    
    df['InternetServices'] = df['InternetService'].map(
        {
            'DSL': 'DSL',
            'Fiber optic': 'Fiber optic',
            'No': 'No Service'
        }
    )
    
    nominal_encoder = NominalEncodingStrategy(encoding_config['nominal_columns'])
    ordinal_encoder = OrdinalEncodingStrategy(encoding_config['ordinal_mappings'])
    binary_encoder = BinaryEncodingStrategy(encoding_config['binary_columns'])
    
    df = nominal_encoder.encode(df)
    df = ordinal_encoder.encode(df)
    df = binary_encoder.encode(df)
    
    print(f"Data shape after feature encoding :  {df.shape}")
    print(f'Data after feature encoding : {df.head(10)}')
    
    
    print('\n=======Step 8 : Feature Scaling=======')
    minmax_scaler = MinMaxScalingStrategy()
    df = minmax_scaler.scale(df, scaling_config['columns_to_scale'])
    print(f'Data after feature scaling : {df.head()}')
    

    print('\n=======Step 9 : Post Processing=======')
    df = df.drop(columns=columns['drop_columns'], axis=1)
    print(f"Data shape after post processing :  {df.shape}")
    print(f'Data after post processing : {df.head()}')
    df.to_csv('data//processed/Telco-Customer-Churn-Processed.csv', index=False)


    print('\n=======Step 10 : Data Splitting=======')
    splitter = SimpleTrainTestSplitStrategy(test_size=splitting_config['test_size'])
    X_train, X_test, y_train, y_test = splitter.split_data(df, 'Churn')
    
    X_train.to_csv(X_train_path, index=False)
    y_train.to_csv(y_train_path, index=False)
    X_test.to_csv(X_test_path, index=False)
    y_test.to_csv(y_test_path, index=False)
    
    print(f"X train size : {X_train.shape}")
    print(f"y train size : {y_train.shape}")
    print(f"X test size : {X_test.shape}")
    print(f"y test size : {y_test.shape}")
    

data_pipeline()
