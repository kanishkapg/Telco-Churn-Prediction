from typing import Dict, List, Optional
import pandas as pd
from type_coercion import DataTransformer  # From your type_coercion.py
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class CategoricalReplacementStrategy(DataTransformer):
    def __init__(
        self,
        replacements: Optional[Dict[str, Dict[str, str]]] = None
    ):
        """
    :param replacements: Dict of column: {old_value: new_value} for categorical replacements.
        """
        self.replacements = replacements or {}
        logging.info(f"Initialized CategoricalReplacementStrategy with replacements: {self.replacements.keys()}")

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        df_replaced = df.copy()

        # Step 1.1: Apply replacements
        for col, replace_map in self.replacements.items():
            if col not in df_replaced.columns:
                logging.warning(f"Column '{col}' specified for replacement but not found in DF")
                continue
            df_replaced[col] = df_replaced[col].replace(replace_map)

        return df_replaced
    
class AggregationFeatureStrategy(DataTransformer):
    def __init__(
        self,
        aggregation_configs: Optional[List[Dict]] = None
    ):
        self.aggregation_configs = aggregation_configs or []
        logging.info(f"Initialized AggregationFeatureStrategy with {len(self.aggregation_configs)} aggregation configs")
        
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        df_aggregated = df.copy()
        for config in self.aggregation_configs:
            new_col = config.get('new_col')
            source_cols = config.get('source_cols', [])
            method = config.get('method', 'count_yes')  # Default to your YAML method
            value_to_count = config.get('value_to_count', 'Yes')  # Configurable

            missing_cols = [c for c in source_cols if c not in df_aggregated.columns]
            if missing_cols or not new_col:
                logging.warning(f"Skipping '{new_col}': Invalid config or missing cols {missing_cols}")
                continue
            if new_col in df_aggregated.columns:
                logging.warning(f"Overwriting '{new_col}'")

            if method == 'count_yes':  # Matches your YAML; extend for sum/mean/etc.
                df_aggregated[new_col] = (df_aggregated[source_cols] == value_to_count).sum(axis=1)
            else:
                raise ValueError(f"Unsupported method: {method}")

            logging.info(f"Created '{new_col}' (method: {method}). Min: {df_aggregated[new_col].min()}, Max: {df_aggregated[new_col].max()}")
        
        return df_aggregated
            
    
if __name__ == "__main__":
    # Lightweight self-test for the feature engineering strategies
    sample_df = pd.DataFrame({
        'MultipleLines': ['No phone service', 'Yes', 'No', 'No phone service', 'Yes'],
        'OnlineSecurity': ['No internet service', 'Yes', 'No', 'No internet service', 'Yes'],
        'OnlineBackup': ['Yes', 'No internet service', 'No', 'Yes', 'No internet service'],
        'DeviceProtection': ['No internet service', 'No', 'Yes', 'No internet service', 'Yes'],
        'TechSupport': ['Yes', 'No internet service', 'No', 'Yes', 'No internet service'],
        'StreamingTV': ['No internet service', 'Yes', 'No', 'No internet service', 'Yes'],
        'StreamingMovies': ['Yes', 'No internet service', 'No', 'Yes', 'No internet service'],
    })

    internet_service_columns = [
        'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 
        'TechSupport', 'StreamingTV', 'StreamingMovies'
    ]
    aggregation_configs = [
        {
            'new_col': 'Add-ons',
            'source_cols': internet_service_columns,
            'method': 'count_yes',
            'value_to_count': 'Yes',
        }
    ]

    # Build replacements dynamically for all source_cols in aggregation_configs
    replacements = {
        'MultipleLines': {'No phone service': 'No'},
        **{col: {'No internet service': 'No'} for col in internet_service_columns}
    }

    # Apply categorical replacements
    cat_strategy = CategoricalReplacementStrategy(replacements=replacements)
    df_after_replace = cat_strategy.transform(sample_df)
    logging.info("After replacements (head):\n%s", df_after_replace.head())

    # Apply aggregation features
    agg_strategy = AggregationFeatureStrategy(aggregation_configs=aggregation_configs)
    df_engineered = agg_strategy.transform(df_after_replace)
    logging.info("After aggregation (head):\n%s", df_engineered.head())