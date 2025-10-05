import logging
import pandas as pd
from abc import ABC, abstractmethod

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class FeatureBinningStrategy(ABC):
    @abstractmethod
    def bin_feature(self, df: pd.DataFrame, column: str) -> pd.DataFrame:
        pass


class CustomBinningStrategy(FeatureBinningStrategy):
    def __init__(self, bin_definitions):
        self.bin_definitions = bin_definitions

    def bin_feature(self, df, column):
        def assign_bin(value):
            # Handle scalar values (single numbers)
            if not hasattr(value, 'dtype'):  # It's a scalar
                
                for bin_label, bin_range in self.bin_definitions.items():
                    if len(bin_range) == 2:
                        if (bin_range[0] <= value <= bin_range[1]):
                            return bin_label
                    elif len(bin_range) == 1:
                        if value >= bin_range[0]:
                            return bin_label
                
                return 'Invalid'
            
            else:  # It's a pandas Series
                # Create result series with 'Invalid' as default
                result = pd.Series('Invalid', index=value.index)
                
                
                # Handle other bin definitions
                for bin_label, bin_range in self.bin_definitions.items():
                    if len(bin_range) == 2:
                        mask = (value >= bin_range[0]) & (value <= bin_range[1])
                        result[mask] = bin_label
                    elif len(bin_range) == 1:
                        mask = (value >= bin_range[0])
                        result[mask] = bin_label
                
                return result
    
        df[f"{column}Bins"] = df[column].apply(assign_bin)
        del df[column]
               
        return df
    
if __name__ == "__main__":
    # Example usage
    data = {
        'tenure': [1, 12, 24, 36, 48, 60, 72, -5, 80]
    }
    df = pd.DataFrame(data)
    
    bin_definitions = {
        'New': [0, 24],
        'Established': [25, 48],
        'Loyal': [49]
    }
    
    binning_strategy = CustomBinningStrategy(bin_definitions)
    df_binned = binning_strategy.bin_feature(df, 'tenure')
    
    logging.info("Binned DataFrame:")
    logging.info(df_binned)