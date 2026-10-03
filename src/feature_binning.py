import logging
import pandas as pd
from abc import ABC, abstractmethod
logging.basicConfig(level=logging.INFO, format=
    '%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class FeatureBinningStrategy(ABC):
    @abstractmethod
    def bin_feature(self, df: pd.DataFrame, column: str) ->pd.DataFrame:
        pass

class CustomBinningStratergy(FeatureBinningStrategy):
    def __init__(self, bin_definitions):
        self.bin_definitions = bin_definitions
        logger.info(f"CustomBinningStrategy initialized with bins: {list(bin_definitions.keys())}") 

    def bin_feature(self, df, column):
        logger.info(f"\n{'='*60}")
        logger.info(f"FEATURE BINNING - {column.upper()}")
        logger.info(f"{'='*60}")
        logger.info(f"Starting binning for column: {column}")
        initial_unique = df[column].nunique()
        value_range = (df[column].min(), df[column].max())
        logger.info(f"  Unique values: {initial_unique}, Range: [{value_range[0]:.2f}, {value_range[1]:.2f}]")
        
        def assign_bin(value):
            if value == 850:
                return "Excellent"
            
            for bin_label, bin_range in self.bin_definitions.items():
                if len(bin_range) == 2:
                    if bin_range[0] <= value <= bin_range[1]:
                        return bin_label
                elif len(bin_range) == 1:
                    if value >= bin_range[0]:
                        return bin_label 
                
            if value > 850:
                return "Invalid"
            
            return "Invalid"
        
        df[f'{column}Bins'] = df[column].apply(assign_bin)
        
        # Log binning results
        bin_counts = df[f'{column}Bins'].value_counts()
        logger.info(f"\nBinning Results:")
        for bin_name, count in bin_counts.items():
            logger.info(f"  ✓ {bin_name}: {count} ({count/len(df)*100:.2f}%)")
        
        invalid_count = (df[f'{column}Bins'] == "Invalid").sum()
        if invalid_count > 0:
            logger.warning(f"  ⚠ Found {invalid_count} invalid values in column '{column}'")
            
        del df[column]
        logger.info(f"✓ Original column '{column}' removed, replaced with '{column}Bins'")
        logger.info(f"{'='*60}\n")

        return df