import logging
import pandas as pd
from enum import Enum
from abc import ABC, abstractmethod
from typing import Tuple
from sklearn.model_selection import train_test_split
logging.basicConfig(level=logging.INFO, format=
    '%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class DataSplittingStrategy(ABC):
    @abstractmethod
    def split_data(self, df: pd.DataFrame, target_column: str) ->Tuple[pd.
        DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        pass


class SplitType(str, Enum):
    SIMPLE = 'simple' # Simple Splitter I utilized
    STRATIFIED = 'stratified'

class SimpleTrainTestSplitStratergy(DataSplittingStrategy):
    def __init__(self, test_size = 0.2):
        self.test_size= test_size
        logger.info(f"SimpleTrainTestSplitStrategy initialized with test_size={test_size}") 

    def split_data(self, df, target_column):
        logger.info(f"\n{'='*60}")
        logger.info(f"DATA SPLITTING")
        logger.info(f"{'='*60}")
        logger.info(f"Starting data splitting with target column: '{target_column}'")
        logger.info(f"Total samples: {len(df)}, Features: {len(df.columns) - 1}")
        
        # Check for missing values
        missing_count = df.isnull().sum().sum()
        if missing_count > 0:
            logger.warning(f"⚠ Dataset contains {missing_count} missing values before splitting")
        
        Y = df[target_column]
        X = df.drop(columns=[target_column])
        
        # Log target distribution
        target_dist = Y.value_counts()
        logger.info(f"\nTarget Variable Distribution:")
        for value, count in target_dist.items():
            logger.info(f"  {value}: {count} ({count/len(Y)*100:.2f}%)")
        
        # Log feature info
        logger.info(f"\nFeature Information:")
        logger.info(f"  Number of features: {X.shape[1]}")
        logger.info(f"  Feature columns: {list(X.columns)}")
        
        # Perform split
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=self.test_size, random_state=42)
        
        # Log split results
        logger.info(f"\nSplit Results:")
        logger.info(f"  ✓ Training set: {len(X_train)} samples ({len(X_train)/len(df)*100:.1f}%)")
        logger.info(f"  ✓ Test set: {len(X_test)} samples ({len(X_test)/len(df)*100:.1f}%)")
        
        # Log target distribution in train/test sets
        train_dist = Y_train.value_counts()
        test_dist = Y_test.value_counts()
        logger.info(f"\nTarget Distribution in Training Set:")
        for value, count in train_dist.items():
            logger.info(f"  {value}: {count} ({count/len(Y_train)*100:.2f}%)")
        logger.info(f"\nTarget Distribution in Test Set:")
        for value, count in test_dist.items():
            logger.info(f"  {value}: {count} ({count/len(Y_test)*100:.2f}%)")
            
        logger.info(f"\n{'='*60}")
        logger.info(f"✓ DATA SPLITTING COMPLETE")
        logger.info(f"{'='*60}\n")
        return X_train, X_test, Y_train, Y_test