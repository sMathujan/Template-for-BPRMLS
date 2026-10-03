import logging
import pandas as pd
from abc import ABC, abstractmethod
logging.basicConfig(level=logging.INFO, format=
    '%(asctime)s - %(levelname)s - %(message)s')

class OutlierDetectionStrategy(ABC):
    @abstractmethod
    def detect_outliers(self, df: pd.DataFrame, columns: list) -> pd.DataFrame:
        pass

class IQROutlierDetection(OutlierDetectionStrategy):
    def detect_outliers(self, df, columns):
        logging.info(f"\n{'='*60}")
        logging.info(f"OUTLIER DETECTION - IQR METHOD")
        logging.info(f"{'='*60}")
        logging.info(f"Starting IQR outlier detection for columns: {columns}")
        outliers =pd.DataFrame(False, index=df.index, columns=columns)
        
        for col in columns:
            logging.info(f"\n--- Processing column: {col} ---")
            df[col] = df[col].astype(float)
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1 
            print(f"IQR for {col}: {IQR}")
            logging.info(f"  Q1: {Q1:.2f}, Q3: {Q3:.2f}, IQR: {IQR:.2f}")
            
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            outliers[col] = (df[col] < lower_bound) | (df[col] > upper_bound)
            outlier_count = outliers[col].sum()
            logging.info(f"  ✓ Found {outlier_count} outliers ({outlier_count/len(df)*100:.2f}%) out of bounds [{lower_bound:.2f}, {upper_bound:.2f}]")
        
        total_outliers = outliers.any(axis=1).sum()
        logging.info(f"\n{'='*60}")
        logging.info(f'✓ OUTLIER DETECTION COMPLETE - Total rows with outliers: {total_outliers} ({total_outliers/len(df)*100:.2f}%)')
        logging.info(f"{'='*60}\n")
        return outliers

class OutlierDetector:
    def __init__(self, strategy):
        self._strategy = strategy
        logging.info(f"OutlierDetector initialized with strategy: {strategy.__class__.__name__}")

    def detect_outliers(self, df, selected_columns):
        logging.info(f"Detecting outliers in {len(selected_columns)} columns from dataframe with {len(df)} rows")
        return self._strategy.detect_outliers(df, selected_columns)
    
    def handle_outliers(self, df, selected_columns, method='remove'):
        logging.info(f"\n{'='*60}")
        logging.info(f"OUTLIER HANDLING - {method.upper()}")
        logging.info(f"{'='*60}")
        logging.info(f"Handling outliers using method: {method}")
        initial_rows = len(df)
        outliers = self.detect_outliers(df, selected_columns)
        outlier_count = outliers.sum(axis=1)
        rows_to_remove = outlier_count >= 2
        rows_removed = rows_to_remove.sum()
        logging.info(f"✓ Removing rows with 2 or more outliers: {rows_removed} rows ({rows_removed/initial_rows*100:.2f}%)")
        cleaned_df = df[~rows_to_remove]
        logging.info(f"✓ Outlier removal complete - Remaining rows: {len(cleaned_df)} ({len(cleaned_df)/initial_rows*100:.2f}%)")
        logging.info(f"{'='*60}\n")
        return cleaned_df