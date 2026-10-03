import logging
import pandas as pd
import numpy as np
import os
import json
import joblib
from enum import Enum
from typing import List
from abc import ABC, abstractmethod
from sklearn.preprocessing import MinMaxScaler
logging.basicConfig(level=logging.INFO, format=
    '%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class FeatureScalingStrategy(ABC):

    @abstractmethod
    def scale(self, df: pd.DataFrame, columns_to_scale: List[str]) -> pd.DataFrame:
        pass


class ScalingType(str, Enum):
    MINMAX = 'minmax'
    STANDARD = 'standard'

class MinMaxScalingStratergy(FeatureScalingStrategy):
    def __init__(self):
        self.scaler = MinMaxScaler()
        self.fitted = False
        logger.info("MinMaxScalingStrategy initialized")

    def scale(self, df, columns_to_scale):
        logger.info(f"\n{'='*60}")
        logger.info(f"FEATURE SCALING - MIN-MAX")
        logger.info(f"{'='*60}")
        logger.info(f'Starting Min-Max scaling for {len(columns_to_scale)} columns: {columns_to_scale}')
        
        # Log statistics before scaling
        logger.info(f"\nStatistics BEFORE scaling:")
        for col in columns_to_scale:
            col_stats = {
                'min': df[col].min(),
                'max': df[col].max(),
                'mean': df[col].mean(),
                'std': df[col].std()
            }
            logger.info(f"  {col}: Min={col_stats['min']:.2f}, Max={col_stats['max']:.2f}, Mean={col_stats['mean']:.2f}, Std={col_stats['std']:.2f}")
        
        # Apply scaling
        df[columns_to_scale] = self.scaler.fit_transform(df[columns_to_scale])
        self.fitted = True
        
        # Log min/max values learned by scaler
        logger.info(f"\nScaler Parameters:")
        for i, col in enumerate(columns_to_scale):
            logger.info(f"  {col}: Data min={self.scaler.data_min_[i]:.2f}, Data max={self.scaler.data_max_[i]:.2f}")
        
        # Log statistics after scaling
        logger.info(f"\nStatistics AFTER scaling:")
        for col in columns_to_scale:
            col_stats = {
                'min': df[col].min(),
                'max': df[col].max(),
                'mean': df[col].mean(),
                'std': df[col].std()
            }
            logger.info(f"  {col}: Min={col_stats['min']:.4f}, Max={col_stats['max']:.4f}, Mean={col_stats['mean']:.4f}, Std={col_stats['std']:.4f}")
            
            # Check if scaling worked correctly
            if abs(col_stats['min']) > 0.001 or abs(col_stats['max'] - 1.0) > 0.001:
                logger.warning(f"  ⚠ Column '{col}' may not be properly scaled to [0,1] range")
        
        logger.info(f"\n{'='*60}")
        logger.info(f'✓ MIN-MAX SCALING COMPLETE - {len(columns_to_scale)} columns processed')
        logger.info(f"{'='*60}\n")
        return df 
    
    def get_scaler(self):
        return self.scaler
    
    def save_scaler(self, columns_to_scale: List[str], save_dir: str = 'artifacts/scale') -> bool:
        """Save the fitted scaler and metadata for inference."""
        logger.info(f"\n{'='*50}")
        logger.info("SAVING SCALER ARTIFACTS")
        logger.info(f"{'='*50}")
        
        if not self.fitted:
            logger.error("✗ Scaler not fitted yet. Cannot save.")
            return False
        
        try:
            # Create save directory
            os.makedirs(save_dir, exist_ok=True)
            
            # Save scaler object
            scaler_path = os.path.join(save_dir, 'minmax_scaler.joblib')
            joblib.dump(self.scaler, scaler_path)
            
            # Save metadata
            metadata = {
                'columns_to_scale': columns_to_scale,
                'data_min': self.scaler.data_min_.tolist(),
                'data_max': self.scaler.data_max_.tolist(),
                'n_features': len(columns_to_scale),
                'scaling_type': 'minmax'
            }
            
            metadata_path = os.path.join(save_dir, 'scaling_metadata.json')
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            logger.info(f"✓ Scaler saved to: {scaler_path}")
            logger.info(f"✓ Metadata saved to: {metadata_path}")
            logger.info(f"✓ Columns scaled: {columns_to_scale}")
            logger.info(f"✓ Data ranges saved: min={metadata['data_min']}, max={metadata['data_max']}")
            logger.info(f"{'='*50}\n")
            
            return True
            
        except Exception as e:
            logger.error(f"✗ Failed to save scaler: {str(e)}")
            return False
    
    def load_scaler(self, save_dir: str = 'artifacts/scale') -> bool:
        """Load the fitted scaler and metadata for inference."""
        logger.info(f"\n{'='*50}")
        logger.info("LOADING SCALER ARTIFACTS")
        logger.info(f"{'='*50}")
        
        scaler_path = os.path.join(save_dir, 'minmax_scaler.joblib')
        metadata_path = os.path.join(save_dir, 'scaling_metadata.json')
        
        if not os.path.exists(scaler_path):
            logger.error(f"✗ Scaler file not found: {scaler_path}")
            return False
        
        if not os.path.exists(metadata_path):
            logger.error(f"✗ Metadata file not found: {metadata_path}")
            return False
        
        try:
            # Load scaler
            self.scaler = joblib.load(scaler_path)
            self.fitted = True
            
            # Load metadata
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            
            logger.info(f"✓ Scaler loaded from: {save_dir}")
            logger.info(f"✓ Columns to scale: {metadata['columns_to_scale']}")
            logger.info(f"✓ Features count: {metadata['n_features']}")
            logger.info(f"✓ Data ranges: min={metadata['data_min']}, max={metadata['data_max']}")
            logger.info(f"{'='*50}\n")
            
            return True
            
        except Exception as e:
            logger.error(f"✗ Failed to load scaler: {str(e)}")
            return False
    
    def transform(self, df: pd.DataFrame, columns_to_scale: List[str]) -> pd.DataFrame:
        """Apply the loaded scaler to transform data (no fitting)."""
        logger.info(f"\n{'='*60}")
        logger.info(f"FEATURE SCALING - TRANSFORM ONLY")
        logger.info(f"{'='*60}")
        logger.info(f'Applying scaling transformation for {len(columns_to_scale)} columns: {columns_to_scale}')
        
        if not self.fitted:
            logger.error("✗ Scaler not loaded/fitted. Cannot transform.")
            raise ValueError("Scaler not loaded/fitted. Call load_scaler() first.")
        
        # Log statistics before scaling
        logger.info(f"\nStatistics BEFORE scaling:")
        for col in columns_to_scale:
            if col in df.columns:
                original_value = df[col].iloc[0] if len(df) > 0 else 0
                logger.info(f"  {col}: {original_value}")
        
        # Apply transformation (not fitting)
        df_scaled = df.copy()
        df_scaled[columns_to_scale] = self.scaler.transform(df[columns_to_scale])
        
        # Log statistics after scaling
        logger.info(f"\nStatistics AFTER scaling:")
        for col in columns_to_scale:
            if col in df_scaled.columns:
                scaled_value = df_scaled[col].iloc[0] if len(df_scaled) > 0 else 0
                original_value = df[col].iloc[0] if len(df) > 0 else 0
                logger.info(f"  ✓ Transformed '{col}': {original_value} → {scaled_value:.4f}")
        
        logger.info(f"\n{'='*60}")
        logger.info(f'✓ SCALING TRANSFORMATION COMPLETE - {len(columns_to_scale)} columns processed')
        logger.info(f"{'='*60}\n")
        return df_scaled
    
    @classmethod
    def create_scaler_artifacts_from_raw_data(
        cls, 
        raw_data_path: str = 'data/raw/ChurnModelling.csv', 
        columns_to_scale: List[str] = None, 
        save_dir: str = 'artifacts/scale'
    ) -> bool:
        """Create scaler artifacts from raw data for inference pipeline."""
        logger.info(f"\n{'='*60}")
        logger.info("CREATING SCALER ARTIFACTS FROM RAW DATA")
        logger.info(f"{'='*60}")
        
        if not os.path.exists(raw_data_path):
            logger.error(f"✗ Raw data file not found: {raw_data_path}")
            return False
        
        if columns_to_scale is None:
            columns_to_scale = ["Balance", "EstimatedSalary", "Age"]
        
        try:
            # Load raw data
            logger.info(f"Loading raw data from: {raw_data_path}")
            df = pd.read_csv(raw_data_path)
            logger.info(f"✓ Raw data loaded: {df.shape}")
            
            # Validate columns exist
            missing_cols = [col for col in columns_to_scale if col not in df.columns]
            if missing_cols:
                logger.error(f"✗ Missing columns in raw data: {missing_cols}")
                return False
            
            # Create and fit scaler on raw data
            logger.info(f"Fitting scaler on columns: {columns_to_scale}")
            scaler_instance = cls()
            
            # Log raw data statistics
            logger.info(f"\nRaw data statistics:")
            for col in columns_to_scale:
                stats = {
                    'min': df[col].min(),
                    'max': df[col].max(),
                    'mean': df[col].mean(),
                    'std': df[col].std()
                }
                logger.info(f"  {col}: Min={stats['min']:.2f}, Max={stats['max']:.2f}, Mean={stats['mean']:.2f}, Std={stats['std']:.2f}")
            
            # Fit the scaler on raw data (just to get parameters, don't scale the data)
            scaler_instance.scaler.fit(df[columns_to_scale])
            scaler_instance.fitted = True
            
            # Log scaler parameters
            logger.info(f"\nScaler parameters learned:")
            for i, col in enumerate(columns_to_scale):
                logger.info(f"  {col}: Data min={scaler_instance.scaler.data_min_[i]:.2f}, Data max={scaler_instance.scaler.data_max_[i]:.2f}")
            
            # Save scaler artifacts
            success = scaler_instance.save_scaler(columns_to_scale, save_dir)
            
            if success:
                logger.info(f"✓ Scaler artifacts created successfully!")
                logger.info(f"{'='*60}\n")
                return True
            else:
                logger.error(f"✗ Failed to save scaler artifacts")
                return False
                
        except Exception as e:
            logger.error(f"✗ Failed to create scaler artifacts: {str(e)}")
            return False


if __name__ == "__main__":
    print("🔧 Creating Scaler Artifacts for Inference")
    success = MinMaxScalingStratergy.create_scaler_artifacts_from_raw_data()
    if success:
        print("🎉 Scaler artifacts created successfully!")
    else:
        print("❌ Failed to create scaler artifacts.")
        exit(1)