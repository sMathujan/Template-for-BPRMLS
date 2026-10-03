import json
import logging
import os
import joblib, sys
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from feature_binning import CustomBinningStratergy
from feature_encoding import OrdinalEncodingStratergy
from feature_scaling import MinMaxScalingStratergy

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'utils'))
from config import get_binning_config, get_encoding_config, get_scaling_config
logging.basicConfig(level=logging.INFO, format=
    '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

""" 
{
  "RowNumber": 1,
  "CustomerId": 15634602,
  "Firstname": "Grace",
  "Lastname": "Williams",
  "CreditScore": 619,
  "Geography": "France",
  "Gender": "Female",
  "Age": 42,
  "Tenure": 2,
  "Balance": 0,
  "NumOfProducts": 1,
  "HasCrCard": 1,
  "IsActiveMember": 1,
  "EstimatedSalary": 101348.88,
}

"""
class ModelInference:
    """
    Enhanced model inference class with comprehensive logging and error handling.
    """
    
    def __init__(self, model_path: str):
        """
        Initialize the model inference system.
        
        Args:
            model_path: Path to the trained model file
            
        Raises:
            ValueError: If model_path is invalid
            FileNotFoundError: If model file doesn't exist
        """
        logger.info(f"\n{'='*60}")
        logger.info("INITIALIZING MODEL INFERENCE")
        logger.info(f"{'='*60}")
        
        if not model_path or not isinstance(model_path, str):
            logger.error("✗ Invalid model path provided")
            raise ValueError("Invalid model path provided")
            
        self.model_path = model_path
        self.encoders = {}
        self.model = None
        
        logger.info(f"Model Path: {model_path}")
        
        try:
            # Load model and configurations
            self.load_model()
            self.binning_config = get_binning_config()
            self.encoding_config = get_encoding_config()
            self.scaling_config = get_scaling_config()
            
            # Initialize and load scaler
            self.scaler = MinMaxScalingStratergy()
            scaler_loaded = self.scaler.load_scaler('artifacts/scale')
            if not scaler_loaded:
                logger.warning("⚠ Could not load scaler artifacts. Feature scaling will be skipped during inference.")
            
            logger.info("✓ Model inference system initialized successfully")
            logger.info(f"{'='*60}\n")
            
        except Exception as e:
            logger.error(f"✗ Failed to initialize model inference: {str(e)}")
            raise

    def load_model(self) -> None:
        """
        Load the trained model from disk with validation.
        
        Raises:
            FileNotFoundError: If model file doesn't exist
            Exception: For any loading errors
        """
        logger.info("Loading trained model...")
        
        if not os.path.exists(self.model_path):
            logger.error(f"✗ Model file not found: {self.model_path}")
            raise FileNotFoundError(f"Model file not found: {self.model_path}")
        
        try:
            self.model = joblib.load(self.model_path)
            file_size = os.path.getsize(self.model_path) / (1024**2)  # MB
            
            logger.info(f"✓ Model loaded successfully:")
            logger.info(f"  • Model Type: {type(self.model).__name__}")
            logger.info(f"  • File Size: {file_size:.2f} MB")
            
        except Exception as e:
            logger.error(f"✗ Failed to load model: {str(e)}")
            raise

    def load_encoders(self, encoders_dir: str) -> None:
        """
        Load feature encoders from directory with validation and logging.
        
        Args:
            encoders_dir: Directory containing encoder JSON files
            
        Raises:
            FileNotFoundError: If encoders directory doesn't exist
            Exception: For any loading errors
        """
        logger.info(f"\n{'='*50}")
        logger.info("LOADING FEATURE ENCODERS")
        logger.info(f"{'='*50}")
        
        if not os.path.exists(encoders_dir):
            logger.error(f"✗ Encoders directory not found: {encoders_dir}")
            raise FileNotFoundError(f"Encoders directory not found: {encoders_dir}")
        
        try:
            encoder_files = [f for f in os.listdir(encoders_dir) if f.endswith('_encoder.json')]
            
            if not encoder_files:
                logger.warning("⚠ No encoder files found in directory")
                return
            
            logger.info(f"Found {len(encoder_files)} encoder files")
            
            for file in encoder_files:
                feature_name = file.split('_encoder.json')[0]
                file_path = os.path.join(encoders_dir, file)
                
                with open(file_path, 'r') as f:
                    encoder_data = json.load(f)
                    self.encoders[feature_name] = encoder_data
                    
                logger.info(f"  ✓ Loaded encoder for '{feature_name}': {len(encoder_data)} mappings")
            
            logger.info(f"✓ All encoders loaded successfully")
            logger.info(f"{'='*50}\n")
            
        except Exception as e:
            logger.error(f"✗ Failed to load encoders: {str(e)}")
            raise

    def preprocess_input(self, data: Dict[str, Any]) -> pd.DataFrame:
        """
        Preprocess input data for model prediction with comprehensive logging.
        
        Args:
            data: Input data dictionary
            
        Returns:
            Preprocessed DataFrame ready for prediction
            
        Raises:
            ValueError: If input data is invalid
            Exception: For any preprocessing errors
        """
        logger.info(f"\n{'='*50}")
        logger.info("PREPROCESSING INPUT DATA")
        logger.info(f"{'='*50}")
        
        if not data or not isinstance(data, dict):
            logger.error("✗ Input data must be a non-empty dictionary")
            raise ValueError("Input data must be a non-empty dictionary")
        
        try:
            # Convert to DataFrame
            df = pd.DataFrame([data])
            logger.info(f"✓ Input data converted to DataFrame: {df.shape}")
            logger.info(f"  • Input features: {list(df.columns)}")
            
            # Apply encoders
            if self.encoders:
                logger.info("Applying feature encoders...")
                for col, encoder_data in self.encoders.items():
                    if col in df.columns:
                        original_value = df[col].iloc[0]
                        
                        # Check if this is one-hot encoding
                        if isinstance(encoder_data, dict) and 'encoding_type' in encoder_data and encoder_data['encoding_type'] == 'one_hot':
                            # Apply one-hot encoding
                            categories = encoder_data['categories']
                            logger.info(f"  Applying one-hot encoding for '{col}' with categories: {categories}")
                            
                            # Create one-hot encoded columns
                            for category in categories:
                                new_col_name = f"{col}_{category}"
                                df[new_col_name] = (df[col] == category).astype(int)
                                logger.info(f"    ✓ Created '{new_col_name}': {df[new_col_name].iloc[0]}")
                            
                            # Drop original column
                            df = df.drop(columns=[col])
                            logger.info(f"  ✓ One-hot encoded '{col}': {original_value} → {len(categories)} binary columns")
                        
                        else:
                            # Legacy label encoding (for backwards compatibility)
                            df[col] = df[col].map(encoder_data)
                            encoded_value = df[col].iloc[0]
                            logger.info(f"  ✓ Label encoded '{col}': {original_value} → {encoded_value}")
                    else:
                        logger.warning(f"  ⚠ Column '{col}' not found in input data")
            else:
                logger.info("No encoders available - skipping encoding step")

            # Apply feature binning (but keep original CreditScore for model compatibility)
            if 'CreditScore' in df.columns:
                logger.info("Applying feature binning for CreditScore...")
                original_score = df['CreditScore'].iloc[0]
                
                # Create binned column without dropping original
                def assign_bin(value):
                    if value <= 580:
                        return 0  # Poor
                    elif value <= 670:
                        return 1  # Fair
                    elif value <= 740:
                        return 2  # Good
                    elif value <= 800:
                        return 3  # Very Good
                    else:
                        return 4  # Excellent
                
                df['CreditScoreBins'] = df['CreditScore'].apply(assign_bin)
                binned_score = df['CreditScoreBins'].iloc[0]
                logger.info(f"  ✓ CreditScore binned: {original_score} → {binned_score} (kept original column)")
            else:
                logger.warning("  ⚠ CreditScore not found - skipping binning")

            # Apply ordinal encoding
            logger.info("Applying ordinal encoding...")
            ordinal_strategy = OrdinalEncodingStratergy(self.encoding_config['ordinal_mappings'])
            df = ordinal_strategy.encode(df)
            logger.info("  ✓ Ordinal encoding applied")

            # Apply feature scaling
            if hasattr(self, 'scaler') and hasattr(self.scaler, 'fitted') and self.scaler.fitted:
                logger.info("Applying feature scaling...")
                try:
                    df = self.scaler.transform(df, self.scaling_config['columns_to_scale'])
                    logger.info("  ✓ Feature scaling applied using pre-trained scaler")
                except Exception as e:
                    logger.warning(f"  ⚠ Feature scaling failed: {str(e)} - continuing without scaling")
            else:
                logger.warning("  ⚠ Scaler not available - skipping feature scaling")

            # Drop unnecessary columns
            drop_columns = ['RowNumber', 'CustomerId', 'Firstname', 'Lastname']
            existing_drop_columns = [col for col in drop_columns if col in df.columns]
            
            if existing_drop_columns:
                df = df.drop(columns=existing_drop_columns)
                logger.info(f"  ✓ Dropped columns: {existing_drop_columns}")
            
            # Reorder columns to match training data order for model compatibility
            expected_order = [
                'CreditScore', 'Age', 'Tenure', 'Balance', 'NumOfProducts', 
                'HasCrCard', 'IsActiveMember', 'EstimatedSalary', 'CreditScoreBins',
                'Geography_France', 'Geography_Germany', 'Geography_Spain',
                'Gender_Female', 'Gender_Male'
            ]
            
            # Only include columns that exist in the current dataframe
            available_columns = [col for col in expected_order if col in df.columns]
            df = df[available_columns]
            
            logger.info(f"✓ Preprocessing completed - Final shape: {df.shape}")
            logger.info(f"  • Final features (ordered): {list(df.columns)}")
            logger.info(f"{'='*50}\n")
            
            return df
            
        except Exception as e:
            logger.error(f"✗ Preprocessing failed: {str(e)}")
            raise
    
    def predict(self, data: Dict[str, Any]) -> Dict[str, str]:
        """
        Make prediction on input data with comprehensive logging.
        
        Args:
            data: Input data dictionary
            
        Returns:
            Dictionary containing prediction status and confidence
            
        Raises:
            ValueError: If input data is invalid
            Exception: For any prediction errors
        """
        logger.info(f"\n{'='*60}")
        logger.info("MAKING PREDICTION")
        logger.info(f"{'='*60}")
        
        if not data:
            logger.error("✗ Input data cannot be empty")
            raise ValueError("Input data cannot be empty")
        
        if self.model is None:
            logger.error("✗ Model not loaded")
            raise ValueError("Model not loaded")
        
        try:
            # Preprocess input data
            processed_data = self.preprocess_input(data)
            
            # Make prediction
            logger.info("Generating predictions...")
            y_pred = self.model.predict(processed_data)
            y_proba = self.model.predict_proba(processed_data)[:, 1]
            
            # Process results
            prediction = int(y_pred[0])
            probability = float(y_proba[0])
            
            status = 'Churn' if prediction == 1 else 'Retain'
            confidence = round(probability * 100, 2)
            
            result = {
                "Status": status,
                "Confidence": f"{confidence}%"
            }
            
            logger.info("✓ Prediction completed:")
            logger.info(f"  • Raw Prediction: {prediction}")
            logger.info(f"  • Raw Probability: {probability:.4f}")
            logger.info(f"  • Final Status: {status}")
            logger.info(f"  • Confidence: {confidence}%")
            logger.info(f"{'='*60}\n")
            
            return result
            
        except Exception as e:
            logger.error(f"✗ Prediction failed: {str(e)}")
            raise