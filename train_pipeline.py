#!/usr/bin/env python3
"""
Main training pipeline for Predictive Maintenance.
Orchestrates data loading, feature engineering, model training, and evaluation.
"""

import logging
import sys
from pathlib import Path

# Add src to path
sys.path.append('src')

from data.data_loader import PredictiveMaintenanceDataLoader
from features.feature_engineer import FeatureEngineer
from models.trainer import PredictiveMaintenanceTrainer

# Configuration
USE_FEATURES_CACHE = True# Set to False to force recalculation of features

# Configure logging
def setup_logging():
    """Setup logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('training.log'),
            logging.StreamHandler()
        ]
    )

def main():
    """Main training pipeline."""
    setup_logging()
    logger = logging.getLogger(__name__)
    
    logger.info("=" * 60)
    logger.info("PREDICTIVE MAINTENANCE TRAINING PIPELINE")
    logger.info("=" * 60)
    
    try:
        # Initialize components
        logger.info("Initializing pipeline components...")
        data_loader = PredictiveMaintenanceDataLoader()
        feature_engineer = FeatureEngineer()
        trainer = PredictiveMaintenanceTrainer()
        
        # Step 1: Load data
        logger.info("\n" + "="*40)
        logger.info("STEP 1: DATA LOADING")
        logger.info("="*40)
        
        data = data_loader.load_all_data()
        
        # Print data summary
        summary = data_loader.get_data_summary()
        for dataset_name, info in summary.items():
            logger.info(f"{dataset_name.upper()}:")
            logger.info(f"  Rows: {info['rows']:,}")
            logger.info(f"  Columns: {info['columns']}")
            if info['date_range']:
                logger.info(f"  Date range: {info['date_range']['start']} to {info['date_range']['end']}")
            if 'machines' in info:
                logger.info(f"  Machines: {info['machines']}")
            logger.info("")
        
        # Step 2: Feature engineering
        logger.info("\n" + "="*40)
        logger.info("STEP 2: FEATURE ENGINEERING")
        logger.info("="*40)
        
        features_df = feature_engineer.build_features(data, use_cache=USE_FEATURES_CACHE)
        
        # Save features (only if we computed them from scratch)
        if not USE_FEATURES_CACHE or not Path('data/features/features.csv').exists():
            feature_engineer.save_features(features_df)
        
        logger.info(f"Feature engineering completed!")
        logger.info(f"Final dataset shape: {features_df.shape}")
        logger.info(f"Failure rate: {features_df['failure'].mean():.4f}")
        
        # Step 3: Model training
        logger.info("\n" + "="*40)
        logger.info("STEP 3: MODEL TRAINING")
        logger.info("="*40)
        
        model = trainer.run_training_pipeline(features_df)
        
        logger.info("Training pipeline completed successfully!")
        
        # Step 4: Final summary
        logger.info("\n" + "="*40)
        logger.info("PIPELINE SUMMARY")
        logger.info("="*40)
        
        logger.info(f"✓ Data loaded: {len(data)} datasets")
        logger.info(f"✓ Features created: {features_df.shape[1] - 4} features")  # Excluding datetime, machineID, failure, failure_type
        logger.info(f"✓ Model trained and saved")
        logger.info(f"✓ Evaluation plots created")
        logger.info(f"✓ Experiment tracked with MLflow")
        
        logger.info("\nArtifacts created:")
        logger.info(f"  - Model: models/predictive_maintenance_model.joblib")
        logger.info(f"  - Features: data/features/features.csv")
        logger.info(f"  - Plots: models/evaluation_plots.png")
        logger.info(f"  - Logs: training.log")
        
        logger.info("\nNext steps:")
        logger.info("  1. Review model performance in the evaluation plots")
        logger.info("  2. Test the API with: python -m uvicorn src.api.app:app --reload")
        logger.info("  3. Monitor model performance in production")
        
    except Exception as e:
        logger.error(f"Pipeline failed with error: {e}")
        raise

if __name__ == "__main__":
    main() 