"""
Model training module for Predictive Maintenance.
Handles model training, validation, and evaluation
"""

import pandas as pd
from sklearn.model_selection import cross_val_score, TimeSeriesSplit
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, precision_recall_curve
from xgboost import XGBClassifier
import joblib
import mlflow
import mlflow.xgboost
from pathlib import Path
import yaml
import logging
from typing import Tuple, Dict
import matplotlib.pyplot as plt
import seaborn as sns

logger = logging.getLogger(__name__)


class PredictiveMaintenanceTrainer:
    """Model trainer for predictive maintenance."""
    
    def __init__(self, config_path: str = "config/config.yaml"):
        """Initialize trainer with configuration.
        
        Args:
            config_path: Path to configuration file
        """
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.model_config = self.config['model']
        self.training_config = self.config['training']
        
        # Create model directory
        self.model_path = Path(self.config['data']['models_path'])
        self.model_path.mkdir(parents=True, exist_ok=True)
        
    def prepare_data(self, features_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """Prepare data for training with time-based split.
        
        Args:
            features_df: Complete feature dataframe
            
        Returns:
            X_train, X_test, y_train, y_test
        """
        logger.info("Preparing data for training...")
        
        # Time-based split to prevent data leakage
        train_end = pd.to_datetime(self.training_config['train_end'])
        test_start = pd.to_datetime(self.training_config['test_start'])
        
        # Split data
        train_mask = features_df['datetime'] <= train_end
        test_mask = features_df['datetime'] >= test_start
        
        train_data = features_df[train_mask].copy()
        test_data = features_df[test_mask].copy()
        
        # Prepare features and target
        feature_cols = [col for col in features_df.columns 
                       if col not in ['datetime', 'machineID', 'failure', 'failure_type']]
        
        X_train = train_data[feature_cols]
        X_test = test_data[feature_cols]
        y_train = train_data['failure']
        y_test = test_data['failure']
        
        logger.info(f"Training set: {X_train.shape}, positive rate: {y_train.mean():.3f}")
        logger.info(f"Test set: {X_test.shape}, positive rate: {y_test.mean():.3f}")
        
        return X_train, X_test, y_train, y_test
    
    def create_model(self) -> XGBClassifier:
        """Create and configure XGBoost model.
        
        Returns:
            Configured XGBoost model
        """
        model_params = self.model_config['xgboost'].copy()
        model = XGBClassifier(**model_params)
        return model
    
    def train_model(self, X_train: pd.DataFrame, y_train: pd.Series) -> XGBClassifier:
        """Train the model.
        
        Args:
            X_train: Training features
            y_train: Training target
            
        Returns:
            Trained model
        """
        logger.info("Training model...")
        
        # Handle class imbalance with scale_pos_weight
        pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
        
        model = self.create_model()
        model.set_params(scale_pos_weight=pos_weight)
        
        # Train model
        model.fit(X_train, y_train)
        
        logger.info("Model training completed")
        return model
    
    def evaluate_model(self, model: XGBClassifier, X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, float]:
        """Evaluate model performance.
        
        Args:
            model: Trained model
            X_test: Test features
            y_test: Test target
            
        Returns:
            Dictionary with evaluation metrics
        """
        logger.info("Evaluating model...")
        
        # Predictions
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        
        # Calculate metrics
        from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
        
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred),
            'recall': recall_score(y_test, y_pred),
            'f1': f1_score(y_test, y_pred),
            'auc': roc_auc_score(y_test, y_pred_proba)
        }
        
        # Print classification report
        logger.info("Classification Report:")
        logger.info("\n" + classification_report(y_test, y_pred))
        
        # Print metrics
        logger.info("Model Performance Metrics:")
        for metric, value in metrics.items():
            logger.info(f"{metric}: {value:.4f}")
        
        return metrics
    
    def plot_evaluation(self, model: XGBClassifier, X_test: pd.DataFrame, y_test: pd.Series, save_plots: bool = True):
        """Create evaluation plots.
        
        Args:
            model: Trained model
            X_test: Test features
            y_test: Test target
            save_plots: Whether to save plots to file
        """
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Confusion Matrix
        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', ax=axes[0, 0])
        axes[0, 0].set_title('Confusion Matrix')
        axes[0, 0].set_xlabel('Predicted')
        axes[0, 0].set_ylabel('Actual')
        
        # ROC Curve
        from sklearn.metrics import roc_curve
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
        axes[0, 1].plot(fpr, tpr, label=f'ROC Curve (AUC = {roc_auc_score(y_test, y_pred_proba):.3f})')
        axes[0, 1].plot([0, 1], [0, 1], 'k--')
        axes[0, 1].set_xlabel('False Positive Rate')
        axes[0, 1].set_ylabel('True Positive Rate')
        axes[0, 1].set_title('ROC Curve')
        axes[0, 1].legend()
        
        # Precision-Recall Curve
        precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
        axes[1, 0].plot(recall, precision)
        axes[1, 0].set_xlabel('Recall')
        axes[1, 0].set_ylabel('Precision')
        axes[1, 0].set_title('Precision-Recall Curve')
        
        # Feature Importance
        feature_importance = pd.DataFrame({
            'feature': X_test.columns,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False).head(20)
        
        sns.barplot(data=feature_importance, y='feature', x='importance', ax=axes[1, 1])
        axes[1, 1].set_title('Top 20 Feature Importances')
        
        plt.tight_layout()
        
        if save_plots:
            plot_path = self.model_path / "evaluation_plots.png"
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            logger.info(f"Evaluation plots saved to {plot_path}")
        
        plt.show()
    
    def cross_validate(self, X_train: pd.DataFrame, y_train: pd.Series) -> Dict[str, float]:
        """Perform cross-validation.
        
        Args:
            X_train: Training features
            y_train: Training target
            
        Returns:
            Cross-validation scores
        """
        logger.info("Performing cross-validation...")
        
        model = self.create_model()
        
        # Use TimeSeriesSplit to respect temporal order
        cv = TimeSeriesSplit(n_splits=self.model_config['cv_folds'])
        
        # Calculate cross-validation scores
        cv_scores = {}
        for metric in ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']:
            scores = cross_val_score(model, X_train, y_train, cv=cv, scoring=metric)
            cv_scores[f'cv_{metric}_mean'] = scores.mean()
            cv_scores[f'cv_{metric}_std'] = scores.std()
        
        logger.info("Cross-validation results:")
        for metric, value in cv_scores.items():
            logger.info(f"{metric}: {value:.4f}")
        
        return cv_scores
    
    def save_model(self, model: XGBClassifier, model_name: str = "predictive_maintenance_model.joblib"):
        """Save trained model.
        
        Args:
            model: Trained model
            model_name: Model filename
        """
        model_file = self.model_path / model_name
        joblib.dump(model, model_file)
        logger.info(f"Model saved to {model_file}")
    
    def load_model(self, model_name: str = "predictive_maintenance_model.joblib") -> XGBClassifier:
        """Load trained model.
        
        Args:
            model_name: Model filename
            
        Returns:
            Loaded model
        """
        model_file = self.model_path / model_name
        model = joblib.load(model_file)
        logger.info(f"Model loaded from {model_file}")
        return model
    
    def track_experiment(self, model: XGBClassifier, metrics: Dict[str, float], 
                        cv_scores: Dict[str, float]):
        """Track experiment with MLflow.
        
        Args:
            model: Trained model
            metrics: Evaluation metrics
            cv_scores: Cross-validation scores
        """
        logger.info("Tracking experiment with MLflow...")
        
        with mlflow.start_run():
            # Log parameters
            mlflow.log_params(self.model_config['xgboost'])
            
            # Log metrics
            mlflow.log_metrics(metrics)
            mlflow.log_metrics(cv_scores)
            
            # Log model
            mlflow.xgboost.log_model(model, "model")
            
            # Log configuration
            mlflow.log_artifact("config/config.yaml")
            
        logger.info("Experiment tracked successfully")
    
    def run_training_pipeline(self, features_df: pd.DataFrame) -> XGBClassifier:
        """Run complete training pipeline.
        
        Args:
            features_df: Complete feature dataframe
            
        Returns:
            Trained model
        """
        logger.info("Running complete training pipeline...")
        
        # Prepare data
        X_train, X_test, y_train, y_test = self.prepare_data(features_df)
        
        # Cross-validation
        cv_scores = self.cross_validate(X_train, y_train)
        
        # Train model
        model = self.train_model(X_train, y_train)
        
        # Evaluate model
        metrics = self.evaluate_model(model, X_test, y_test)
        
        # Create evaluation plots
        self.plot_evaluation(model, X_test, y_test)
        
        # Save model
        self.save_model(model)
        
        # Track experiment
        self.track_experiment(model, metrics, cv_scores)
        
        logger.info("Training pipeline completed successfully")
        
        return model 