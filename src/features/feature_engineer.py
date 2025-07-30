"""
Feature engineering module for Predictive Maintenance.
Creates lag features, aggregations, and target variables for machine failure prediction.
"""

import pandas as pd
from typing import Dict
import logging
import yaml
from pathlib import Path

logger = logging.getLogger(__name__)


class FeatureEngineer:
    """Feature engineering for predictive maintenance dataset."""
    
    def __init__(self, config_path: str = "config/config.yaml"):
        """Initialize feature engineer with configuration.
        
        Args:
            config_path: Path to configuration file
        """
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.telemetry_features = self.config['feature_engineering']['telemetry_features']
        self.telemetry_windows = self.config['feature_engineering']['telemetry_windows']
        self.error_windows = self.config['feature_engineering']['error_windows']
        self.telemetry_stats = self.config['feature_engineering']['telemetry_stats']
        self.prediction_window = self.config['feature_engineering']['prediction_window']
        
    def create_telemetry_features(self, telemetry_df: pd.DataFrame) -> pd.DataFrame:
        """Create aggregated features from telemetry data (optimized version).
        
        Args:
            telemetry_df: Raw telemetry data
            
        Returns:
            DataFrame with aggregated telemetry features
        """
        logger.info("Creating telemetry features...")
        
        # Create datetime index for easier resampling
        telemetry_df = telemetry_df.set_index('datetime')
        
        feature_dfs = []
        
        # Group by machine for efficient processing
        for machine_id in telemetry_df['machineID'].unique():
            if machine_id % 20 == 0:  # Progress logging
                logger.info(f"Processing machine {machine_id}...")
                
            machine_data = telemetry_df[telemetry_df['machineID'] == machine_id].copy()
            
            # Resample to 3H intervals
            resampled = machine_data.resample('3H').mean()
            
            machine_features = pd.DataFrame()
            machine_features['datetime'] = resampled.index
            machine_features['machineID'] = machine_id
            
            # Create rolling features for each sensor
            for feature in self.telemetry_features:
                if feature in resampled.columns:
                    # Only use 3h and 24h windows for efficiency
                    for window_hours in [3, 24]:
                        window_periods = max(1, window_hours // 3)  # Convert to 3H periods
                        
                        rolling = resampled[feature].rolling(window=window_periods, min_periods=1)
                        machine_features[f'{feature}_mean_{window_hours}h'] = rolling.mean().values
                        # Use fillna(0) for std to avoid NaN values that would be dropped
                        machine_features[f'{feature}_std_{window_hours}h'] = rolling.std().fillna(0).values
                        
                        # Only add min/max for 24h window to reduce features
                        if window_hours == 24:
                            machine_features[f'{feature}_min_{window_hours}h'] = rolling.min().values
                            machine_features[f'{feature}_max_{window_hours}h'] = rolling.max().values
            
            feature_dfs.append(machine_features)
        
        # Concatenate all machines
        telemetry_features = pd.concat(feature_dfs, ignore_index=True)
        
        # Fill any remaining NaN values with 0 instead of dropping rows
        telemetry_features = telemetry_features.fillna(0)
        
        logger.info(f"Created telemetry features: {telemetry_features.shape}")
        return telemetry_features
    
    def create_error_features(self, errors_df: pd.DataFrame, telemetry_df: pd.DataFrame) -> pd.DataFrame:
        """Create error count features (optimized version).
        
        Args:
            errors_df: Error data
            telemetry_df: Telemetry data for time alignment
            
        Returns:
            DataFrame with error count features
        """
        logger.info("Creating error features...")
        
        # Create a simpler time grid - use 3H intervals from telemetry
        # Create time grid manually to avoid index issues
        time_points = []
        for machine_id in telemetry_df['machineID'].unique():
            machine_data = telemetry_df[telemetry_df['machineID'] == machine_id]
            dates = pd.date_range(
                start=machine_data['datetime'].min(), 
                end=machine_data['datetime'].max(), 
                freq='3H'
            )
            for date in dates:
                time_points.append({'datetime': date, 'machineID': machine_id})
        
        time_grid = pd.DataFrame(time_points)
        
        # Process errors by machine for efficiency
        error_features_list = []
        
        for machine_id in time_grid['machineID'].unique():
            machine_time_grid = time_grid[time_grid['machineID'] == machine_id].copy()
            machine_errors = errors_df[errors_df['machineID'] == machine_id].copy()
            
            if len(machine_errors) == 0:
                # No errors for this machine, create zero features
                machine_features = machine_time_grid.copy()
                for error_type in ['error1', 'error2', 'error3', 'error4', 'error5']:
                    machine_features[f'{error_type}_count_24h'] = 0
                error_features_list.append(machine_features)
                continue
            
            # Count errors in time windows
            machine_features = machine_time_grid.copy()
            
            for error_type in ['error1', 'error2', 'error3', 'error4', 'error5']:
                error_times = machine_errors[machine_errors['errorID'] == error_type]['datetime']
                
                # Count errors in 24h window before each time point
                counts = []
                for timestamp in machine_time_grid['datetime']:
                    window_start = timestamp - pd.Timedelta(hours=24)
                    count = len(error_times[(error_times >= window_start) & (error_times < timestamp)])
                    counts.append(count)
                
                machine_features[f'{error_type}_count_24h'] = counts
            
            error_features_list.append(machine_features)
        
        # Combine all machines
        error_features = pd.concat(error_features_list, ignore_index=True)
        
        logger.info(f"Created error features: {error_features.shape}")
        return error_features
    
    def create_maintenance_features(self, maintenance_df: pd.DataFrame, telemetry_df: pd.DataFrame) -> pd.DataFrame:
        """Create days since last maintenance features (simplified version).
        
        Args:
            maintenance_df: Maintenance data
            telemetry_df: Telemetry data for time alignment
            
        Returns:
            DataFrame with maintenance features
        """
        logger.info("Creating maintenance features...")
        
        # Create time grid using same approach as error features
        time_points = []
        for machine_id in telemetry_df['machineID'].unique():
            machine_data = telemetry_df[telemetry_df['machineID'] == machine_id]
            dates = pd.date_range(
                start=machine_data['datetime'].min(), 
                end=machine_data['datetime'].max(), 
                freq='3H'
            )
            for date in dates:
                time_points.append({'datetime': date, 'machineID': machine_id})
        
        time_grid = pd.DataFrame(time_points)
        
        maintenance_features_list = []
        
        for machine_id in time_grid['machineID'].unique():
            machine_time_grid = time_grid[time_grid['machineID'] == machine_id].copy()
            machine_maint = maintenance_df[maintenance_df['machineID'] == machine_id].copy()
            
            machine_features = machine_time_grid.copy()
            
            # Calculate days since last maintenance for each component
            for comp in ['comp1', 'comp2', 'comp3', 'comp4']:
                comp_maint_dates = machine_maint[machine_maint['comp'] == comp]['datetime']
                
                days_since = []
                for timestamp in machine_time_grid['datetime']:
                    # Find last maintenance before this timestamp
                    previous_maint = comp_maint_dates[comp_maint_dates <= timestamp]
                    
                    if len(previous_maint) > 0:
                        last_maint = previous_maint.max()
                        days_diff = (timestamp - last_maint).total_seconds() / (24 * 3600)
                        days_since.append(days_diff)
                    else:
                        # No previous maintenance, use large value
                        days_since.append(365.0)  # Default to 1 year
                
                machine_features[f'{comp}_days_since'] = days_since
            
            maintenance_features_list.append(machine_features)
        
        # Combine all machines
        maintenance_features = pd.concat(maintenance_features_list, ignore_index=True)
        
        logger.info(f"Created maintenance features: {maintenance_features.shape}")
        return maintenance_features
    
    def create_target_variable(self, failures_df: pd.DataFrame, telemetry_df: pd.DataFrame) -> pd.DataFrame:
        """Create target variable for failure prediction.
        
        Args:
            failures_df: Failure data
            telemetry_df: Telemetry data for time alignment
            
        Returns:
            DataFrame with target variable
        """
        logger.info("Creating target variable...")
        
        # Create time grid
        time_grid = telemetry_df[['datetime', 'machineID']].drop_duplicates()
        time_grid = time_grid.sort_values(['machineID', 'datetime'])
        
        # Initialize target variable
        time_grid['failure'] = 0
        time_grid['failure_type'] = 'none'
        
        # For each failure, mark the prediction window before it
        for _, failure in failures_df.iterrows():
            failure_time = failure['datetime']
            machine_id = failure['machineID']
            failure_type = failure['failure']
            
            # Find time window before failure
            window_start = failure_time - pd.Timedelta(hours=self.prediction_window)
            
            # Mark samples in this window as positive
            mask = (
                (time_grid['machineID'] == machine_id) &
                (time_grid['datetime'] >= window_start) &
                (time_grid['datetime'] < failure_time)
            )
            
            time_grid.loc[mask, 'failure'] = 1
            time_grid.loc[mask, 'failure_type'] = failure_type
        
        logger.info(f"Created target variable: {time_grid['failure'].sum()} positive samples out of {len(time_grid)}")
        
        return time_grid[['datetime', 'machineID', 'failure', 'failure_type']]
    
    def create_machine_features(self, machines_df: pd.DataFrame, telemetry_df: pd.DataFrame) -> pd.DataFrame:
        """Add machine metadata features.
        
        Args:
            machines_df: Machine metadata
            telemetry_df: Telemetry data for merging
            
        Returns:
            DataFrame with machine features
        """
        logger.info("Adding machine features...")
        
        # Create dummy variables for model type
        machine_features = pd.get_dummies(machines_df, columns=['model'], prefix='model')
        
        # Create time grid using same approach as other features
        time_points = []
        for machine_id in telemetry_df['machineID'].unique():
            machine_data = telemetry_df[telemetry_df['machineID'] == machine_id]
            dates = pd.date_range(
                start=machine_data['datetime'].min(), 
                end=machine_data['datetime'].max(), 
                freq='3H'
            )
            for date in dates:
                time_points.append({'datetime': date, 'machineID': machine_id})
        
        time_grid = pd.DataFrame(time_points)
        
        result = time_grid.merge(machine_features, on='machineID', how='left')
        
        logger.info(f"Added machine features: {result.shape}")
        return result
    
    def build_features(self, data_dict: Dict[str, pd.DataFrame], use_cache: bool = True) -> pd.DataFrame:
        """Build complete feature set.
        
        Args:
            data_dict: Dictionary containing all datasets
            use_cache: If True, try to load existing features from cache
            
        Returns:
            Complete feature dataframe
        """
        logger.info("Building complete feature set...")
        
        # Check if features already exist and should use cache
        features_path = Path(self.config['data']['features_path']) / "features.csv"
        if use_cache and features_path.exists():
            logger.info("Found existing features, loading from cache...")
            logger.info("To recalculate features, set use_cache=False or delete the features file")
            return self.load_features()
        
        logger.info("Computing features from scratch...")
        
        # Extract datasets
        telemetry_df = data_dict['telemetry']
        errors_df = data_dict['errors']
        failures_df = data_dict['failures']
        maintenance_df = data_dict['maintenance']
        machines_df = data_dict['machines']
        
        # Create individual feature sets
        telemetry_features = self.create_telemetry_features(telemetry_df)
        error_features = self.create_error_features(errors_df, telemetry_df)
        maintenance_features = self.create_maintenance_features(maintenance_df, telemetry_df)
        machine_features = self.create_machine_features(machines_df, telemetry_df)
        target_variable = self.create_target_variable(failures_df, telemetry_df)
        
        # Merge all features
        logger.info("Merging all features...")
        
        # Start with telemetry features as base
        final_features = telemetry_features
        
        # Merge other features
        final_features = final_features.merge(
            error_features, on=['datetime', 'machineID'], how='left'
        )
        
        final_features = final_features.merge(
            maintenance_features, on=['datetime', 'machineID'], how='left'
        )
        
        final_features = final_features.merge(
            machine_features, on=['datetime', 'machineID'], how='left'
        )
        
        final_features = final_features.merge(
            target_variable, on=['datetime', 'machineID'], how='left'
        )
        
        # Fill missing values
        final_features = final_features.fillna(0)
        
        # Remove samples without target variable
        final_features = final_features.dropna(subset=['failure'])
        
        logger.info(f"Final feature set: {final_features.shape}")
        logger.info(f"Positive samples: {final_features['failure'].sum()}")
        logger.info(f"Feature columns: {len([col for col in final_features.columns if col not in ['datetime', 'machineID', 'failure', 'failure_type']])}")
        
        return final_features
    
    def save_features(self, features_df: pd.DataFrame, filename: str = "features.csv"):
        """Save features to file.
        
        Args:
            features_df: Feature dataframe
            filename: Output filename
        """
        output_path = Path(self.config['data']['features_path'])
        output_path.mkdir(parents=True, exist_ok=True)
        
        filepath = output_path / filename
        features_df.to_csv(filepath, index=False)
        
        logger.info(f"Features saved to {filepath}")
    
    def load_features(self, filename: str = "features.csv") -> pd.DataFrame:
        """Load features from file.
        
        Args:
            filename: Input filename
            
        Returns:
            Feature dataframe
        """
        filepath = Path(self.config['data']['features_path']) / filename
        features_df = pd.read_csv(filepath)
        # Convert datetime column back to datetime type
        features_df['datetime'] = pd.to_datetime(features_df['datetime'])
        
        logger.info(f"Features loaded from {filepath}")
        return features_df 