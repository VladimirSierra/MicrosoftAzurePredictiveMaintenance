"""
Data loading and basic preprocessing module for Predictive Maintenance.
"""

import pandas as pd
from pathlib import Path
import logging
from typing import Dict
import yaml

logger = logging.getLogger(__name__)


class PredictiveMaintenanceDataLoader:
    """Data loader for the Predictive Maintenance dataset."""
    
    def __init__(self, config_path: str = "config/config.yaml"):
        """Initialize the data loader with configuration.
        
        Args:
            config_path: Path to the configuration file
        """
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.data_path = Path(self.config['data']['raw_data_path'])
        
    def load_telemetry(self) -> pd.DataFrame:
        """Load and preprocess telemetry data.
        
        Returns:
            DataFrame with telemetry data
        """
        logger.info("Loading telemetry data...")
        file_path = self.data_path / self.config['data']['telemetry_file']
        
        df = pd.read_csv(file_path)
        df['datetime'] = pd.to_datetime(df['datetime'])
        
        # Sort by machine and datetime for time series processing
        df = df.sort_values(['machineID', 'datetime']).reset_index(drop=True)
        
        logger.info(f"Loaded telemetry data: {len(df)} records, "
                   f"{df['machineID'].nunique()} machines")
        
        return df
    
    def load_errors(self) -> pd.DataFrame:
        """Load and preprocess error data.
        
        Returns:
            DataFrame with error data
        """
        logger.info("Loading error data...")
        file_path = self.data_path / self.config['data']['errors_file']
        
        df = pd.read_csv(file_path)
        df['datetime'] = pd.to_datetime(df['datetime'])
        df = df.sort_values(['machineID', 'datetime']).reset_index(drop=True)
        
        logger.info(f"Loaded error data: {len(df)} records")
        
        return df
    
    def load_failures(self) -> pd.DataFrame:
        """Load and preprocess failure data.
        
        Returns:
            DataFrame with failure data
        """
        logger.info("Loading failure data...")
        file_path = self.data_path / self.config['data']['failures_file']
        
        df = pd.read_csv(file_path)
        df['datetime'] = pd.to_datetime(df['datetime'])
        df = df.sort_values(['machineID', 'datetime']).reset_index(drop=True)
        
        logger.info(f"Loaded failure data: {len(df)} records")
        
        return df
    
    def load_maintenance(self) -> pd.DataFrame:
        """Load and preprocess maintenance data.
        
        Returns:
            DataFrame with maintenance data
        """
        logger.info("Loading maintenance data...")
        file_path = self.data_path / self.config['data']['maintenance_file']
        
        df = pd.read_csv(file_path)
        df['datetime'] = pd.to_datetime(df['datetime'])
        df = df.sort_values(['machineID', 'datetime']).reset_index(drop=True)
        
        logger.info(f"Loaded maintenance data: {len(df)} records")
        
        return df
    
    def load_machines(self) -> pd.DataFrame:
        """Load machine metadata.
        
        Returns:
            DataFrame with machine metadata
        """
        logger.info("Loading machine data...")
        file_path = self.data_path / self.config['data']['machines_file']
        
        df = pd.read_csv(file_path)
        
        logger.info(f"Loaded machine data: {len(df)} machines")
        
        return df
    
    def load_all_data(self) -> Dict[str, pd.DataFrame]:
        """Load all datasets.
        
        Returns:
            Dictionary containing all datasets
        """
        return {
            'telemetry': self.load_telemetry(),
            'errors': self.load_errors(),
            'failures': self.load_failures(),
            'maintenance': self.load_maintenance(),
            'machines': self.load_machines()
        }
    
    def get_data_summary(self) -> Dict:
        """Get summary statistics of the datasets.
        
        Returns:
            Dictionary with summary statistics
        """
        data = self.load_all_data()
        
        summary = {}
        for name, df in data.items():
            summary[name] = {
                'rows': len(df),
                'columns': df.columns.tolist(),
                'date_range': None
            }
            
            if 'datetime' in df.columns:
                summary[name]['date_range'] = {
                    'start': df['datetime'].min().strftime('%Y-%m-%d'),
                    'end': df['datetime'].max().strftime('%Y-%m-%d')
                }
            
            if 'machineID' in df.columns:
                summary[name]['machines'] = df['machineID'].nunique()
        
        return summary 