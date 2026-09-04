"""
File handling utilities for managing outputs, CSVs, and configurations.
"""

import json
import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import pandas as pd
import yaml
from datetime import datetime
import shutil

from ..logger import logger


class FileHandler:
    """
    Handles all file I/O operations for the project.
    """
    
    def __init__(self, base_path: Union[str, Path] = "outputs") -> None:
        """
        Initialize FileHandler with base output path.
        
        Args:
            base_path: Base directory for all outputs
        """
        self.base_path = Path(base_path)
        self._create_directory_structure()
    
    def _create_directory_structure(self) -> None:
        """Create standard output directory structure."""
        directories = [
            "csv/raw_features",
            "csv/temporal_features",
            "csv/clusters",
            "csv/evaluation",
            "plots/density",
            "plots/occupancy",
            "plots/speed",
            "plots/flow",
            "plots/clusters",
            "plots/heatmaps",
            "clusters/assignments",
            "clusters/statistics",
            "videos/tracking",
            "videos/roi",
            "videos/traffic_state",
            "logs/pipeline_runs"
        ]
        
        for directory in directories:
            dir_path = self.base_path / directory
            dir_path.mkdir(parents=True, exist_ok=True)
            logger.debug(f"Created directory: {dir_path}")
    
    def save_csv(self, 
                 data: Union[pd.DataFrame, Dict[str, Any]], 
                 filename: str, 
                 subfolder: str = "csv") -> Path:
        """
        Save data to CSV file.
        
        Args:
            data: DataFrame or dictionary to save
            filename: Name of the file (without extension)
            subfolder: Subfolder within base_path
            
        Returns:
            Path to saved file
        """
        if isinstance(data, dict):
            data = pd.DataFrame([data])
        elif not isinstance(data, pd.DataFrame):
            data = pd.DataFrame(data)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filepath = self.base_path / subfolder / f"{filename}_{timestamp}.csv"
        
        data.to_csv(filepath, index=False)
        logger.info(f"CSV saved to: {filepath}")
        return filepath
    
    def load_csv(self, filepath: Union[str, Path]) -> pd.DataFrame:
        """
        Load CSV file into DataFrame.
        
        Args:
            filepath: Path to CSV file
            
        Returns:
            DataFrame with loaded data
        """
        return pd.read_csv(filepath)
    
    def save_json(self, 
                  data: Dict[str, Any], 
                  filename: str, 
                  subfolder: str = "clusters") -> Path:
        """
        Save data to JSON file.
        
        Args:
            data: Dictionary to save
            filename: Name of the file (without extension)
            subfolder: Subfolder within base_path
            
        Returns:
            Path to saved file
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filepath = self.base_path / subfolder / f"{filename}_{timestamp}.json"
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        
        logger.info(f"JSON saved to: {filepath}")
        return filepath
    
    def load_json(self, filepath: Union[str, Path]) -> Dict[str, Any]:
        """
        Load JSON file.
        
        Args:
            filepath: Path to JSON file
            
        Returns:
            Dictionary with loaded data
        """
        with open(filepath, 'r') as f:
            return json.load(f)
    
    def save_pickle(self, 
                    data: Any, 
                    filename: str, 
                    subfolder: str = "clusters") -> Path:
        """
        Save Python object to pickle file.
        
        Args:
            data: Python object to serialize
            filename: Name of the file (without extension)
            subfolder: Subfolder within base_path
            
        Returns:
            Path to saved file
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filepath = self.base_path / subfolder / f"{filename}_{timestamp}.pkl"
        
        with open(filepath, 'wb') as f:
            pickle.dump(data, f)
        
        logger.info(f"Pickle saved to: {filepath}")
        return filepath
    
    def load_pickle(self, filepath: Union[str, Path]) -> Any:
        """
        Load pickle file.
        
        Args:
            filepath: Path to pickle file
            
        Returns:
            Deserialized Python object
        """
        with open(filepath, 'rb') as f:
            return pickle.load(f)
    
    def save_yaml(self, 
                  data: Dict[str, Any], 
                  filepath: Union[str, Path]) -> Path:
        """
        Save configuration to YAML file.
        
        Args:
            data: Configuration dictionary
            filepath: Path for YAML file
            
        Returns:
            Path to saved file
        """
        filepath = Path(filepath)
        with open(filepath, 'w') as f:
            yaml.dump(data, f, default_flow_style=False)
        
        logger.info(f"YAML saved to: {filepath}")
        return filepath
    
    def ensure_directory(self, path: Union[str, Path]) -> Path:
        """
        Ensure directory exists, create if not.
        
        Args:
            path: Directory path
            
        Returns:
            Path object for directory
        """
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        return path
    
    def get_latest_file(self, directory: Union[str, Path], pattern: str = "*") -> Optional[Path]:
        """
        Get the most recently modified file in directory.
        
        Args:
            directory: Directory to search
            pattern: Glob pattern to match files
            
        Returns:
            Path to latest file or None if empty
        """
        directory = Path(directory)
        files = list(directory.glob(pattern))
        
        if not files:
            return None
        
        return max(files, key=lambda p: p.stat().st_mtime)
    
    def backup_file(self, filepath: Union[str, Path]) -> Path:
        """
        Create a backup of a file.
        
        Args:
            filepath: Path to file to backup
            
        Returns:
            Path to backup file
        """
        filepath = Path(filepath)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_path = filepath.parent / f"{filepath.stem}_backup_{timestamp}{filepath.suffix}"
        
        shutil.copy2(filepath, backup_path)
        logger.info(f"Backup created: {backup_path}")
        return backup_path
    
    def clean_outputs(self, keep_latest: int = 5) -> None:
        """
        Clean old output files, keeping only the latest N files.
        
        Args:
            keep_latest: Number of latest files to keep
        """
        for subdir in self.base_path.rglob("*"):
            if subdir.is_dir():
                files = sorted(
                    subdir.glob("*"),
                    key=lambda p: p.stat().st_mtime,
                    reverse=True
                )
                
                for old_file in files[keep_latest:]:
                    if old_file.is_file():
                        old_file.unlink()
                        logger.debug(f"Removed old file: {old_file}")