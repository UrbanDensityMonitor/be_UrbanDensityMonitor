import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional, Tuple, Union
from pathlib import Path

from ..logger import logger


class DataValidator:
    """
    Validates input data for the traffic state discovery pipeline.
    Ensures data quality before processing.
    """
    
    def __init__(self) -> None:
        """Initialize data validator."""
        self.validation_results: Dict[str, Any] = {}
        
        logger.debug("DataValidator initialized")
    
    def validate_video_file(self, video_path: Union[str, Path]) -> Tuple[bool, str]:
        """
        Validate video file existence and format.
        
        Args:
            video_path: Path to video file
            
        Returns:
            Tuple of (is_valid, message)
        """
        video_path = Path(video_path)
        
        if not video_path.exists():
            return False, f"Video file not found: {video_path}"
        
        if not video_path.is_file():
            return False, f"Path is not a file: {video_path}"
        
        # Check file extension
        valid_extensions = {'.mp4', '.avi', '.mov', '.mkv', '.webm'}
        
        if video_path.suffix.lower() not in valid_extensions:
            return False, f"Unsupported video format: {video_path.suffix}. "
            f"Supported: {valid_extensions}"
        
        # Check file size
        file_size_mb = video_path.stat().st_size / (1024 * 1024)
        
        if file_size_mb == 0:
            return False, "Video file is empty"
        
        if file_size_mb > 10000:  # 10 GB
            logger.warning(f"Large video file: {file_size_mb:.1f} MB")
        
        logger.info(f"Video file valid: {video_path} ({file_size_mb:.1f} MB)")
        
        return True, "Valid video file"
    
    def validate_features_dataframe(self, df: pd.DataFrame) -> Tuple[bool, List[str]]:
        """
        Validate features DataFrame structure and content.
        
        Args:
            df: Features DataFrame
            
        Returns:
            Tuple of (is_valid, list_of_issues)
        """
        issues = []
        
        # Check if empty
        if df.empty:
            issues.append("DataFrame is empty")
            return False, issues
        
        # Check required columns (flexible)
        expected_columns = ['vehicle_count', 'average_speed', 'vehicle_density', 'road_occupancy']
        
        found_columns = []
        for col in expected_columns:
            if col in df.columns or any(col in c.lower() for c in df.columns):
                found_columns.append(col)
            else:
                issues.append(f"Missing expected column: {col}")
        
        if len(found_columns) < 2:
            issues.append("Insufficient feature columns found")
        
        # Check for NaN values
        nan_counts = dict(df.isna().sum())
        columns_with_nan = {col: count for col, count in nan_counts.items() if count > 0}
        
        if len(columns_with_nan) > 0:
            for col, count in columns_with_nan.items():
                if count > len(df) * 0.5:
                    issues.append(f"Column '{col}' has {count} NaN values (>50%)")
                else:
                    logger.warning(f"Column '{col}' has {count} NaN values")
        
        # Check for infinite values
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            if np.any(np.isinf(df[col].dropna())):
                issues.append(f"Column '{col}' contains infinite values")
        
        # Check value ranges
        if 'average_speed' in df.columns:
            speed = df['average_speed'].dropna()
            
            if len(speed) > 0:
                if speed.max() > 300:
                    issues.append(f"Unrealistic speed values detected (max: {speed.max():.1f})")
                if speed.min() < 0:
                    issues.append(f"Negative speed values detected (min: {speed.min():.1f})")
        
        if 'vehicle_density' in df.columns:
            density = df['vehicle_density'].dropna()
            
            if len(density) > 0 and density.max() > 500:
                issues.append(f"Unrealistic density values detected (max: {density.max():.1f})")
        
        if 'road_occupancy' in df.columns:
            occupancy = df['road_occupancy'].dropna()
            
            if len(occupancy) > 0:
                if occupancy.max() > 100:
                    issues.append(f"Occupancy exceeds 100% (max: {occupancy.max():.1f})")
                if occupancy.min() < 0:
                    issues.append(f"Negative occupancy values detected")
        
        is_valid = len(issues) == 0
        
        if is_valid:
            logger.info(f"Features DataFrame valid: {len(df)} rows, {len(df.columns)} columns")
        else:
            logger.warning(f"Features DataFrame has {len(issues)} issues")
        
        return is_valid, issues
    
    def validate_clustering_input(self, 
                                 X: np.ndarray,
                                 min_samples: int = 10) -> Tuple[bool, str]:
        """
        Validate input for clustering.
        
        Args:
            X: Feature matrix
            min_samples: Minimum required samples
            
        Returns:
            Tuple of (is_valid, message)
        """
        if X.size == 0:
            return False, "Empty feature matrix"
        
        if len(X) < min_samples:
            return False, f"Insufficient samples: {len(X)} (minimum: {min_samples})"
        
        if np.any(np.isnan(X)):
            return False, "Feature matrix contains NaN values"
        
        if np.any(np.isinf(X)):
            return False, "Feature matrix contains infinite values"
        
        # Check for zero variance features
        variances = np.var(X, axis=0)
        zero_var_features = np.where(variances == 0)[0]
        
        if len(zero_var_features) > 0:
            logger.warning(f"Features with zero variance: {zero_var_features}")
        
        logger.info(f"Clustering input valid: {X.shape[0]} samples, {X.shape[1]} features")
        
        return True, "Valid clustering input"
    
    def validate_roi_points(self, 
                           points: List[Tuple[float, float]],
                           image_shape: Tuple[int, int]) -> Tuple[bool, str]:
        """
        Validate ROI polygon points.
        
        Args:
            points: List of (x, y) points
            image_shape: (height, width) of image
            
        Returns:
            Tuple of (is_valid, message)
        """
        if len(points) < 3:
            return False, "Polygon must have at least 3 points"
        
        height, width = image_shape
        
        # Check boundaries
        for x, y in points:
            if x < 0 or x > width:
                return False, f"Point ({x}, {y}) is outside image width ({width})"
            if y < 0 or y > height:
                return False, f"Point ({x}, {y}) is outside image height ({height})"
        
        # Check for self-intersection
        from shapely.geometry import Polygon
        
        try:
            polygon = Polygon(points)
            
            if not polygon.is_valid:
                return False, f"Invalid polygon: {polygon.is_valid}"
            
            if polygon.is_empty:
                return False, "Polygon is empty"
            
            if polygon.area < 100:  # Minimum area in pixels
                return False, f"Polygon area too small: {polygon.area:.1f} pixels²"
        
        except Exception as e:
            return False, f"Failed to create polygon: {str(e)}"
        
        logger.info(f"ROI valid: {len(points)} points, area={polygon.area:.1f} pixels²")
        
        return True, "Valid ROI polygon"
    
    def validate_configuration(self, config_dict: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """
        Validate configuration dictionary.
        
        Args:
            config_dict: Configuration dictionary
            
        Returns:
            Tuple of (is_valid, list_of_issues)
        """
        issues = []
        
        # Check required sections
        required_sections = ['detection', 'tracking', 'features', 'clustering']
        
        for section in required_sections:
            if section not in config_dict:
                issues.append(f"Missing configuration section: {section}")
        
        # Validate detection config
        if 'detection' in config_dict:
            det = config_dict['detection']
            
            if det.get('confidence_threshold', -1) < 0 or det.get('confidence_threshold', 2) > 1:
                issues.append("confidence_threshold must be between 0 and 1")
            
            if det.get('iou_threshold', -1) < 0 or det.get('iou_threshold', 2) > 1:
                issues.append("iou_threshold must be between 0 and 1")
        
        # Validate clustering config
        if 'clustering' in config_dict:
            clust = config_dict['clustering']
            
            if clust.get('eps', -1) <= 0:
                issues.append("eps must be positive")
            
            if clust.get('min_samples', 0) < 1:
                issues.append("min_samples must be at least 1")
        
        is_valid = len(issues) == 0
        
        if is_valid:
            logger.info("Configuration is valid")
        else:
            logger.warning(f"Configuration has {len(issues)} issues")
        
        return is_valid, issues
    
    def generate_validation_report(self) -> Dict[str, Any]:
        """Generate validation report."""
        return self.validation_results