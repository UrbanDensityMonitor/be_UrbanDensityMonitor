import re
from pathlib import Path
import pandas as pd
from src.config import ProjectConfig
from src.clustering.dbscan_clustering import DBSCANClusterer

def get_latest_temporal_features_csv(temporal_dir: Path) -> Path:
    csv_files = list(temporal_dir.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No reference data CSV found in '{temporal_dir}'. Please run the offline pipeline first.")
    return max(csv_files, key=lambda f: f.stat().st_mtime)

def parse_window_size_from_filename(filename: str) -> float:
    match = re.search(r"temporal_features_(\d+)s_", filename)
    if match:
        return float(match.group(1))
    return 30.0

def fit_dbscan_on_reference_csv(csv_path: Path, config: ProjectConfig) -> tuple[DBSCANClusterer, float]:
    df = pd.read_csv(csv_path)
    feature_columns = [
        'avg_vehicle_count', 'avg_density', 'avg_occupancy',
        'avg_speed', 'avg_flow', 'avg_congestion_index',
        'speed_variance', 'density_variance'
    ]
    feature_columns = [col for col in feature_columns if col in df.columns]
    X = df[feature_columns].values
    clusterer = DBSCANClusterer(config.clustering)
    clusterer.fit(X, feature_columns)
    window_size = parse_window_size_from_filename(csv_path.name)
    return clusterer, window_size
