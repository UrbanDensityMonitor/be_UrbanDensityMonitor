"""
Main pipeline orchestrator for Traffic State Discovery.
Integrates all modules into complete workflow.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple
from datetime import datetime
import json
from tqdm import tqdm

from .logger import logger
from .config import ProjectConfig, DetectionConfig, TrackingConfig
from .utils.file_handler import FileHandler
from .utils.video_reader import VideoReader
from .detection.model_loader import ModelLoader
from .detection.detector import VehicleDetector
from .tracking.byte_tracker import ByteTrackTracker
from .roi.roi_manager import ROIManager
from .roi.roi_selector import ROISelector
from .features.feature_extractor import FeatureExtractor
from .temporal.temporal_aggregator import TemporalAggregator
from .clustering.dbscan_clustering import DBSCANClusterer
from .clustering.grid_search import GridSearchOptimizer
from .clustering.cluster_evaluator import ClusterEvaluator
from .visualization.traffic_visualizer import TrafficVisualizer
from .visualization.cluster_visualizer import ClusterVisualizer
from .visualization.heatmap_generator import HeatmapGenerator
from .visualization.video_renderer import VideoRenderer
from .visualization.paper_plotter import PaperPlotter


class TrafficStatePipeline:
    """
    Complete pipeline for unsupervised traffic state discovery.
    """
    
    def __init__(self, config: ProjectConfig) -> None:
        """
        Initialize complete pipeline.
        
        Args:
            config: Project configuration
        """
        self.config = config
        self.file_handler = FileHandler(config.paths.outputs)
        
        # Initialize components
        self.detector: Optional[VehicleDetector] = None
        self.tracker: Optional[ByteTrackTracker] = None
        self.roi_manager = ROIManager(config.roi.roi_path)
        self.feature_extractor: Optional[FeatureExtractor] = None
        self.temporal_aggregator = TemporalAggregator(config.temporal)
        self.clusterer = DBSCANClusterer(config.clustering)
        self.grid_searcher = GridSearchOptimizer(config.clustering)
        self.evaluator = ClusterEvaluator()
        
        # Visualizers
        self.traffic_viz = TrafficVisualizer(config.visualization)
        self.cluster_viz = ClusterVisualizer(config.visualization)
        self.heatmap_gen = HeatmapGenerator(config.visualization)
        self.video_renderer = VideoRenderer(config.visualization)
        self.paper_plotter = PaperPlotter(config.visualization)
        
        # Pipeline state
        self.is_initialized = False
        self.feature_matrix: Optional[np.ndarray] = None
        self.cluster_result: Optional[Any] = None
        self.aggregated_df: Optional[pd.DataFrame] = None
        
        logger.info("Traffic State Discovery Pipeline initialized")
    
    def initialize_models(self) -> None:
        """Initialize detection and tracking models."""
        logger.info("Initializing models...")
        
        # Initialize detector
        self.detector = VehicleDetector(self.config.detection)
        
        # Initialize tracker
        self.tracker = ByteTrackTracker(self.config.tracking)
        
        # Initialize feature extractor with ROI manager
        self.feature_extractor = FeatureExtractor(
            self.config.features,
            self.roi_manager
        )
        
        self.is_initialized = True
        logger.info("All models initialized successfully")
    
    def select_roi_interactive(self, video_path: Path, frame_number: int = 0) -> None:
        """
        Run interactive ROI selection tool.
        
        Args:
            video_path: Path to video file
            frame_number: Frame to use for selection
        """
        logger.info("Starting interactive ROI selection...")
        
        selector = ROISelector()
        rois = selector.select_roi_from_video(video_path, frame_number)
        
        if rois:
            # Add ROIs to manager
            for i, roi_points in enumerate(rois):
                self.roi_manager.add_roi(f"ROI_{i+1}", roi_points)
            
            # Save to config path
            self.roi_manager.save_roi(self.config.roi.roi_path)
            logger.info(f"Saved {len(rois)} ROIs to {self.config.roi.roi_path}")
        else:
            logger.warning("No ROIs selected, using full frame")
    
    def run_detection_tracking(self, 
                              video_path: Path,
                              start_frame: int = 0,
                              max_frames: Optional[int] = None,
                              save_video: bool = True) -> Tuple[pd.DataFrame, Path]:
        """
        Run detection and tracking on video.
        
        Args:
            video_path: Path to input video
            start_frame: Starting frame
            max_frames: Maximum frames to process
            save_video: Whether to save annotated video
            
        Returns:
            Tuple of (features DataFrame, output video path)
        """
        if not self.is_initialized:
            self.initialize_models()
        
        logger.info(f"Processing video: {video_path}")
        
        video_reader = VideoReader(video_path)
        
        if max_frames is None:
            max_frames = video_reader.metadata.total_frames
        
        # Tracking data storage
        all_features = []
        frame_positions = []
        frame_speeds = []
        
        # Output video setup
        output_video_path = None
        if save_video:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_video_path = (
                self.config.paths.outputs / "videos" / "tracking" / 
                f"tracking_output_{timestamp}.mp4"
            )
            output_video_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Process frames
        with tqdm(total=max_frames, desc="Processing frames", unit="frame") as pbar:
            for frame_number, frame in video_reader.stream_frames(
                start_frame, start_frame + max_frames
            ):
                # Detection
                detections = self.detector.detect(frame, frame_number)
                
                # Filter by ROI
                roi_detections = [
                    d for d in detections 
                    if self.roi_manager.is_point_inside(d.center)
                ]
                
                # Tracking
                track_result = self.tracker.update(roi_detections, frame, frame_number)
                
                # Feature extraction
                timestamp = frame_number / video_reader.metadata.fps
                features = self.feature_extractor.extract_frame_features(
                    track_result.tracks, frame_number, timestamp
                )
                all_features.append(features)
                
                # Store positions for heatmap
                for track in track_result.tracks:
                    if track.positions:
                        frame_positions.append(track.positions[-1])
                        if track.velocities:
                            frame_speeds.append(track.velocities[-1])
                
                # Update temporal aggregator
                self.temporal_aggregator.add_frame_features(features, timestamp)
                
                # Save annotated frame to video
                if save_video and output_video_path:
                    if frame_number == start_frame:
                        writer = video_reader.get_video_writer(output_video_path)
                    
                    annotated = self._create_annotated_frame(
                        frame, track_result.tracks, features, frame_number,
                        video_reader.metadata.fps
                    )
                    writer.write(annotated)
                
                pbar.update(1)
                
                # Progress logging
                if frame_number % 100 == 0:
                    stats = self.tracker.get_statistics()
                    logger.debug(
                        f"Frame {frame_number}: {features.vehicle_count} vehicles, "
                        f"{stats['active_tracks']} active tracks"
                    )
        
        # Release video writer
        if save_video and output_video_path:
            writer.release()
        
        # Save frame positions and speeds for heatmap generation
        if frame_positions:
            self._save_tracking_data(frame_positions, frame_speeds, video_reader.metadata)
        
        video_reader.__del__()
        
        logger.info("Detection and tracking complete")
        
        # Return features DataFrame
        features_df = self.feature_extractor.get_feature_dataframe()
        return features_df, output_video_path
    
    def _create_annotated_frame(self,
                               frame: np.ndarray,
                               tracks: List[Any],
                               features: Any,
                               frame_number: int,
                               fps: float) -> np.ndarray:
        """Create annotated frame with all overlays."""
        # Draw tracks
        annotated = self.video_renderer.draw_tracks(frame, tracks, show_trail=True)
        
        # Draw ROI
        annotated = self.video_renderer.draw_roi_overlay(annotated, self.roi_manager)
        
        # Prepare info overlay
        info = {
            'Frame': frame_number,
            'FPS': f"{fps:.1f}",
            'Vehicles': features.vehicle_count,
            'Density': f"{features.vehicle_density:.1f}",
            'Speed': f"{features.average_speed:.1f} km/h",
            'Occupancy': f"{features.road_occupancy:.1f}%",
            'Congestion': f"{features.congestion_index:.2f}"
        }
        
        # Draw info
        annotated = self.video_renderer.draw_info_overlay(annotated, info)
        
        return annotated
    
    def _save_tracking_data(self, 
                           positions: List[Tuple[float, float]],
                           speeds: List[float],
                           metadata: Any) -> None:
        """Save tracking data for later use."""
        data = {
            'positions': [(float(x), float(y)) for x, y in positions],
            'speeds': [float(s) for s in speeds],
            'frame_width': metadata.width,
            'frame_height': metadata.height
        }
        
        save_path = self.config.paths.outputs / "csv" / "tracking_positions.json"
        with open(save_path, 'w') as f:
            json.dump(data, f)
        
        logger.info(f"Tracking data saved to {save_path}")
    
    def run_temporal_aggregation(self) -> pd.DataFrame:
        """
        Run temporal aggregation on extracted features.
        
        Returns:
            DataFrame with aggregated features
        """
        logger.info("Running temporal aggregation...")
        
        # Process all windows
        self.temporal_aggregator.process_all_windows()
        
        # Get aggregated DataFrame
        self.aggregated_df = self.temporal_aggregator.get_aggregated_dataframe()
        
        if self.aggregated_df.empty:
            logger.warning("No aggregated features generated")
            return self.aggregated_df
        
        # Save to CSV
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.file_handler.save_csv(
            self.aggregated_df,
            f"temporal_features_{self.config.temporal.default_window}s",
            "csv/temporal_features"
        )
        
        logger.info(f"Temporal aggregation complete: {len(self.aggregated_df)} windows")
        
        return self.aggregated_df
    
    def run_clustering(self, 
                      run_grid_search: bool = True) -> Tuple[np.ndarray, Any, Dict[str, Any]]:
        """
        Run DBSCAN clustering on aggregated features.
        
        Args:
            run_grid_search: Whether to run grid search for parameters
            
        Returns:
            Tuple of (feature matrix, cluster result, evaluation report)
        """
        if self.aggregated_df is None or self.aggregated_df.empty:
            raise ValueError("No aggregated data available. Run temporal aggregation first.")
        
        logger.info("Running clustering analysis...")
        
        # Get feature matrix
        X, feature_names = self.temporal_aggregator.get_feature_matrix(
            self.config.temporal.default_window,
            normalize=True
        )
        
        if X.size == 0:
            raise ValueError("Empty feature matrix")
        
        self.feature_matrix = X
        
        # Run grid search if requested
        if run_grid_search and self.config.clustering.grid_search:
            logger.info("Performing grid search for optimal parameters...")
            grid_result = self.grid_searcher.search(X, feature_names)
            
            # Use best parameters
            best_params = grid_result.best_params
            self.config.clustering.eps = best_params['eps']
            self.config.clustering.min_samples = best_params['min_samples']
            
            # Save grid search results
            grid_df = self.grid_searcher.get_search_dataframe()
            self.file_handler.save_csv(
                grid_df,
                "grid_search_results",
                "csv/clusters"
            )
        
        # Run clustering with best parameters
        self.cluster_result = self.clusterer.fit(
            X, 
            feature_names,
            eps=self.config.clustering.eps,
            min_samples=self.config.clustering.min_samples
        )
        
        # Evaluate clustering
        evaluation_report = self.evaluator.evaluate(X, self.cluster_result)
        
        # Save cluster labels
        labels_df = self.clusterer.get_labels_dataframe(
            timestamps=self.aggregated_df['window_start'].tolist()
        )
        self.file_handler.save_csv(
            labels_df,
            "cluster_labels",
            "csv/clusters"
        )
        
        # Save cluster summary
        cluster_summary = self.clusterer.get_cluster_summary()
        self.file_handler.save_json(
            cluster_summary,
            "cluster_summary",
            "clusters/statistics"
        )
        
        # Save evaluation report
        eval_report = self.evaluator.export_report()
        self.file_handler.save_json(
            eval_report,
            "evaluation_report",
            "csv/evaluation"
        )
        
        logger.info(
            f"Clustering complete: {cluster_summary['n_clusters']} clusters, "
            f"silhouette={evaluation_report.silhouette_score:.3f}"
        )
        
        return X, self.cluster_result, eval_report
    
    def generate_visualizations(self) -> Dict[str, Path]:
        """
        Generate all visualizations.
        
        Returns:
            Dictionary of visualization paths
        """
        if self.aggregated_df is None or self.cluster_result is None:
            raise ValueError("Run full pipeline before generating visualizations")
        
        logger.info("Generating visualizations...")
        
        output_paths = {}
        
        # Traffic visualizations
        logger.info("Creating traffic metric plots...")
        
        # Vehicle count
        path = self.config.paths.outputs / "plots" / "vehicle_count.png"
        self.traffic_viz.plot_vehicle_count(
            self.aggregated_df, save_path=path
        )
        output_paths['vehicle_count'] = path
        
        # Density
        path = self.config.paths.outputs / "plots" / "density" / "density_over_time.png"
        self.traffic_viz.plot_density(
            self.aggregated_df, save_path=path
        )
        output_paths['density'] = path
        
        # Occupancy
        path = self.config.paths.outputs / "plots" / "occupancy" / "occupancy_over_time.png"
        self.traffic_viz.plot_occupancy(
            self.aggregated_df, save_path=path
        )
        output_paths['occupancy'] = path
        
        # Speed
        path = self.config.paths.outputs / "plots" / "speed" / "speed_over_time.png"
        self.traffic_viz.plot_speed(
            self.aggregated_df, save_path=path
        )
        output_paths['speed'] = path
        
        # Multi-feature dashboard
        path = self.config.paths.outputs / "plots" / "traffic_dashboard.png"
        self.traffic_viz.plot_multi_feature_dashboard(
            self.aggregated_df, save_path=path
        )
        output_paths['dashboard'] = path
        
        # Heatmap visualizations
        tracking_data_path = self.config.paths.outputs / "csv" / "tracking_positions.json"
        if tracking_data_path.exists():
            logger.info("Generating spatial heatmaps...")
            try:
                import cv2
                with open(tracking_data_path, 'r') as f:
                    tracking_data = json.load(f)
                
                positions = [(float(x), float(y)) for x, y in tracking_data.get('positions', [])]
                speeds = [float(s) for s in tracking_data.get('speeds', [])]
                width = tracking_data.get('frame_width', 1280)
                height = tracking_data.get('frame_height', 720)
                
                if positions:
                    # Generate density heatmap
                    density_heatmap = self.heatmap_gen.generate_density_heatmap(
                        positions, (height, width)
                    )
                    density_heatmap_path = self.config.paths.outputs / "plots" / "heatmaps" / "density_heatmap.png"
                    density_heatmap_path.parent.mkdir(parents=True, exist_ok=True)
                    cv2.imwrite(str(density_heatmap_path), density_heatmap)
                    output_paths['density_heatmap'] = density_heatmap_path
                    
                    if speeds:
                        # Generate speed heatmap
                        speed_heatmap = self.heatmap_gen.generate_speed_heatmap(
                            positions, speeds, (height, width)
                        )
                        speed_heatmap_path = self.config.paths.outputs / "plots" / "heatmaps" / "speed_heatmap.png"
                        cv2.imwrite(str(speed_heatmap_path), speed_heatmap)
                        output_paths['speed_heatmap'] = speed_heatmap_path
                        
                        # Generate congestion heatmap figure
                        max_speed = max(speeds) if speeds else 100.0
                        congestion_values = [max(0.0, 1.0 - (s / max_speed)) for s in speeds]
                        congestion_heatmap_path = self.config.paths.outputs / "plots" / "heatmaps" / "congestion_heatmap.png"
                        self.heatmap_gen.create_congestion_heatmap_figure(
                            positions[:len(speeds)], congestion_values, (height, width),
                            title='Traffic Congestion Heatmap',
                            save_path=congestion_heatmap_path
                        )
                        output_paths['congestion_heatmap'] = congestion_heatmap_path
            except Exception as e:
                logger.error(f"Error generating heatmaps: {e}", exc_info=True)
        
        # Cluster visualizations
        logger.info("Creating cluster plots...")
        
        # Scatter plot
        path = self.config.paths.outputs / "plots" / "clusters" / "cluster_scatter.png"
        self.cluster_viz.plot_cluster_scatter(
            self.feature_matrix, self.cluster_result, save_path=path
        )
        output_paths['cluster_scatter'] = path
        
        # State distribution
        path = self.config.paths.outputs / "plots" / "clusters" / "state_distribution.png"
        self.cluster_viz.plot_state_distribution(
            self.cluster_result, save_path=path
        )
        output_paths['state_distribution'] = path
        
        # Timeline
        path = self.config.paths.outputs / "plots" / "clusters" / "state_timeline.png"
        self.cluster_viz.plot_timeline_states(
            self.cluster_result.labels,
            self.cluster_result.state_mapping,
            self.aggregated_df['window_start'].tolist(),
            save_path=path
        )
        output_paths['state_timeline'] = path
        
        # Feature importance
        path = self.config.paths.outputs / "plots" / "clusters" / "feature_importance.png"
        self.cluster_viz.plot_feature_importance(
            self.feature_matrix, self.cluster_result, save_path=path
        )
        output_paths['feature_importance'] = path
        
        # Paper-ready figures
        logger.info("Generating paper figures...")
        
        # Pipeline overview
        path = self.config.paths.paper / "figures" / "pipeline_overview"
        self.paper_plotter.plot_pipeline_overview(path)
        output_paths['pipeline_overview'] = path
        
        # Cluster comparison
        path = self.config.paths.paper / "figures" / "cluster_comparison"
        self.paper_plotter.plot_cluster_comparison(
            self.feature_matrix, self.cluster_result, path
        )
        output_paths['cluster_comparison'] = path
        
        # Evaluation summary
        eval_data = self.evaluator.export_report()
        eval_data['grid_search'] = self.grid_searcher.get_search_dataframe().to_dict('records')
        path = self.config.paths.paper / "figures" / "evaluation_summary"
        self.paper_plotter.plot_evaluation_summary(eval_data, path)
        output_paths['evaluation_summary'] = path
        
        logger.info(f"All visualizations generated: {len(output_paths)} files")
        
        return output_paths
    
    def generate_paper_tables(self) -> Dict[str, Path]:
        """
        Generate LaTeX tables for paper.
        
        Returns:
            Dictionary of table paths
        """
        if self.cluster_result is None:
            raise ValueError("Run clustering first")
        
        logger.info("Generating paper tables...")
        
        table_paths = {}
        
        # Cluster statistics table
        cluster_stats = self.clusterer.get_cluster_summary()
        table_path = self.config.paths.paper / "tables" / "cluster_statistics.tex"
        self._generate_cluster_table(cluster_stats, table_path)
        table_paths['cluster_statistics'] = table_path
        
        # Evaluation metrics table
        eval_report = self.evaluator.export_report()
        table_path = self.config.paths.paper / "tables" / "evaluation_metrics.tex"
        self._generate_evaluation_table(eval_report, table_path)
        table_paths['evaluation_metrics'] = table_path
        
        # Dataset summary table
        if self.aggregated_df is not None:
            table_path = self.config.paths.paper / "tables" / "dataset_summary.tex"
            self._generate_dataset_table(self.aggregated_df, table_path)
            table_paths['dataset_summary'] = table_path
        
        logger.info(f"Paper tables generated: {len(table_paths)} tables")
        
        return table_paths
    
    def _generate_cluster_table(self, stats: Dict[str, Any], path: Path) -> None:
        """Generate LaTeX table for cluster statistics."""
        latex = []
        latex.append("\\begin{table}[h]")
        latex.append("\\centering")
        latex.append("\\caption{Traffic State Cluster Statistics}")
        latex.append("\\label{tab:cluster_stats}")
        latex.append("\\begin{tabular}{lrrr}")
        latex.append("\\hline")
        latex.append("Cluster & State & Size & Percentage \\\\")
        latex.append("\\hline")
        
        total = sum(stats.get('cluster_sizes', {}).values()) + stats.get('n_noise', 0)
        
        for label, size in stats.get('cluster_sizes', {}).items():
            state = stats.get('state_mapping', {}).get(label, f'Cluster {label}')
            percentage = (size / total * 100) if total > 0 else 0
            latex.append(f"{label} & {state} & {size} & {percentage:.1f}\\% \\\\")
        
        if stats.get('n_noise', 0) > 0:
            percentage = stats['n_noise'] / total * 100 if total > 0 else 0
            latex.append(f"Noise & - & {stats['n_noise']} & {percentage:.1f}\\% \\\\")
        
        latex.append("\\hline")
        latex.append(f"Total & - & {total} & 100.0\\% \\\\")
        latex.append("\\hline")
        latex.append("\\end{tabular}")
        latex.append("\\end{table}")
        
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w') as f:
            f.write('\n'.join(latex))
    
    def _generate_evaluation_table(self, eval_report: Dict[str, Any], path: Path) -> None:
        """Generate LaTeX table for evaluation metrics."""
        latex = []
        latex.append("\\begin{table}[h]")
        latex.append("\\centering")
        latex.append("\\caption{Clustering Evaluation Metrics}")
        latex.append("\\label{tab:evaluation}")
        latex.append("\\begin{tabular}{lr}")
        latex.append("\\hline")
        latex.append("Metric & Value \\\\")
        latex.append("\\hline")
        
        metric_names = {
            'silhouette_score': 'Silhouette Score',
            'davies_bouldin_index': 'Davies-Bouldin Index',
            'calinski_harabasz_index': 'Calinski-Harabasz Index',
            'noise_ratio': 'Noise Ratio',
            'cluster_separation': 'Cluster Separation',
            'stability_score': 'Stability Score'
        }
        
        for key, name in metric_names.items():
            value = eval_report.get(key, 'N/A')
            if isinstance(value, float):
                latex.append(f"{name} & {value:.4f} \\\\")
            else:
                latex.append(f"{name} & {value} \\\\")
        
        latex.append("\\hline")
        latex.append("\\end{tabular}")
        latex.append("\\end{table}")
        
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w') as f:
            f.write('\n'.join(latex))
    
    def _generate_dataset_table(self, df: pd.DataFrame, path: Path) -> None:
        """Generate LaTeX table for dataset summary."""
        latex = []
        latex.append("\\begin{table}[h]")
        latex.append("\\centering")
        latex.append("\\caption{Dataset Summary Statistics}")
        latex.append("\\label{tab:dataset}")
        latex.append("\\begin{tabular}{lrrrr}")
        latex.append("\\hline")
        latex.append("Feature & Mean & Std & Min & Max \\\\")
        latex.append("\\hline")
        
        feature_columns = [
            'avg_vehicle_count', 'avg_density', 'avg_occupancy',
            'avg_speed', 'avg_flow', 'avg_congestion_index'
        ]
        
        feature_names = {
            'avg_vehicle_count': 'Vehicle Count',
            'avg_density': 'Density (veh/km)',
            'avg_occupancy': 'Occupancy (%)',
            'avg_speed': 'Speed (km/h)',
            'avg_flow': 'Flow (veh/h)',
            'avg_congestion_index': 'Congestion Index'
        }
        
        for col in feature_columns:
            if col in df.columns:
                name = feature_names.get(col, col)
                mean = df[col].mean()
                std = df[col].std()
                min_val = df[col].min()
                max_val = df[col].max()
                latex.append(f"{name} & {mean:.2f} & {std:.2f} & {min_val:.2f} & {max_val:.2f} \\\\")
        
        latex.append("\\hline")
        latex.append("\\end{tabular}")
        latex.append("\\end{table}")
        
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w') as f:
            f.write('\n'.join(latex))
    
    def run_full_pipeline(self,
                         video_path: Path,
                         select_roi: bool = True,
                         max_frames: Optional[int] = None,
                         run_grid_search: bool = True) -> Dict[str, Any]:
        """
        Run complete pipeline from start to finish.
        
        Args:
            video_path: Path to input video
            select_roi: Whether to run interactive ROI selection
            max_frames: Maximum frames to process
            run_grid_search: Whether to run grid search
            
        Returns:
            Dictionary with all results
        """
        logger.info("=" * 60)
        logger.info("STARTING FULL PIPELINE")
        logger.info("=" * 60)
        
        start_time = datetime.now()
        
        results = {}
        
        # Step 1: ROI Selection
        if select_roi:
            logger.info("\n[STEP 1/6] ROI Selection")
            self.select_roi_interactive(video_path)
        else:
            logger.info("\n[STEP 1/6] Using existing ROI configuration")
        
        # Step 2: Detection & Tracking
        logger.info("\n[STEP 2/6] Vehicle Detection & Tracking")
        features_df, output_video = self.run_detection_tracking(
            video_path, 
            max_frames=max_frames,
            save_video=True
        )
        results['features_df'] = features_df
        results['output_video'] = output_video
        
        # Save raw features
        self.file_handler.save_csv(features_df, "raw_features", "csv/raw_features")
        
        # Step 3: Temporal Aggregation
        logger.info("\n[STEP 3/6] Temporal Aggregation")
        self.aggregated_df = self.run_temporal_aggregation()
        results['aggregated_df'] = self.aggregated_df
        
        # Step 4: Clustering
        logger.info("\n[STEP 4/6] Traffic State Clustering")
        feature_matrix, cluster_result, eval_report = self.run_clustering(run_grid_search)
        results['feature_matrix'] = feature_matrix
        results['cluster_result'] = cluster_result
        results['evaluation_report'] = eval_report
        
        # Step 5: Visualizations
        logger.info("\n[STEP 5/6] Generating Visualizations")
        visualization_paths = self.generate_visualizations()
        results['visualizations'] = visualization_paths
        
        # Step 6: Paper Tables
        logger.info("\n[STEP 6/6] Generating Paper Tables")
        table_paths = self.generate_paper_tables()
        results['tables'] = table_paths
        
        # Final summary
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        logger.info("\n" + "=" * 60)
        logger.info("PIPELINE COMPLETE")
        logger.info(f"Total duration: {duration:.2f} seconds")
        logger.info(f"Outputs saved to: {self.config.paths.outputs}")
        logger.info("=" * 60)
        
        # Save pipeline summary
        summary = {
            'video_path': str(video_path),
            'frames_processed': len(features_df),
            'temporal_windows': len(self.aggregated_df),
            'n_clusters': cluster_result.n_clusters if cluster_result else 0,
            'silhouette_score': eval_report.get('silhouette_score', 0),
            'duration_seconds': duration,
            'timestamp': start_time.isoformat()
        }
        
        self.file_handler.save_json(summary, "pipeline_summary", "logs/pipeline_runs")
        
        return results