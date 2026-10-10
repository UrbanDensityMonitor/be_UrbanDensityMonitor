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
        from .pipeline_exporters import PipelineVisualizer
        viz = PipelineVisualizer(self.config)
        return viz.generate_all(self.aggregated_df, self.cluster_result, self.feature_matrix, self.grid_searcher, self.evaluator)

    def generate_paper_tables(self) -> Dict[str, Path]:
        from .pipeline_exporters import PipelineReporter
        reporter = PipelineReporter(self.config, self.clusterer, self.evaluator)
        return reporter.generate_paper_tables(self.aggregated_df)

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
