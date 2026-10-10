import json
from pathlib import Path
from typing import Dict, Any
import pandas as pd
from datetime import datetime

from .logger import logger
from .visualization.traffic_visualizer import TrafficVisualizer
from .visualization.cluster_visualizer import ClusterVisualizer
from .visualization.heatmap_generator import HeatmapGenerator
from .visualization.paper_plotter import PaperPlotter

class PipelineVisualizer:
    def __init__(self, config):
        self.config = config
        self.traffic_viz = TrafficVisualizer(config.visualization)
        self.cluster_viz = ClusterVisualizer(config.visualization)
        self.heatmap_gen = HeatmapGenerator(config.visualization)
        self.paper_plotter = PaperPlotter(config.visualization)

    def generate_all(self, aggregated_df, cluster_result, feature_matrix, grid_searcher, evaluator) -> Dict[str, Path]:
        logger.info("Generating visualizations...")
        output_paths = {}

        # Traffic visualizations
        path = self.config.paths.outputs / "plots" / "vehicle_count.png"
        self.traffic_viz.plot_vehicle_count(aggregated_df, save_path=path)
        output_paths['vehicle_count'] = path

        path = self.config.paths.outputs / "plots" / "density" / "density_over_time.png"
        self.traffic_viz.plot_density(aggregated_df, save_path=path)
        output_paths['density'] = path

        path = self.config.paths.outputs / "plots" / "occupancy" / "occupancy_over_time.png"
        self.traffic_viz.plot_occupancy(aggregated_df, save_path=path)
        output_paths['occupancy'] = path

        path = self.config.paths.outputs / "plots" / "speed" / "speed_over_time.png"
        self.traffic_viz.plot_speed(aggregated_df, save_path=path)
        output_paths['speed'] = path

        path = self.config.paths.outputs / "plots" / "traffic_dashboard.png"
        self.traffic_viz.plot_multi_feature_dashboard(aggregated_df, save_path=path)
        output_paths['dashboard'] = path

        # Heatmaps
        tracking_data_path = self.config.paths.outputs / "csv" / "tracking_positions.json"
        if tracking_data_path.exists():
            try:
                import cv2
                with open(tracking_data_path, 'r') as f:
                    tracking_data = json.load(f)
                
                positions = [(float(x), float(y)) for x, y in tracking_data.get('positions', [])]
                speeds = [float(s) for s in tracking_data.get('speeds', [])]
                width = tracking_data.get('frame_width', 1280)
                height = tracking_data.get('frame_height', 720)
                
                if positions:
                    density_heatmap = self.heatmap_gen.generate_density_heatmap(positions, (height, width))
                    density_heatmap_path = self.config.paths.outputs / "plots" / "heatmaps" / "density_heatmap.png"
                    density_heatmap_path.parent.mkdir(parents=True, exist_ok=True)
                    cv2.imwrite(str(density_heatmap_path), density_heatmap)
                    output_paths['density_heatmap'] = density_heatmap_path
                    
                    if speeds:
                        speed_heatmap = self.heatmap_gen.generate_speed_heatmap(positions, speeds, (height, width))
                        speed_heatmap_path = self.config.paths.outputs / "plots" / "heatmaps" / "speed_heatmap.png"
                        cv2.imwrite(str(speed_heatmap_path), speed_heatmap)
                        output_paths['speed_heatmap'] = speed_heatmap_path
                        
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

        # Cluster visualisasi
        path = self.config.paths.outputs / "plots" / "clusters" / "cluster_scatter.png"
        self.cluster_viz.plot_cluster_scatter(feature_matrix, cluster_result, save_path=path)
        output_paths['cluster_scatter'] = path

        path = self.config.paths.outputs / "plots" / "clusters" / "state_distribution.png"
        self.cluster_viz.plot_state_distribution(cluster_result, save_path=path)
        output_paths['state_distribution'] = path

        path = self.config.paths.outputs / "plots" / "clusters" / "state_timeline.png"
        self.cluster_viz.plot_timeline_states(
            cluster_result.labels, cluster_result.state_mapping,
            aggregated_df['window_start'].tolist(), save_path=path
        )
        output_paths['state_timeline'] = path

        path = self.config.paths.outputs / "plots" / "clusters" / "feature_importance.png"
        self.cluster_viz.plot_feature_importance(feature_matrix, cluster_result, save_path=path)
        output_paths['feature_importance'] = path

        # Paper-ready figures
        path = self.config.paths.paper / "figures" / "pipeline_overview"
        self.paper_plotter.plot_pipeline_overview(path)
        output_paths['pipeline_overview'] = path

        path = self.config.paths.paper / "figures" / "cluster_comparison"
        self.paper_plotter.plot_cluster_comparison(feature_matrix, cluster_result, path)
        output_paths['cluster_comparison'] = path

        eval_data = evaluator.export_report()
        eval_data['grid_search'] = grid_searcher.get_search_dataframe().to_dict('records')
        path = self.config.paths.paper / "figures" / "evaluation_summary"
        self.paper_plotter.plot_evaluation_summary(eval_data, path)
        output_paths['evaluation_summary'] = path

        logger.info(f"All visualizations generated: {len(output_paths)} files")
        return output_paths

class PipelineReporter:
    def __init__(self, config, clusterer, evaluator):
        self.config = config
        self.clusterer = clusterer
        self.evaluator = evaluator

    def generate_paper_tables(self, aggregated_df) -> Dict[str, Path]:
        logger.info("Generating paper tables...")
        table_paths = {}

        cluster_stats = self.clusterer.get_cluster_summary()
        table_path = self.config.paths.paper / "tables" / "cluster_statistics.tex"
        self._generate_cluster_table(cluster_stats, table_path)
        table_paths['cluster_statistics'] = table_path
        
        eval_report = self.evaluator.export_report()
        table_path = self.config.paths.paper / "tables" / "evaluation_metrics.tex"
        self._generate_evaluation_table(eval_report, table_path)
        table_paths['evaluation_metrics'] = table_path
        
        if aggregated_df is not None:
            table_path = self.config.paths.paper / "tables" / "dataset_summary.tex"
            self._generate_dataset_table(aggregated_df, table_path)
            table_paths['dataset_summary'] = table_path
            
        logger.info(f"Paper tables generated: {len(table_paths)} tables")
        return table_paths

    def _generate_cluster_table(self, stats: Dict[str, Any], path: Path) -> None:
        latex = [
            "\\begin{table}[h]", "\\centering", "\\caption{Traffic State Cluster Statistics}",
            "\\label{tab:cluster_stats}", "\\begin{tabular}{lrrr}", "\\hline",
            "Cluster & State & Size & Percentage \\\\", "\\hline"
        ]
        total = sum(stats.get('cluster_sizes', {}).values()) + stats.get('n_noise', 0)
        
        for label, size in stats.get('cluster_sizes', {}).items():
            state = stats.get('state_mapping', {}).get(label, f'Cluster {label}')
            percentage = (size / total * 100) if total > 0 else 0
            latex.append(f"{label} & {state} & {size} & {percentage:.1f}\\% \\\\")
        
        if stats.get('n_noise', 0) > 0:
            percentage = stats['n_noise'] / total * 100 if total > 0 else 0
            latex.append(f"Noise & - & {stats['n_noise']} & {percentage:.1f}\\% \\\\")
        
        latex.extend(["\\hline", f"Total & - & {total} & 100.0\\% \\\\", "\\hline", "\\end{tabular}", "\\end{table}"])
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w') as f:
            f.write('\n'.join(latex))

    def _generate_evaluation_table(self, eval_report: Dict[str, Any], path: Path) -> None:
        latex = [
            "\\begin{table}[h]", "\\centering", "\\caption{Clustering Evaluation Metrics}",
            "\\label{tab:evaluation}", "\\begin{tabular}{lr}", "\\hline",
            "Metric & Value \\\\", "\\hline"
        ]
        metric_names = {
            'silhouette_score': 'Silhouette Score', 'davies_bouldin_index': 'Davies-Bouldin Index',
            'calinski_harabasz_index': 'Calinski-Harabasz Index', 'noise_ratio': 'Noise Ratio',
            'cluster_separation': 'Cluster Separation', 'stability_score': 'Stability Score'
        }
        for key, name in metric_names.items():
            value = eval_report.get(key, 'N/A')
            latex.append(f"{name} & {value:.4f} \\\\" if isinstance(value, float) else f"{name} & {value} \\\\")
        latex.extend(["\\hline", "\\end{tabular}", "\\end{table}"])
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w') as f:
            f.write('\n'.join(latex))

    def _generate_dataset_table(self, df: pd.DataFrame, path: Path) -> None:
        latex = [
            "\\begin{table}[h]", "\\centering", "\\caption{Dataset Summary Statistics}",
            "\\label{tab:dataset}", "\\begin{tabular}{lrrrr}", "\\hline",
            "Feature & Mean & Std & Min & Max \\\\", "\\hline"
        ]
        feature_columns = ['avg_vehicle_count', 'avg_density', 'avg_occupancy', 'avg_speed', 'avg_flow', 'avg_congestion_index']
        feature_names = {
            'avg_vehicle_count': 'Vehicle Count', 'avg_density': 'Density (veh/km)',
            'avg_occupancy': 'Occupancy (%)', 'avg_speed': 'Speed (km/h)',
            'avg_flow': 'Flow (veh/h)', 'avg_congestion_index': 'Congestion Index'
        }
        for col in feature_columns:
            if col in df.columns:
                name = feature_names.get(col, col)
                mean, std, min_val, max_val = df[col].mean(), df[col].std(), df[col].min(), df[col].max()
                latex.append(f"{name} & {mean:.2f} & {std:.2f} & {min_val:.2f} & {max_val:.2f} \\\\")
        latex.extend(["\\hline", "\\end{tabular}", "\\end{table}"])
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w') as f:
            f.write('\n'.join(latex))
