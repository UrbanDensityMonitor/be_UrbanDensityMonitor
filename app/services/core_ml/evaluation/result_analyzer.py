"""
Result analysis and interpretation for clustering outcomes.
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from pathlib import Path

from ..logger import logger
from ..clustering.dbscan_clustering import ClusterResult


@dataclass
class AnalysisReport:
    """Container for comprehensive analysis report."""
    clustering_quality: Dict[str, Any]
    cluster_profiles: Dict[int, Dict[str, Any]]
    state_transitions: List[Dict[str, Any]]
    temporal_patterns: Dict[str, Any]
    recommendations: List[str]
    summary: str


class ResultAnalyzer:
    """
    Analyzes and interprets clustering results for traffic state discovery.
    Provides insights and generates comprehensive reports.
    """
    
    def __init__(self) -> None:
        """Initialize result analyzer."""
        self.report: Optional[AnalysisReport] = None
        
        logger.info("ResultAnalyzer initialized")
    
    def analyze(self,
                cluster_result: ClusterResult,
                features_df: pd.DataFrame,
                raw_features_df: Optional[pd.DataFrame] = None) -> AnalysisReport:
        """
        Perform comprehensive analysis of clustering results.
        
        Args:
            cluster_result: DBSCAN clustering result
            features_df: DataFrame with aggregated features
            raw_features_df: DataFrame with raw features (optional)
            
        Returns:
            AnalysisReport with complete analysis
        """
        logger.info("Analyzing clustering results...")
        
        # Analyze clustering quality
        quality = self._analyze_clustering_quality(cluster_result)
        
        # Profile each cluster
        profiles = self._profile_clusters(cluster_result, features_df)
        
        # Analyze state transitions
        transitions = self._analyze_transitions(cluster_result, features_df)
        
        # Detect temporal patterns
        temporal = self._detect_temporal_patterns(cluster_result, features_df)
        
        # Generate recommendations
        recommendations = self._generate_recommendations(
            quality, profiles, transitions, temporal
        )
        
        # Generate summary
        summary = self._generate_summary(quality, profiles, temporal)
        
        self.report = AnalysisReport(
            clustering_quality=quality,
            cluster_profiles=profiles,
            state_transitions=transitions,
            temporal_patterns=temporal,
            recommendations=recommendations,
            summary=summary
        )
        
        logger.info("Analysis complete")
        logger.info(f"Summary: {summary[:200]}...")
        
        return self.report
    
    def _analyze_clustering_quality(self, 
                                   cluster_result: ClusterResult) -> Dict[str, Any]:
        """
        Analyze clustering quality metrics.
        
        Args:
            cluster_result: Clustering result
            
        Returns:
            Quality analysis dictionary
        """
        n_samples = len(cluster_result.labels)
        n_noise = cluster_result.n_noise
        noise_ratio = n_noise / n_samples if n_samples > 0 else 0
        
        quality = {
            'n_clusters': cluster_result.n_clusters,
            'n_samples': n_samples,
            'n_noise': n_noise,
            'noise_ratio': noise_ratio,
            'silhouette_score': cluster_result.silhouette_score,
            'parameters': cluster_result.parameters,
        }
        
        # Quality assessment
        if cluster_result.n_clusters == 0:
            quality['assessment'] = "No clusters found - data may not have structure"
        elif cluster_result.silhouette_score > 0.7:
            quality['assessment'] = "Excellent clustering structure"
        elif cluster_result.silhouette_score > 0.5:
            quality['assessment'] = "Good clustering structure"
        elif cluster_result.silhouette_score > 0.3:
            quality['assessment'] = "Moderate clustering structure"
        elif cluster_result.silhouette_score >= 0:
            quality['assessment'] = "Weak clustering structure"
        else:
            quality['assessment'] = "Poor clustering structure"
        
        # Noise assessment
        if noise_ratio > 0.3:
            quality['noise_assessment'] = "High noise ratio - consider parameter tuning"
        elif noise_ratio > 0.1:
            quality['noise_assessment'] = "Moderate noise ratio"
        else:
            quality['noise_assessment'] = "Acceptable noise ratio"
        
        return quality
    
    def _profile_clusters(self,
                         cluster_result: ClusterResult,
                         features_df: pd.DataFrame) -> Dict[int, Dict[str, Any]]:
        """
        Create detailed profiles for each cluster.
        
        Args:
            cluster_result: Clustering result
            features_df: Features DataFrame
            
        Returns:
            Dictionary of cluster profiles
        """
        profiles = {}
        
        feature_columns = [
            c for c in features_df.columns 
            if c in (cluster_result.feature_names or [])
        ]
        
        if not feature_columns:
            feature_columns = features_df.select_dtypes(include=[np.number]).columns.tolist()
            feature_columns = [c for c in feature_columns if 'timestamp' not in c.lower()]
        
        for label in cluster_result.cluster_centers.keys():
            mask = cluster_result.labels == label
            
            profile = {
                'size': int(np.sum(mask)),
                'percentage': float(np.sum(mask) / len(cluster_result.labels) * 100),
                'state': cluster_result.state_mapping.get(label, f'Cluster_{label}'),
            }
            
            # Feature statistics
            for i, feature_name in enumerate(feature_columns):
                if i < features_df[feature_columns].shape[1]:
                    cluster_values = features_df[feature_columns].values[mask, i]
                    
                    profile[f'{feature_name}_mean'] = float(np.mean(cluster_values))
                    profile[f'{feature_name}_std'] = float(np.std(cluster_values))
                    profile[f'{feature_name}_median'] = float(np.median(cluster_values))
            
            # Characteristics based on dominant features
            profile['characteristics'] = self._interpret_cluster_characteristics(
                profile, feature_columns
            )
            
            profiles[label] = profile
        
        return profiles
    
    def _interpret_cluster_characteristics(self,
                                          profile: Dict[str, Any],
                                          feature_columns: List[str]) -> List[str]:
        """
        Interpret cluster characteristics in human-readable form.
        
        Args:
            profile: Cluster profile dictionary
            feature_columns: Feature column names
            
        Returns:
            List of characteristic descriptions
        """
        characteristics = []
        
        # Speed interpretation
        for col in feature_columns:
            if 'speed' in col.lower():
                mean_speed = profile.get(f'{col}_mean', 0)
                
                if mean_speed > 60:
                    characteristics.append("High-speed traffic")
                elif mean_speed > 40:
                    characteristics.append("Moderate-speed traffic")
                elif mean_speed > 20:
                    characteristics.append("Low-speed traffic")
                elif mean_speed > 5:
                    characteristics.append("Very slow traffic")
                else:
                    characteristics.append("Near-stationary traffic")
                break
        
        # Density interpretation
        for col in feature_columns:
            if 'density' in col.lower():
                mean_density = profile.get(f'{col}_mean', 0)
                
                if mean_density > 30:
                    characteristics.append("High density")
                elif mean_density > 15:
                    characteristics.append("Moderate density")
                else:
                    characteristics.append("Low density")
                break
        
        # Congestion interpretation
        for col in feature_columns:
            if 'congestion' in col.lower():
                mean_congestion = profile.get(f'{col}_mean', 0)
                
                if mean_congestion > 0.7:
                    characteristics.append("Severe congestion")
                elif mean_congestion > 0.4:
                    characteristics.append("Moderate congestion")
                elif mean_congestion > 0.1:
                    characteristics.append("Light congestion")
                else:
                    characteristics.append("Free-flowing")
                break
        
        return characteristics
    
    def _analyze_transitions(self,
                            cluster_result: ClusterResult,
                            features_df: pd.DataFrame) -> List[Dict[str, Any]]:
        """
        Analyze transitions between traffic states.
        
        Args:
            cluster_result: Clustering result
            features_df: Features DataFrame
            
        Returns:
            List of transition analyses
        """
        if 'window_start' not in features_df.columns:
            return []
        
        timestamps = features_df['window_start'].values
        labels = cluster_result.labels
        
        transitions = []
        prev_state = None
        prev_ts = None
        state_durations = {}
        
        for ts, label in zip(timestamps, labels):
            current_state = cluster_result.state_mapping.get(label, 'Noise')
            
            if prev_state is not None and current_state != prev_state:
                transition = {
                    'from_state': prev_state,
                    'to_state': current_state,
                    'timestamp': float(ts),
                    'duration': float(ts - prev_ts) if prev_ts is not None else 0
                }
                transitions.append(transition)
                
                # Track state durations
                if prev_state not in state_durations:
                    state_durations[prev_state] = []
                if prev_ts is not None:
                    state_durations[prev_state].append(ts - prev_ts)
            
            prev_state = current_state
            prev_ts = ts
        
        # Calculate transition statistics
        transition_counts = {}
        for t in transitions:
            key = f"{t['from_state']} -> {t['to_state']}"
            transition_counts[key] = transition_counts.get(key, 0) + 1
        
        # Add probabilities
        total = sum(transition_counts.values())
        for transition in transitions:
            key = f"{transition['from_state']} -> {transition['to_state']}"
            transition['probability'] = transition_counts[key] / total if total > 0 else 0
        
        return transitions
    
    def _detect_temporal_patterns(self,
                                 cluster_result: ClusterResult,
                                 features_df: pd.DataFrame) -> Dict[str, Any]:
        """
        Detect temporal patterns in traffic states.
        
        Args:
            cluster_result: Clustering result
            features_df: Features DataFrame
            
        Returns:
            Dictionary of temporal patterns
        """
        patterns = {
            'peak_hours': [],
            'off_peak_hours': [],
            'transition_periods': [],
            'stable_periods': [],
            'daily_pattern': None
        }
        
        if 'window_start' not in features_df.columns:
            return patterns
        
        # Convert timestamps to hours
        timestamps = features_df['window_start'].values
        labels = cluster_result.labels
        
        # Hourly state distribution
        hourly_states = {h: {} for h in range(24)}
        
        for ts, label in zip(timestamps, labels):
            hour = int((ts % 86400) / 3600) if ts < 86400 else int((ts / 3600) % 24)
            hour = min(hour, 23)
            
            state = cluster_result.state_mapping.get(label, 'Noise')
            hourly_states[hour][state] = hourly_states[hour].get(state, 0) + 1
        
        # Identify peak hours (high congestion hours)
        congestion_states = ['Congested', 'Heavy Congestion', 'Heavy', 'congested', 'heavy']
        
        for hour, states in hourly_states.items():
            total = sum(states.values())
            if total == 0:
                continue
            
            congestion_count = sum(states.get(s, 0) for s in congestion_states)
            congestion_ratio = congestion_count / total
            
            if congestion_ratio > 0.5:
                patterns['peak_hours'].append({
                    'hour': hour,
                    'congestion_ratio': congestion_ratio
                })
            elif congestion_ratio < 0.1:
                patterns['off_peak_hours'].append({
                    'hour': hour,
                    'free_flow_ratio': 1 - congestion_ratio
                })
        
        # Detect stable vs transition periods
        stability_scores = []
        for i in range(1, len(labels)):
            if labels[i] == labels[i-1]:
                stability_scores.append(1)
            else:
                stability_scores.append(0)
        
        # Smooth stability
        window_size = min(10, len(stability_scores) // 2)
        if window_size > 0:
            smoothed = np.convolve(stability_scores, np.ones(window_size)/window_size, mode='valid')
            
            for i, score in enumerate(smoothed):
                ts = timestamps[i + window_size // 2] if i + window_size // 2 < len(timestamps) else timestamps[i]
                
                if score > 0.8:
                    patterns['stable_periods'].append({
                        'timestamp': float(ts),
                        'stability': float(score)
                    })
                elif score < 0.3:
                    patterns['transition_periods'].append({
                        'timestamp': float(ts),
                        'stability': float(score)
                    })
        
        return patterns
    
    def _generate_recommendations(self,
                                 quality: Dict[str, Any],
                                 profiles: Dict[int, Dict[str, Any]],
                                 transitions: List[Dict[str, Any]],
                                 temporal: Dict[str, Any]) -> List[str]:
        """
        Generate actionable recommendations based on analysis.
        
        Args:
            quality: Clustering quality
            profiles: Cluster profiles
            transitions: State transitions
            temporal: Temporal patterns
            
        Returns:
            List of recommendations
        """
        recommendations = []
        
        # Clustering quality recommendations
        if quality.get('noise_ratio', 0) > 0.3:
            recommendations.append(
                "High noise ratio detected. Consider adjusting DBSCAN parameters "
                "or using additional features for better cluster separation."
            )
        
        if quality.get('n_clusters', 0) == 0:
            recommendations.append(
                "No traffic states discovered. Data may need additional preprocessing "
                "or different clustering approach."
            )
        
        # Traffic management recommendations
        for label, profile in profiles.items():
            if 'congestion' in str(profile.get('state', '')).lower():
                size = profile.get('percentage', 0)
                
                if size > 30:
                    recommendations.append(
                        f"Critical: {profile['state']} accounts for {size:.1f}% of observations. "
                        "Immediate traffic management intervention recommended."
                    )
                elif size > 15:
                    recommendations.append(
                        f"Warning: {profile['state']} accounts for {size:.1f}% of observations. "
                        "Monitor and prepare mitigation strategies."
                    )
        
        # Peak hour recommendations
        peak_hours = temporal.get('peak_hours', [])
        if peak_hours:
            peak_times = ', '.join([f"{p['hour']}:00" for p in peak_hours[:3]])
            recommendations.append(
                f"Peak congestion detected at hours: {peak_times}. "
                "Consider implementing peak-hour traffic management strategies."
            )
        
        # Transition recommendations
        frequent_transitions = {}
        for t in transitions:
            key = f"{t['from_state']}->{t['to_state']}"
            frequent_transitions[key] = frequent_transitions.get(key, 0) + 1
        
        worst_transition = max(frequent_transitions.items(), key=lambda x: x[1]) if frequent_transitions else None
        
        if worst_transition and worst_transition[1] > len(transitions) * 0.2:
            recommendations.append(
                f"Frequent state transition detected: {worst_transition[0]}. "
                "Investigate causes of this recurring pattern."
            )
        
        if not recommendations:
            recommendations.append(
                "Traffic patterns appear stable. Continue monitoring for changes."
            )
        
        return recommendations
    
    def _generate_summary(self,
                         quality: Dict[str, Any],
                         profiles: Dict[int, Dict[str, Any]],
                         temporal: Dict[str, Any]) -> str:
        """
        Generate executive summary of analysis.
        
        Args:
            quality: Clustering quality
            profiles: Cluster profiles
            temporal: Temporal patterns
            
        Returns:
            Summary string
        """
        n_states = len(profiles)
        n_clusters = quality.get('n_clusters', 0)
        
        summary_parts = []
        
        # Clustering summary
        summary_parts.append(
            f"Analysis discovered {n_clusters} distinct traffic states "
            f"with a quality score of {quality.get('silhouette_score', 0):.3f}. "
        )
        
        # State distribution summary
        state_sizes = [(p.get('state', 'Unknown'), p.get('percentage', 0)) 
                      for p in profiles.values()]
        state_sizes.sort(key=lambda x: x[1], reverse=True)
        
        if state_sizes:
            summary_parts.append(
                f"The dominant state is '{state_sizes[0][0]}' "
                f"({state_sizes[0][1]:.1f}% of observations)"
            )
            
            if len(state_sizes) > 1:
                summary_parts.append(
                    f", followed by '{state_sizes[1][0]}' "
                    f"({state_sizes[1][1]:.1f}%)"
                )
            summary_parts.append(". ")
        
        # Temporal patterns summary
        peak_hours = temporal.get('peak_hours', [])
        if peak_hours:
            summary_parts.append(
                f"Peak congestion typically occurs at hour {peak_hours[0]['hour']}:00 "
                f"with {len(peak_hours)} peak hours identified. "
            )
        
        # Assessment
        if quality.get('silhouette_score', 0) >= 0.5:
            summary_parts.append(
                "The clustering provides reliable traffic state identification "
                "suitable for traffic management applications."
            )
        else:
            summary_parts.append(
                "The clustering provides exploratory insights but may require "
                "refinement for operational use."
            )
        
        return ''.join(summary_parts)
    
    def export_report(self, output_path: Path) -> None:
        """
        Export analysis report to files.
        
        Args:
            output_path: Output directory path
        """
        if self.report is None:
            logger.warning("No report generated yet")
            return
        
        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)
        
        import json
        
        # Convert to serializable format
        report_dict = {
            'clustering_quality': self.report.clustering_quality,
            'cluster_profiles': {
                str(k): v for k, v in self.report.cluster_profiles.items()
            },
            'state_transitions': self.report.state_transitions[:100],  # Limit for file size
            'temporal_patterns': {
                k: v[:20] if isinstance(v, list) else v 
                for k, v in self.report.temporal_patterns.items()
            },
            'recommendations': self.report.recommendations,
            'summary': self.report.summary
        }
        
        # Save as JSON
        with open(output_path / 'analysis_report.json', 'w') as f:
            json.dump(report_dict, f, indent=2)
        
        # Save recommendations as text
        with open(output_path / 'recommendations.txt', 'w') as f:
            f.write("TRAFFIC STATE ANALYSIS RECOMMENDATIONS\n")
            f.write("=" * 50 + "\n\n")
            for i, rec in enumerate(self.report.recommendations, 1):
                f.write(f"{i}. {rec}\n")
        
        # Save summary
        with open(output_path / 'executive_summary.txt', 'w') as f:
            f.write("EXECUTIVE SUMMARY\n")
            f.write("=" * 50 + "\n\n")
            f.write(self.report.summary)
        
        logger.info(f"Report exported to: {output_path}")

