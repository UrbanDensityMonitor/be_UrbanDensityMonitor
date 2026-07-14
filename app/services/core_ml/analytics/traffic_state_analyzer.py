"""
Traffic state analysis and classification from discovered clusters.
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from collections import defaultdict
from enum import Enum
from pathlib import Path

from ..logger import logger
from ..clustering.dbscan_clustering import ClusterResult


class TrafficState(Enum):
    """Traffic state enumeration."""
    FREE_FLOW = "free_flow"
    NORMAL = "normal"
    MODERATE = "moderate"
    SLOW = "slow"
    HEAVY = "heavy"
    CONGESTED = "congested"
    UNKNOWN = "unknown"


@dataclass
class StateTransition:
    """Container for state transition information."""
    from_state: str
    to_state: str
    count: int
    probability: float
    avg_duration: float  # seconds


@dataclass
class TrafficPattern:
    """Container for identified traffic pattern."""
    pattern_id: int
    states: List[str]
    duration: float
    occurrence_count: int
    time_of_day: str
    characteristics: Dict[str, float] = field(default_factory=dict)


class TrafficStateAnalyzer:
    """
    Analyzes discovered traffic states and identifies patterns.
    Provides interpretation of clustering results for traffic engineering.
    """
    
    def __init__(self) -> None:
        """Initialize traffic state analyzer."""
        self.states: Dict[int, TrafficState] = {}
        self.state_durations: Dict[str, List[float]] = defaultdict(list)
        self.transitions: List[StateTransition] = []
        self.patterns: List[TrafficPattern] = []
        self.hourly_statistics: Dict[int, Dict[str, Any]] = {}
        
        logger.info("TrafficStateAnalyzer initialized")
    
    def analyze_clusters(self, 
                         cluster_result: ClusterResult,
                         aggregated_df: pd.DataFrame) -> Dict[str, Any]:
        """
        Analyze clustering results to characterize traffic states.
        
        Args:
            cluster_result: DBSCAN clustering result
            aggregated_df: DataFrame with temporal features
            
        Returns:
            Dictionary with traffic state analysis
        """
        logger.info("Analyzing traffic states from clusters...")
        
        analysis = {
            'state_characteristics': {},
            'state_durations': {},
            'transitions': [],
            'patterns': [],
            'hourly_distribution': {},
            'summary': {}
        }
        
        # Map clusters to traffic states
        state_mapping = self._map_to_traffic_states(cluster_result)
        analysis['state_mapping'] = {
            k: v.value for k, v in state_mapping.items()
        }
        
        # Analyze each state's characteristics
        if cluster_result.cluster_statistics:
            analysis['state_characteristics'] = self._analyze_state_characteristics(
                cluster_result, state_mapping
            )
        
        # Calculate state durations
        if 'window_start' in aggregated_df.columns and len(cluster_result.labels) > 0:
            timestamps = aggregated_df['window_start'].values
            labels = cluster_result.labels
            
            analysis['state_durations'] = self._calculate_state_durations(
                timestamps, labels, state_mapping
            )
            
            # Analyze transitions
            analysis['transitions'] = self._analyze_transitions(
                timestamps, labels, state_mapping
            )
            
            # Detect patterns
            analysis['patterns'] = self._detect_patterns(
                timestamps, labels, state_mapping
            )
            
            # Hourly distribution
            analysis['hourly_distribution'] = self._calculate_hourly_distribution(
                timestamps, labels, state_mapping
            )
        
        # Generate summary
        analysis['summary'] = self._generate_summary(analysis)
        
        self._log_analysis_results(analysis)
        
        return analysis
    
    def _map_to_traffic_states(self, 
                               cluster_result: ClusterResult) -> Dict[int, TrafficState]:
        """
        Map cluster labels to traffic states based on feature characteristics.
        
        Args:
            cluster_result: Clustering result
            
        Returns:
            Dictionary mapping cluster label to traffic state
        """
        state_mapping = {}
        
        for label, center in cluster_result.cluster_centers.items():
            state = self._classify_state_from_center(center, cluster_result.feature_names)
            state_mapping[label] = state
        
        # Override with clusterer's state mapping if available
        if cluster_result.state_mapping:
            for label, state_name in cluster_result.state_mapping.items():
                try:
                    state_mapping[label] = TrafficState(state_name.lower().replace(' ', '_'))
                except ValueError:
                    # Keep the automatically determined state
                    pass
        
        self.states = state_mapping
        return state_mapping
    
    def _classify_state_from_center(self, 
                                    center: np.ndarray, 
                                    feature_names: List[str]) -> TrafficState:
        """
        Classify traffic state from cluster center values.
        
        Args:
            center: Cluster center array
            feature_names: Feature names
            
        Returns:
            TrafficState enum
        """
        # Find key feature indices
        speed_idx = self._find_feature_index(feature_names, 'speed')
        density_idx = self._find_feature_index(feature_names, 'density')
        congestion_idx = self._find_feature_index(feature_names, 'congestion')
        occupancy_idx = self._find_feature_index(feature_names, 'occupancy')
        
        speed = center[speed_idx] if speed_idx < len(center) else 0
        density = center[density_idx] if density_idx < len(center) else 0
        congestion = center[congestion_idx] if congestion_idx < len(center) else 0
        occupancy = center[occupancy_idx] if occupancy_idx < len(center) else 0
        
        # Decision tree for classification
        if congestion > 0.8 or speed < 5 or occupancy > 80:
            return TrafficState.CONGESTED
        elif congestion > 0.6 or speed < 15 or occupancy > 60:
            return TrafficState.HEAVY
        elif congestion > 0.4 or speed < 30:
            return TrafficState.SLOW
        elif congestion > 0.2 or speed < 45:
            return TrafficState.MODERATE
        elif density < 10 and speed > 50:
            return TrafficState.FREE_FLOW
        elif speed > 30:
            return TrafficState.NORMAL
        else:
            return TrafficState.UNKNOWN
    
    def _find_feature_index(self, feature_names: List[str], keyword: str) -> int:
        """Find feature index by keyword."""
        for i, name in enumerate(feature_names):
            if keyword.lower() in name.lower():
                return i
        return 0
    
    def _analyze_state_characteristics(self, 
                                      cluster_result: ClusterResult,
                                      state_mapping: Dict[int, TrafficState]) -> Dict[str, Dict[str, float]]:
        """
        Analyze detailed characteristics of each traffic state.
        
        Args:
            cluster_result: Clustering result
            state_mapping: Label to state mapping
            
        Returns:
            Dictionary of state characteristics
        """
        characteristics = {}
        
        for label, stats in cluster_result.cluster_statistics.items():
            state = state_mapping.get(label, TrafficState.UNKNOWN)
            
            chars = {
                'size': stats.get('size', 0),
                'percentage': stats.get('density', 0) * 100,
            }
            
            # Extract key metrics
            for key, value in stats.items():
                if key.endswith('_mean'):
                    chars[key] = value
            
            characteristics[state.value] = chars
        
        return characteristics
    
    def _calculate_state_durations(self, 
                                   timestamps: np.ndarray,
                                   labels: np.ndarray,
                                   state_mapping: Dict[int, TrafficState]) -> Dict[str, Dict[str, float]]:
        """
        Calculate duration statistics for each state.
        
        Args:
            timestamps: Array of timestamps
            labels: Cluster labels
            state_mapping: Label to state mapping
            
        Returns:
            Dictionary of duration statistics per state
        """
        durations = defaultdict(list)
        current_state = None
        current_start = None
        
        for i, (ts, label) in enumerate(zip(timestamps, labels)):
            state = state_mapping.get(label, TrafficState.UNKNOWN).value
            
            if state != current_state:
                if current_state is not None and current_start is not None:
                    duration = ts - current_start
                    durations[current_state].append(duration)
                
                current_state = state
                current_start = ts
        
        # Add last segment
        if current_state is not None and current_start is not None:
            duration = timestamps[-1] - current_start
            durations[current_state].append(duration)
        
        # Calculate statistics
        duration_stats = {}
        for state, durs in durations.items():
            if durs:
                duration_stats[state] = {
                    'mean_duration': float(np.mean(durs)),
                    'std_duration': float(np.std(durs)),
                    'min_duration': float(np.min(durs)),
                    'max_duration': float(np.max(durs)),
                    'total_duration': float(np.sum(durs)),
                    'count': len(durs)
                }
        
        return duration_stats
    
    def _analyze_transitions(self, 
                            timestamps: np.ndarray,
                            labels: np.ndarray,
                            state_mapping: Dict[int, TrafficState]) -> List[Dict[str, Any]]:
        """
        Analyze state transitions and create transition matrix.
        
        Args:
            timestamps: Array of timestamps
            labels: Cluster labels
            state_mapping: Label to state mapping
            
        Returns:
            List of transition dictionaries
        """
        transitions = defaultdict(lambda: defaultdict(lambda: {'count': 0, 'durations': []}))
        
        prev_state = None
        prev_ts = None
        
        for ts, label in zip(timestamps, labels):
            current_state = state_mapping.get(label, TrafficState.UNKNOWN).value
            
            if prev_state is not None and current_state != prev_state:
                transitions[prev_state][current_state]['count'] += 1
                
                if prev_ts is not None:
                    duration = ts - prev_ts
                    transitions[prev_state][current_state]['durations'].append(duration)
            
            prev_state = current_state
            prev_ts = ts
        
        # Build transition list
        transition_list = []
        total_transitions = sum(
            sum(t['count'] for t in to_states.values())
            for to_states in transitions.values()
        )
        
        for from_state, to_states in transitions.items():
            for to_state, data in to_states.items():
                transition_list.append({
                    'from_state': from_state,
                    'to_state': to_state,
                    'count': data['count'],
                    'probability': data['count'] / total_transitions if total_transitions > 0 else 0,
                    'avg_duration': float(np.mean(data['durations'])) if data['durations'] else 0
                })
        
        # Sort by count descending
        transition_list.sort(key=lambda x: x['count'], reverse=True)
        
        self.transitions = [
            StateTransition(**t) for t in transition_list
        ]
        
        return transition_list
    
    def _detect_patterns(self, 
                        timestamps: np.ndarray,
                        labels: np.ndarray,
                        state_mapping: Dict[int, TrafficState]) -> List[Dict[str, Any]]:
        """
        Detect recurring traffic patterns.
        
        Args:
            timestamps: Array of timestamps
            labels: Cluster labels
            state_mapping: Label to state mapping
            
        Returns:
            List of detected patterns
        """
        patterns = []
        pattern_sequences = []
        current_sequence = []
        current_start = None
        
        # Build state sequence
        for ts, label in zip(timestamps, labels):
            state = state_mapping.get(label, TrafficState.UNKNOWN).value
            
            if not current_sequence or state != current_sequence[-1]:
                if current_sequence:
                    pattern_sequences.append({
                        'states': current_sequence.copy(),
                        'duration': ts - current_start if current_start else 0
                    })
                current_sequence = [state]
                current_start = ts
            else:
                current_sequence.append(state)
        
        # Add last sequence
        if current_sequence and current_start:
            pattern_sequences.append({
                'states': current_sequence.copy(),
                'duration': timestamps[-1] - current_start
            })
        
        # Find repeated patterns
        pattern_counts = defaultdict(int)
        pattern_durations = defaultdict(list)
        
        for seq in pattern_sequences:
            pattern_key = '->'.join(dict.fromkeys(seq['states']))  # Unique states in order
            pattern_counts[pattern_key] += 1
            pattern_durations[pattern_key].append(seq['duration'])
        
        # Build pattern list
        pattern_id = 0
        for pattern_key, count in sorted(pattern_counts.items(), 
                                        key=lambda x: x[1], reverse=True):
            if count >= 2:  # Only patterns that repeat
                pattern_id += 1
                
                # Determine time of day (simplified)
                avg_duration = np.mean(pattern_durations[pattern_key])
                time_category = self._categorize_time(avg_duration)
                
                patterns.append({
                    'pattern_id': pattern_id,
                    'states': pattern_key.split('->'),
                    'occurrence_count': count,
                    'avg_duration': float(avg_duration),
                    'time_of_day': time_category
                })
        
        self.patterns = [TrafficPattern(**p) for p in patterns]
        return patterns
    
    def _categorize_time(self, duration: float) -> str:
        """Categorize time based on duration."""
        hours = duration / 3600
        if hours < 0.5:
            return "short_term"
        elif hours < 2:
            return "medium_term"
        else:
            return "long_term"
    
    def _calculate_hourly_distribution(self, 
                                      timestamps: np.ndarray,
                                      labels: np.ndarray,
                                      state_mapping: Dict[int, TrafficState]) -> Dict[int, Dict[str, int]]:
        """
        Calculate hourly distribution of traffic states.
        
        Args:
            timestamps: Array of timestamps
            labels: Cluster labels
            state_mapping: Label to state mapping
            
        Returns:
            Dictionary of hourly distributions
        """
        hourly = defaultdict(lambda: defaultdict(int))
        
        for ts, label in zip(timestamps, labels):
            # Convert timestamp to hour of day
            hour = int((ts % 86400) / 3600)  # Assuming ts in seconds from midnight
            
            state = state_mapping.get(label, TrafficState.UNKNOWN).value
            hourly[hour][state] += 1
        
        # Convert to regular dict
        hourly_dict = {}
        for hour in range(24):
            hourly_dict[hour] = dict(hourly[hour])
        
        self.hourly_statistics = hourly_dict
        return hourly_dict
    
    def _generate_summary(self, analysis: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate executive summary of traffic analysis.
        
        Args:
            analysis: Complete analysis dictionary
            
        Returns:
            Summary dictionary
        """
        summary = {
            'total_states': len(analysis.get('state_mapping', {})),
            'dominant_state': self._find_dominant_state(analysis),
            'state_diversity': self._calculate_diversity(analysis),
            'stability_index': self._calculate_stability(analysis),
            'peak_congestion_hour': self._find_peak_hour(analysis),
            'recommendations': self._generate_recommendations(analysis)
        }
        
        return summary
    
    def _find_dominant_state(self, analysis: Dict[str, Any]) -> str:
        """Find the most frequent traffic state."""
        characteristics = analysis.get('state_characteristics', {})
        
        if not characteristics:
            return "unknown"
        
        max_size = 0
        dominant = "unknown"
        
        for state, chars in characteristics.items():
            if chars.get('size', 0) > max_size:
                max_size = chars['size']
                dominant = state
        
        return dominant
    
    def _calculate_diversity(self, analysis: Dict[str, Any]) -> float:
        """Calculate state diversity (entropy-based)."""
        characteristics = analysis.get('state_characteristics', {})
        
        if not characteristics:
            return 0.0
        
        total = sum(chars.get('size', 0) for chars in characteristics.values())
        
        if total == 0:
            return 0.0
        
        entropy = 0.0
        for chars in characteristics.values():
            p = chars.get('size', 0) / total
            if p > 0:
                entropy -= p * np.log(p)
        
        max_entropy = np.log(len(characteristics))
        
        return float(entropy / max_entropy if max_entropy > 0 else 0)
    
    def _calculate_stability(self, analysis: Dict[str, Any]) -> float:
        """Calculate traffic stability based on transitions."""
        transitions = analysis.get('transitions', [])
        
        if not transitions:
            return 0.0
        
        # More transitions = less stable
        total_transitions = sum(t['count'] for t in transitions)
        unique_transitions = len(transitions)
        
        # Normalize
        stability = 1.0 / (1.0 + unique_transitions * total_transitions / 100)
        
        return float(stability)
    
    def _find_peak_hour(self, analysis: Dict[str, Any]) -> int:
        """Find hour with most congestion."""
        hourly = analysis.get('hourly_distribution', {})
        
        max_congestion = 0
        peak_hour = 0
        
        for hour, states in hourly.items():
            congestion_count = states.get('congested', 0) + states.get('heavy', 0)
            
            if congestion_count > max_congestion:
                max_congestion = congestion_count
                peak_hour = hour
        
        return peak_hour
    
    def _generate_recommendations(self, analysis: Dict[str, Any]) -> List[str]:
        """Generate traffic management recommendations."""
        recommendations = []
        
        summary = analysis.get('summary', {})
        transitions = analysis.get('transitions', [])
        hourly = analysis.get('hourly_distribution', {})
        
        # Check for congestion
        peak_hour = summary.get('peak_congestion_hour', 0)
        if peak_hour:
            recommendations.append(
                f"Consider traffic management interventions during hour {peak_hour}"
            )
        
        # Check stability
        stability = summary.get('stability_index', 0)
        if stability < 0.3:
            recommendations.append(
                "Traffic flow is unstable - consider adaptive signal timing"
            )
        
        # Check frequent transitions
        if transitions and len(transitions) > 10:
            recommendations.append(
                "High number of state transitions detected - investigate causes"
            )
        
        # Check for persistent congestion
        hourly_congestion = [
            hour for hour, states in hourly.items()
            if states.get('congested', 0) > 10
        ]
        
        if len(hourly_congestion) > 6:
            recommendations.append(
                "Persistent congestion detected across multiple hours - "
                "consider infrastructure improvements"
            )
        
        if not recommendations:
            recommendations.append("Traffic flow is generally stable")
        
        return recommendations
    
    def _log_analysis_results(self, analysis: Dict[str, Any]) -> None:
        """Log analysis summary."""
        summary = analysis.get('summary', {})
        
        logger.info("=" * 50)
        logger.info("TRAFFIC STATE ANALYSIS SUMMARY")
        logger.info("=" * 50)
        logger.info(f"Total States: {summary.get('total_states', 0)}")
        logger.info(f"Dominant State: {summary.get('dominant_state', 'unknown')}")
        logger.info(f"Diversity Index: {summary.get('state_diversity', 0):.3f}")
        logger.info(f"Stability Index: {summary.get('stability_index', 0):.3f}")
        logger.info(f"Peak Congestion Hour: {summary.get('peak_congestion_hour', 0)}")
        logger.info("=" * 50)
        
        for rec in summary.get('recommendations', []):
            logger.info(f"  • {rec}")
    
    def export_analysis(self, analysis: Dict[str, Any], output_path: Path) -> None:
        """
        Export analysis results to CSV files.
        
        Args:
            analysis: Analysis dictionary
            output_path: Output directory path
        """
        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Export state characteristics
        if analysis.get('state_characteristics'):
            chars_df = pd.DataFrame(analysis['state_characteristics']).T
            chars_df.to_csv(output_path / 'state_characteristics.csv')
        
        # Export transitions
        if analysis.get('transitions'):
            trans_df = pd.DataFrame(analysis['transitions'])
            trans_df.to_csv(output_path / 'state_transitions.csv', index=False)
        
        # Export patterns
        if analysis.get('patterns'):
            patterns_df = pd.DataFrame(analysis['patterns'])
            patterns_df.to_csv(output_path / 'traffic_patterns.csv', index=False)
        
        # Export hourly distribution
        if analysis.get('hourly_distribution'):
            hourly_df = pd.DataFrame(analysis['hourly_distribution']).T
            hourly_df.to_csv(output_path / 'hourly_distribution.csv')
        
        logger.info(f"Analysis exported to: {output_path}")
    
    def get_transition_matrix(self) -> pd.DataFrame:
        """
        Get state transition matrix.
        
        Returns:
            DataFrame with transition probabilities
        """
        if not self.transitions:
            return pd.DataFrame()
        
        states = sorted(set([t.from_state for t in self.transitions] + 
                           [t.to_state for t in self.transitions]))
        
        matrix = pd.DataFrame(0.0, index=states, columns=states)
        
        for transition in self.transitions:
            matrix.loc[transition.from_state, transition.to_state] = transition.probability
        
        return matrix