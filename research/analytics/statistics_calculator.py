"""
Statistical analysis and hypothesis testing for traffic data.
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional, Tuple
from scipy import stats
from dataclasses import dataclass
from pathlib import Path

from ..logger import logger


@dataclass
class StatisticalTest:
    """Container for statistical test results."""
    test_name: str
    statistic: float
    p_value: float
    significant: bool
    effect_size: float
    interpretation: str


class StatisticsCalculator:
    """
    Comprehensive statistical analysis for traffic metrics.
    """
    
    def __init__(self, alpha: float = 0.05) -> None:
        """
        Initialize statistics calculator.
        
        Args:
            alpha: Significance level
        """
        self.alpha = alpha
        self.results: Dict[str, Any] = {}
        
        logger.info(f"StatisticsCalculator initialized with alpha={alpha}")
    
    def calculate_descriptive_stats(self, df: pd.DataFrame) -> Dict[str, Dict[str, float]]:
        """
        Calculate descriptive statistics for all numeric columns.
        
        Args:
            df: Input DataFrame
            
        Returns:
            Dictionary with descriptive statistics per column
        """
        logger.info("Calculating descriptive statistics...")
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        stats_dict = {}
        
        for col in numeric_cols:
            data = df[col].dropna()
            
            if len(data) == 0:
                continue
            
            stats_dict[col] = {
                'count': len(data),
                'mean': float(np.mean(data)),
                'median': float(np.median(data)),
                'std': float(np.std(data)),
                'variance': float(np.var(data)),
                'min': float(np.min(data)),
                'max': float(np.max(data)),
                'range': float(np.max(data) - np.min(data)),
                'skewness': float(stats.skew(data)),
                'kurtosis': float(stats.kurtosis(data)),
                'q25': float(np.percentile(data, 25)),
                'q75': float(np.percentile(data, 75)),
                'iqr': float(np.percentile(data, 75) - np.percentile(data, 25)),
            }
            
            # Confidence interval for mean
            ci = stats.t.interval(
                0.95, 
                len(data)-1,
                loc=np.mean(data),
                scale=stats.sem(data)
            )
            stats_dict[col]['ci_95_lower'] = float(ci[0])
            stats_dict[col]['ci_95_upper'] = float(ci[1])
        
        self.results['descriptive'] = stats_dict
        return stats_dict
    
    def compare_groups(self, 
                      group1: np.ndarray, 
                      group2: np.ndarray,
                      group1_name: str = "Group 1",
                      group2_name: str = "Group 2",
                      test_type: str = 'auto') -> StatisticalTest:
        """
        Compare two groups using appropriate statistical test.
        
        Args:
            group1: First group data
            group2: Second group data
            group1_name: Name of first group
            group2_name: Name of second group
            test_type: Type of test ('auto', 'ttest', 'mannwhitney')
            
        Returns:
            StatisticalTest result
        """
        logger.info(f"Comparing {group1_name} vs {group2_name}")
        
        # Check normality for auto selection
        if test_type == 'auto':
            _, p_normal1 = stats.shapiro(group1[:min(5000, len(group1))])
            _, p_normal2 = stats.shapiro(group2[:min(5000, len(group2))])
            
            if p_normal1 > 0.05 and p_normal2 > 0.05:
                test_type = 'ttest'
            else:
                test_type = 'mannwhitney'
        
        if test_type == 'ttest':
            statistic, p_value = stats.ttest_ind(group1, group2)
            test_name = "Independent t-test"
            
            # Cohen's d effect size
            pooled_std = np.sqrt((np.var(group1) + np.var(group2)) / 2)
            effect_size = (np.mean(group1) - np.mean(group2)) / pooled_std if pooled_std > 0 else 0
            
        else:
            statistic, p_value = stats.mannwhitneyu(group1, group2, alternative='two-sided')
            test_name = "Mann-Whitney U test"
            
            # Effect size for Mann-Whitney
            n1, n2 = len(group1), len(group2)
            effect_size = 1 - (2 * statistic) / (n1 * n2)
        
        significant = p_value < self.alpha
        
        # Interpretation
        if significant:
            if effect_size > 0.8:
                interpretation = f"Large significant difference between {group1_name} and {group2_name}"
            elif effect_size > 0.5:
                interpretation = f"Medium significant difference between {group1_name} and {group2_name}"
            else:
                interpretation = f"Small significant difference between {group1_name} and {group2_name}"
        else:
            interpretation = f"No significant difference between {group1_name} and {group2_name}"
        
        result = StatisticalTest(
            test_name=test_name,
            statistic=float(statistic),
            p_value=float(p_value),
            significant=significant,
            effect_size=float(effect_size),
            interpretation=interpretation
        )
        
        logger.info(f"{test_name}: statistic={statistic:.4f}, p={p_value:.4f}, "
                   f"significant={significant}, effect_size={effect_size:.4f}")
        
        return result
    
    def correlation_analysis(self, 
                            df: pd.DataFrame,
                            method: str = 'pearson') -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Calculate correlation matrix with p-values.
        
        Args:
            df: Input DataFrame
            method: Correlation method ('pearson', 'spearman', 'kendall')
            
        Returns:
            Tuple of (correlation matrix, p-value matrix)
        """
        logger.info(f"Calculating {method} correlations...")
        
        numeric_df = df.select_dtypes(include=[np.number])
        
        n_cols = len(numeric_df.columns)
        corr_matrix = pd.DataFrame(
            np.zeros((n_cols, n_cols)),
            columns=numeric_df.columns,
            index=numeric_df.columns
        )
        pval_matrix = pd.DataFrame(
            np.ones((n_cols, n_cols)),
            columns=numeric_df.columns,
            index=numeric_df.columns
        )
        
        for i, col1 in enumerate(numeric_df.columns):
            for j, col2 in enumerate(numeric_df.columns):
                if i == j:
                    corr_matrix.loc[col1, col2] = 1.0
                    pval_matrix.loc[col1, col2] = 0.0
                elif i < j:
                    valid_mask = ~(numeric_df[col1].isna() | numeric_df[col2].isna())
                    
                    if valid_mask.sum() > 2:
                        if method == 'pearson':
                            corr, pval = stats.pearsonr(
                                numeric_df[col1][valid_mask],
                                numeric_df[col2][valid_mask]
                            )
                        elif method == 'spearman':
                            corr, pval = stats.spearmanr(
                                numeric_df[col1][valid_mask],
                                numeric_df[col2][valid_mask]
                            )
                        else:
                            corr, pval = stats.kendalltau(
                                numeric_df[col1][valid_mask],
                                numeric_df[col2][valid_mask]
                            )
                        
                        corr_matrix.loc[col1, col2] = corr
                        corr_matrix.loc[col2, col1] = corr
                        pval_matrix.loc[col1, col2] = pval
                        pval_matrix.loc[col2, col1] = pval
        
        self.results['correlation'] = {
            'method': method,
            'matrix': corr_matrix.to_dict(),
            'p_values': pval_matrix.to_dict()
        }
        
        return corr_matrix, pval_matrix
    
    def trend_analysis(self, 
                      df: pd.DataFrame,
                      time_column: str = 'window_start',
                      value_column: str = 'avg_congestion_index') -> Dict[str, Any]:
        """
        Analyze trends in time series data.
        
        Args:
            df: DataFrame with time series
            time_column: Column for timestamps
            value_column: Column for values to analyze
            
        Returns:
            Dictionary with trend analysis results
        """
        logger.info(f"Analyzing trends for {value_column}...")
        
        data = df[value_column].dropna().values
        
        if len(data) < 2:
            return {'error': 'Insufficient data'}
        
        # Linear trend
        x = np.arange(len(data))
        slope, intercept, r_value, p_value, std_err = stats.linregress(x, data)
        
        # Mann-Kendall trend test
        kendall_tau, kendall_p = stats.kendalltau(x, data)
        
        # Moving averages
        window_sizes = [5, 10, 20]
        moving_avgs = {}
        
        for window in window_sizes:
            if len(data) >= window:
                ma = np.convolve(data, np.ones(window)/window, mode='valid')
                moving_avgs[f'MA_{window}'] = ma.tolist()
        
        # Calculate trend strength
        trend_strength = abs(r_value)
        
        if trend_strength > 0.7:
            trend_description = "Strong"
        elif trend_strength > 0.4:
            trend_description = "Moderate"
        elif trend_strength > 0.2:
            trend_description = "Weak"
        else:
            trend_description = "Negligible"
        
        direction = "increasing" if slope > 0 else "decreasing"
        
        results = {
            'slope': float(slope),
            'intercept': float(intercept),
            'r_squared': float(r_value ** 2),
            'p_value': float(p_value),
            'significant': p_value < self.alpha,
            'trend_strength': trend_strength,
            'trend_description': f"{trend_description} {direction} trend",
            'mann_kendall_tau': float(kendall_tau),
            'mann_kendall_p': float(kendall_p),
            'moving_averages': moving_avgs,
        }
        
        self.results['trend'] = results
        
        logger.info(f"Trend analysis: {results['trend_description']} "
                   f"(R²={results['r_squared']:.4f}, p={results['p_value']:.4f})")
        
        return results
    
    def anomaly_detection(self, 
                         df: pd.DataFrame,
                         columns: Optional[List[str]] = None,
                         method: str = 'zscore',
                         threshold: float = 3.0) -> pd.DataFrame:
        """
        Detect anomalies in traffic data.
        
        Args:
            df: Input DataFrame
            columns: Columns to check (None for all numeric)
            method: Detection method ('zscore', 'iqr', 'isolation_forest')
            threshold: Threshold for anomaly detection
            
        Returns:
            DataFrame with anomaly flags
        """
        logger.info(f"Detecting anomalies using {method} method...")
        
        if columns is None:
            columns = df.select_dtypes(include=[np.number]).columns.tolist()
        
        result_df = df.copy()
        result_df['anomaly'] = False
        result_df['anomaly_score'] = 0.0
        
        for col in columns:
            if col not in df.columns:
                continue
            
            data = df[col].dropna()
            
            if method == 'zscore':
                z_scores = np.abs(stats.zscore(data))
                anomalies = z_scores > threshold
                anomaly_scores = z_scores / threshold
                
            elif method == 'iqr':
                Q1 = data.quantile(0.25)
                Q3 = data.quantile(0.75)
                IQR = Q3 - Q1
                
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                anomalies = (data < lower_bound) | (data > upper_bound)
                anomaly_scores = np.maximum(
                    (lower_bound - data) / IQR,
                    (data - upper_bound) / IQR
                )
                
            else:
                logger.warning(f"Unknown method: {method}")
                continue
            
            # Update results
            valid_indices = data.index
            result_df.loc[valid_indices, f'{col}_anomaly'] = anomalies.values
            result_df.loc[valid_indices, f'{col}_anomaly_score'] = anomaly_scores.values
            
            # Combine anomaly flags
            result_df['anomaly'] = result_df['anomaly'] | result_df[f'{col}_anomaly']
            result_df['anomaly_score'] = result_df['anomaly_score'].clip(
                result_df[f'{col}_anomaly_score']
            )
        
        n_anomalies = result_df['anomaly'].sum()
        logger.info(f"Detected {n_anomalies} anomalies "
                   f"({n_anomalies/len(result_df)*100:.2f}%)")
        
        return result_df
    
    def time_series_decomposition(self, 
                                  df: pd.DataFrame,
                                  value_column: str,
                                  period: int = 12) -> Dict[str, np.ndarray]:
        """
        Decompose time series into trend, seasonal, and residual components.
        
        Args:
            df: Input DataFrame
            value_column: Column to decompose
            period: Seasonal period
            
        Returns:
            Dictionary with decomposed components
        """
        logger.info(f"Decomposing time series: {value_column}")
        
        data = df[value_column].dropna().values        
        if len(data) < period * 2:
            logger.warning("Insufficient data for decomposition")
            return {}
        
        try:
            from statsmodels.tsa.seasonal import seasonal_decompose
            
            decomposition = seasonal_decompose(
                data, 
                model='additive', 
                period=period,
                extrapolate_trend='freq'
            )
            
            results = {
                'observed': data,
                'trend': decomposition.trend,
                'seasonal': decomposition.seasonal,
                'residual': decomposition.resid,
            }
            
            # Calculate variance explained
            total_var = np.var(data)
            trend_var = np.nanvar(decomposition.trend)
            seasonal_var = np.nanvar(decomposition.seasonal)
            residual_var = np.nanvar(decomposition.resid)
            
            results['variance_explained'] = {
                'trend': float(trend_var / total_var * 100) if total_var > 0 else 0,
                'seasonal': float(seasonal_var / total_var * 100) if total_var > 0 else 0,
                'residual': float(residual_var / total_var * 100) if total_var > 0 else 0,
            }
            
            self.results['decomposition'] = results
            return results
            
        except ImportError:
            logger.warning("statsmodels not available for decomposition")
            return {}
    
    def distribution_fit(self, 
                        data: np.ndarray,
                        distributions: List[str] = None) -> Dict[str, Dict[str, float]]:
        """
        Fit multiple distributions to data and find best fit.
        
        Args:
            data: Input data array
            distributions: List of distribution names to try
            
        Returns:
            Dictionary with fitting results
        """
        logger.info("Fitting distributions...")
        
        if distributions is None:
            distributions = ['norm', 'lognorm', 'expon', 'gamma', 'weibull_min']
        
        results = {}
        best_fit = None
        best_pvalue = 0
        
        for dist_name in distributions:
            try:
                dist = getattr(stats, dist_name)
                params = dist.fit(data)
                
                # Kolmogorov-Smirnov test
                ks_statistic, ks_pvalue = stats.kstest(data, dist_name, args=params)
                
                results[dist_name] = {
                    'params': [float(p) for p in params],
                    'ks_statistic': float(ks_statistic),
                    'ks_pvalue': float(ks_pvalue),
                    'fits_well': ks_pvalue > self.alpha
                }
                
                if ks_pvalue > best_pvalue:
                    best_pvalue = ks_pvalue
                    best_fit = dist_name
                    
            except Exception as e:
                logger.debug(f"Failed to fit {dist_name}: {e}")
                continue
        
        results['best_fit'] = best_fit
        
        logger.info(f"Best distribution fit: {best_fit} (p={best_pvalue:.4f})")
        
        self.results['distribution_fit'] = results
        return results
    
    def export_results(self, output_path: Path) -> None:
        """
        Export all statistical results.
        
        Args:
            output_path: Output directory path
        """
        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Export descriptive stats
        if 'descriptive' in self.results:
            desc_df = pd.DataFrame(self.results['descriptive']).T
            desc_df.to_csv(output_path / 'descriptive_statistics.csv')
        
        # Export correlation matrix
        if 'correlation' in self.results:
            corr_df = pd.DataFrame(self.results['correlation']['matrix'])
            corr_df.to_csv(output_path / 'correlation_matrix.csv')
            
            pval_df = pd.DataFrame(self.results['correlation']['p_values'])
            pval_df.to_csv(output_path / 'correlation_pvalues.csv')
        
        logger.info(f"Statistical results exported to: {output_path}")