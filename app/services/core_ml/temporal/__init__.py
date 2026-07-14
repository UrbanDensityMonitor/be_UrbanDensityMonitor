"""
Temporal aggregation modules for traffic feature time series.
"""

from .temporal_aggregator import TemporalAggregator
from .window_manager import WindowManager

__all__ = [
    'TemporalAggregator',
    'WindowManager'
]