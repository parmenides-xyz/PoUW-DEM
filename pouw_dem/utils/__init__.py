"""
Utilities Module

Provides utility functions and classes for the PoUW-DEM system:
- Analytics tracking and database management
- Performance metrics calculation
- Data recording and analysis tools
"""

from .analytics_tracker import (
    AnalyticsDatabase,
    AnalyticsTracker,
    AnalyticsCalculator,
    get_analytics_status,
    get_win_rate,
    get_cumulative_profits,
    get_market_spreads,
    get_best_worst_hours,
    get_hourly_statistics,
    get_grid_response_stats,
    record_decision,
    record_revenue,
    record_grid_urgency,
    start_recording,
    stop_recording,
    integrate_with_agent
)

__all__ = [
    # Classes
    "AnalyticsDatabase",
    "AnalyticsTracker",
    "AnalyticsCalculator",
    
    # Analytics functions
    "get_analytics_status",
    "get_win_rate",
    "get_cumulative_profits",
    "get_market_spreads",
    "get_best_worst_hours",
    "get_hourly_statistics",
    "get_grid_response_stats",
    
    # Recording functions
    "record_decision",
    "record_revenue",
    "record_grid_urgency",
    "start_recording",
    "stop_recording",
    
    # Integration
    "integrate_with_agent",
]