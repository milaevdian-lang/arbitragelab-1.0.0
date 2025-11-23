"""
Various utility functions used in cointegration/mean-reversion trading.
"""

import numpy as np
import pandas as pd
import statsmodels.api as sm


def get_half_life_of_mean_reversion(data) -> float:
    """
    Get half-life of mean-reversion under the assumption that data follows the Ornstein-Uhlenbeck process.

    :param data: (pd.Series or np.array) Data points.
    :return: (float) Half-life of mean reversion.
    """
    # Convert to pandas Series if numpy array
    if isinstance(data, np.ndarray):
        data = pd.Series(data)
    
    training_data = data.shift(1).dropna().values.reshape(-1, 1)
    target_values = data.diff().dropna()
    
    # Add constant term
    X = sm.add_constant(training_data)
    y = target_values.values
    
    # Fit OLS model using statsmodels
    reg = sm.OLS(y, X).fit()
    
    half_life = -np.log(2) / reg.params[1]  # params[0] is constant, params[1] is coefficient

    return half_life


def get_hurst_exponent(data: np.array, max_lags: int = 100) -> float:
    """
    Hurst Exponent Calculation.

    :param data: (np.array) Time Series that is going to be analyzed.
    :param max_lags: (int) Maximum amount of lags to be used calculating tau.
    :return: (float) Hurst exponent.
    """

    lags = range(2, max_lags)
    tau = [np.sqrt(np.std(np.subtract(data[lag:], data[:-lag])))
           for lag in lags]
    poly = np.polyfit(np.log(lags), np.log(tau), 1)

    return poly[0] * 2.0
