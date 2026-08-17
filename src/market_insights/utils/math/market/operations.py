import numpy as np
import numpy.typing as npt

from market_insights.utils.math import ScientificNotation

# Simple Moving Average (SMA)

def SMA(vec: npt.NDArray[np.float64], period: int) -> ScientificNotation:
    average = np.average(vec[0:period])
    return ScientificNotation(average)

# Exponential Moving Average (EMA)

# Moving Average Convergence Divergence (MACD)

def MACD() -> npt.NDArray[np.float64]:
    return np.array([])
