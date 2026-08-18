import math

import numpy as np
import numpy.typing as npt

from market_insights.utils.math import ScientificNotation

# Simple Moving Average (SMA)

def SMA(vec: npt.NDArray[np.float64], period: int) -> ScientificNotation:
    average = np.average(vec[0:period])
    return ScientificNotation(average)

# Exponential Moving Average (EMA)

def EMA(vec: npt.NDArray[np.float64], smooth: float, period: int) -> ScientificNotation:
    index = period
    span = index + period

    factor = smooth / (1 + period)

    last_ema = SMA(vec[index:span], period).value

    ema = 0

    for i in range(index - 1, -1, -1):
        ema = vec[i] * factor + last_ema * (1 - factor)
        last_ema = ema

    return ScientificNotation(ema)

# Moving Average Convergence Divergence (MACD)

def MACD(vec: npt.NDArray[np.float64]) -> ScientificNotation:
    ema12 = EMA(vec, 2, 12).value
    ema26 = EMA(vec, 2, 26).value

    return ScientificNotation(ema26 - ema12)

# Desvio Padrão

def StandardDeviation(vec: npt.NDArray[np.float64], period: int) -> float:
    averg = SMA(vec, period).value

    sd = 0

    for i in range(period):
        v = vec[i] - averg
        sd += v * v

    sd /= period
    sd = math.sqrt(sd)

    return sd

# Bandas de Bollinger

def BollingerBands(vec: npt.NDArray[np.float64], period: int) -> list[ScientificNotation]:
    averg = SMA(vec, period).value
    sd = StandardDeviation(vec, period)

    factor = sd * 2

    sup = ScientificNotation(averg + factor)
    inf = ScientificNotation(averg - factor)

    return [sup, inf]

# Relative Strength Index (RSI)

def RSI(high: npt.NDArray[np.float64], low: npt.NDArray[np.float64], period: int) -> float:
    averg_high = SMA(high, period).value
    averg_low = SMA(low, period).value

    rs = averg_high / averg_low

    rsi = 100 - 100 / (1 + rs)

    return rsi

# Average True Range (ATR)

def ATR(vec: npt.NDArray[np.float64], closes: npt.NDArray[np.float64], period: int) -> float:
    index = period
    span = index + period

    mx = np.max(vec[index:span])
    mn = np.min(vec[index:span])
    
    close = closes[index + 1]
    
    diff1 = mx - mn
    diff2 = mn - close
    diff3 = mx - close

    tr = (diff1 + diff2 + diff3) / 3

    atr = 0

    span -= 1
    
    for i in range(index - 1, -1, -1):
        mx = np.max(vec[index:span])
        mn = np.min(vec[index:span])

        close = closes[i + 1]

        diff1 = mx - mn
        diff2 = mn - close
        diff3 = mx - close

        averg = (diff1 + diff2 + diff3) / 3

        atr = (tr * 13 + averg) / 14
        tr = atr

        span -= 1

    return atr
    
if __name__ == "__main__":
    pass
