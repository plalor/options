import numpy as np
from scipy.stats import norm

def valueAtExpiration(strikePrice, stockValue):
    diff = np.array(stockValue - strikePrice)
    return np.maximum(diff, 0)

def BlackScholes(strikePrice, stockValue, daysToExp, volatility=0.17):
    if daysToExp <= 0:
        return valueAtExpiration(strikePrice, stockValue) 
    S = stockValue
    K = strikePrice
    t = daysToExp/365
    sigma = volatility
    r = 0.03
    d1 = (np.log(S/K) + (r + sigma**2/2)*t) / (sigma*np.sqrt(t))
    d2 = d1 - sigma*np.sqrt(t)
    C = norm.cdf(d1)*S - norm.cdf(d2)*K*np.exp(-r*t)
    return C