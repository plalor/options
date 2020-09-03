import numpy as np
from scipy.stats import norm

def valueAtExpiration(strikePrice, stockValue):
    diff = np.array(stockValue - strikePrice)
    return np.maximum(diff, 0)

def BlackScholes(strikePrice, stockValue, daysToExp, volatility=0.17, r=0.03):
    if daysToExp <= 0:
        return valueAtExpiration(strikePrice, stockValue) 
    S, K, t, sigma, d1, d2, PV = calcVars(strikePrice, stockValue, daysToExp, volatility, r)
    C = norm.cdf(d1)*S - norm.cdf(d2)*PV
    return C

def GreeksAtExpiration(strikePrice, stockValue, greek, r=0.03):
    greek = greek.lower()
    if greek == "delta":
        return np.heaviside(stockValue - strikePrice, 0.5)
    elif greek == "gamma":
        return np.where(stockValue != strikePrice, 0, np.inf)
    elif greek == "vega":
        return 0.
    elif greek == "theta":
        return np.where(stockValue != strikePrice, 0, -np.inf) - r*strikePrice*np.heaviside(stockValue - strikePrice, 0.5) / 365
    elif greek == "rho":
        return 0.
    else:
        raise ValueError("greek must be 'delta', 'gamma', 'vega', 'theta', or 'rho'")

def Greeks(strikePrice, stockValue, daysToExp, greek, volatility=0.17, r=0.03):
    if daysToExp <= 0:
        return GreeksAtExpiration(strikePrice, stockValue, greek)
    greek = greek.lower()
    S, K, t, sigma, d1, d2, PV = calcVars(strikePrice, stockValue, daysToExp, volatility, r)
    if greek == "delta":
        return norm.cdf(d1)
    elif greek == "gamma":
        return norm.pdf(d1) / (S * sigma * np.sqrt(t))
    elif greek == "vega":
        return S * norm.pdf(d1) * np.sqrt(t) / 100
    elif greek == "theta":
        return (-S*norm.pdf(d1)*sigma / (2*np.sqrt(t)) - r*PV*norm.cdf(d2)) / 365
    elif greek == "rho":
        return t*PV*norm.cdf(d2) / 10000
    else:
        raise ValueError("greek must be 'delta', 'gamma', 'vega', 'theta', or 'rho'")
        
def calcVars(strikePrice, stockValue, daysToExp, volatility, r):
    S = stockValue
    K = strikePrice
    t = daysToExp/365
    sigma = volatility
    r = 0.03
    d1 = (np.log(S/K) + (r + sigma**2/2)*t) / (sigma*np.sqrt(t))
    d2 = d1 - sigma*np.sqrt(t)
    PV = K*np.exp(-r*t)
    return S, K, t, sigma, d1, d2, PV