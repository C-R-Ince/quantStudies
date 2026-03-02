import numpy as np
import argparse
from scipy.stats import norm

#Function to check decimalised percentage
def decimalPercentage(value):
    try:
        value = float(value)
    except ValueError:
        raise argparse.ArgumentTypeError(f"{value} is not a valid number")
    
    if not 0 <= value <= 1:
        raise argparse.ArgumentTypeError(
            f"{value} is not a decimalised percentage (must be between 0 and 1)"
        )
    return value
    
#Function to check for a positive float 
def positiveStrict(value):
    try:
        value = float(value)
    except ValueError:
        raise argparse.ArgumentTypeError(f"{value} is not a valid number")
    if value <= 0:
        raise argparse.ArgumentTypeError(f"{value} must be strictly positive")
    return value
    
#Validates the rate
def rateValidator(value):
    try:
        value = float(value)
    except ValueError:
        raise argparse.ArgumentTypeError(f"{value} is not a valid number")
    if not -1 <= value <= 1:
        raise argparse.ArgumentTypeError(
            f"{value} must be between -1 and 1 (decimal form)"
        )
    return value

#Define arguments
parser = argparse.ArgumentParser(description="NPricing commodities coffee")
parser.add_argument("-s", "--spotPrice",
                    required=True,
                    type=positiveStrict,
                    help="Current price of coffee")
parser.add_argument("-r", "--rate",
                    required=True,
                    type=rateValidator,
                    help="Current risk free rate as a decimalised percentage per annum")
parser.add_argument("-d", "--storage",
                    required=True,
                    type=decimalPercentage,
                    help="Current cost of storage as a decimalised percentage per annum")
parser.add_argument("-T", "--timeToMaturity",
                    required=True,
                    type=positiveStrict,
                    help="Time for contract to mature in months")
parser.add_argument("-X", "--strikePrice",
                    required=True,
                    type=positiveStrict,
                    help="Strike price for the commodity")
parser.add_argument("-v", "--volatility",
                    required=True,
                    type=positiveFloat,
                    help="Volitility as a decimilised percentage")

#10000 simulations
n_sim = 10000
     
#Cost of Carry
def costOfCarry(s, r , d ,T):
    return s*np.exp((r+d)*T)

F = costOfCarry(s, r, d, T)
print("Cost of Carry: ", round(F, 4))

#Black-Scholes Model
def blackScolesCall(s, X, r, d, vol, T):
    d1 = (np.log(s/X) + (r - d + 0.5*vol**2)*T) / (vol*np.sqrt(T))
    d2 = d1 - vol*np.sqrt(T)
    
    callPrice = (s*np.exp(-d*T)*norm.cdf(d1) -
                  X*np.exp(-r*T)*norm.cdf(d2))
    return callPrice

bsPrice = blackScolesCall(s, X, r, d, vol, T)
print("Black-Scholes Call Price:", round(bsPrice, 4))

#Monte Carlo
def monteCarlo(s, X, r, d, vol, T, n_sim):
    Z = np.random.standard_normal(n_sim)
    
    ST = s * np.exp((r - d - 0.5*vol**2)*T +
                     vol*np.sqrt(T)*Z)
    
    payoff = np.maximum(ST - X, 0)
    
    callPrice = np.exp(-r*T) * np.mean(payoff)
    return callPrice

mcPrice = monteCarlo(s, X, r, d, vol, T, n_sim)
print("Monte Carlo Call Price:", round(mcPrice, 4))
