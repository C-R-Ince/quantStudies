import numpy as np
from scipy.stats import norm

s = 1.2
r = 0.02
d = 0.01
T = 0.5
X = 1.25
vol = 0.25
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
