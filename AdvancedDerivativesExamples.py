#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import AdvancedDerivatives1 as ad
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#####################
# Runner
S = 100
K = 100
t = 0
T = 1
r = 0.02
q = 0.01
b = r-q
sigma = 0.2


# Create model, derivative and pricer
model = ad.BlackScholesModel(S=S, r=r, q=q, sigma=sigma, t=t)
deriv = ad.Vanilla( T=T, K=K, Type='Call')
p = ad.BlackScholesPricer(model,deriv)

# Print the greeks
p.print_greeks()

# Graph price as a function of spot
size = 50
x = np.linspace(80,120,size)
y = np.zeros(size)
for i in range(size):
    y[i] = ad.BlackScholes(x[i], r, b, sigma, t, T, K)

plt.plot(x,y)

