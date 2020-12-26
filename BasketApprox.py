#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import AdvancedDerivatives1 as ad
import matplotlib.pyplot as plt

def basket_vol(F,rho,sig,T):
    M2 = 0
    for i in range(3):
        for j in range(3):
            M2 += F[i] * F[j]* np.exp(rho[i][j]*sig[i]*sig[j]*T)   
    return np.sqrt(np.log(M2/M1/M1)/T)


rho12 = .8
rho13 = .5
rho23 = .6
T = 1
K = 300
r = 0.02

F = [100,100,100]
sig=[.25,.2,.15]
rho=[ [1,rho12,rho13], [rho12,1,rho23], [rho13,rho23,1]] 

M1 = 0
for i in range(3):
    M1 += F[i]

    
M2 = 0
for i in range(3):
    for j in range(3):
        M2 += F[i] * F[j]* np.exp(rho[i][j]*sig[i]*sig[j]*T)
print(M2)

vol = np.sqrt(np.log(M2/M1/M1)/T)
print(vol)

#def BlackScholes(S, r, b, sigma, t, T, K, CallPut="Call", Greek="Price"):
p = np.exp(-r*T) * ad.BlackScholes(M1,0,0,vol,0,T,300)
print(p)

def basket_pricer(F,rho,sig,r, K,T):
    M1 = 0
    for i in range(3):
        M1 += F[i]
    vol = basket_vol(F, rho, sig, T)
    return np.exp(-r*T) * ad.BlackScholes(M1,0,0,vol,0,T,K)

p2 = basket_pricer(F,rho,sig,r,K,T)
print('p2=%f'%p2)

sig_bump = sig
sig_bump[0] += 0.01
p2_bump = basket_pricer(F,rho,sig_bump,r,K,T)
vega = (p2_bump - p2) / 0.01
print('vega=%f'%vega)
print('p2_bump=%f'%p2_bump)

points = 50
x = np.linspace(-.99,.99,points)
y = np.zeros(points)
for i in range(points):
    rho[1][2] = x[i]
    rho[2][1] = x[i]
    y[i] = basket_vol(F,rho,sig,T)*100
plt.plot(x,y)
plt.xlabel('$\\rho_{12}$')
plt.ylabel('Basket Vol')
plt.title('Basket Vol as a function of Stock Correlation')
#plt.savefig('BasketVol.pdf')