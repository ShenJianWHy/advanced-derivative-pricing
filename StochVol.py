#!/usr/bin/env python3
# -*- coding: utf-8 -*-


from scipy.stats import lognorm
import matplotlib.pyplot as plt
import numpy as np
import scipy.integrate as integrate
import AdvancedDerivatives1 as ad

def log_pdf(x,mu,sig):
    return np.exp(-(np.log(x)-mu)**2/2/sig/sig)/x/sig/np.sqrt(2*np.pi)

def call_option_logsv(S,r,b,t,T,K,v_mean,v_sig2):
    def sv_int(v):
        v_mean2 = v_mean * v_mean
        sig2 = np.log( 1 + v_sig2 / v_mean2 ) 
        mu = np.log( v_mean2/np.sqrt(v_mean2+v_sig2))
        sigma = np.sqrt(sig2)
        return log_pdf(v,mu,sigma) * ad.BlackScholes(S,r,b,np.sqrt(v),t,T,K)
    
    return integrate.quad(sv_int,0.0,np.inf)[0]

def binary_call_logsv(S,r,b,t,T,K,v_mean,v_sig2):
    def sv_int(v):
        v_mean2 = v_mean * v_mean
        sig2 = np.log( 1 + v_sig2 / v_mean2 ) 
        mu = np.log( v_mean2/np.sqrt(v_mean2+v_sig2))
        sigma = np.sqrt(sig2)
        return log_pdf(v,mu,sigma) * ad.black_scholes_digital(S,r,b,np.sqrt(v),t,T,K)
    
    return integrate.quad(sv_int,0.0,np.inf)[0]



S = 100
r = 0.02
q = 0.01
b = r-q
t=0
T = 1
K = 100
sigma = 0.2
# Create model, derivative and pricer
model = ad.BlackScholesModel(S=S, r=r, q=q, sigma=sigma, t=t)
deriv = ad.Vanilla( T=T, K=K, Type='Call')
p = ad.BlackScholesPricer(model,deriv)

# SV parameters
v_mean = 0.048
v_sig = 0.05*np.sqrt(T)
v_sig2 = v_sig*v_sig

# Calculate the lognormal distribution parameters
v_mean2 = v_mean * v_mean
sig2 = np.log( 1 + v_sig2 / v_mean2 ) 
mu = np.log( v_mean2/np.sqrt(v_mean2+v_sig2))
sig = np.sqrt(sig2)

# Plot the variance distribution
size = 1000    
x = np.linspace(0,.08,size)
y = np.zeros(size)
for i in range(size):
     y[i] = log_pdf(x[i],mu,sig)   
plt.plot(x,y)
plt.xlabel('Average Variance')
plt.ylabel('Density')
plt.title('$\\mu_V=$%.3f and $\\sigma^2_V$=%.3f'%(v_mean,v_sig))
#plt.savefig('LogSVdist4.pdf')
plt.show()
plt.close()



# PLot the volatility smile
size2 = 20
strike = np.linspace(80,120,size2)
imp_vol = np.zeros(size2)
for i in range(size2):
    svp = call_option_logsv(S, r, b, t, T, strike[i], v_mean, v_sig2)
    p.d.data['Strike'] = strike[i]
    imp_vol[i] = p.implied_vol(svp)
 
plt.plot(strike,imp_vol)
plt.xlabel('Strike')
plt.ylabel('Implied Vol')
plt.title('LogVariance StochVol')
#plt.savefig('LogSVsmile4.pdf')
plt.show()
plt.close()

# PLot the binary options
size3 = 20
K2 = np.linspace(80,120,size3)
dig_bsm = np.zeros(size3)
dig_lsv = np.zeros(size3)
for i in range(size3):
    dig_lsv[i] = binary_call_logsv(S, r, b, t, T, K2[i], v_mean, v_sig2)
    dig_bsm[i] = ad.black_scholes_digital(S,r,b,sigma,t,T,K2[i])
    
plt.plot(K2,dig_lsv,label='SV')
plt.legend()
plt.plot(K2,dig_bsm,label='BSM')
plt.legend()
plt.xlabel('Strike')
plt.ylabel('Binary Price')
plt.title('LogVariance StochVol')
#plt.savefig('LogSVbinary1.pdf')
plt.show()


