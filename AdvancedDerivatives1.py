import numpy as np
import pandas as pd
from scipy.stats import norm
import matplotlib.pyplot as plt
import copy
from scipy import optimize




def BlackScholes(S, r, b, sigma, t, T, K, CallPut="Call", Greek="Price"):
    """
    Prices the BlackScholes Greeks
    """
    tau = T-t
    if tau == 0:
        if Greek=="Price":
            if CallPut=="Call":
                return max(S-K,0)
            else:
                return max(K-S,0)
        elif Greek=="Delta":
            if CallPut=="Call":
                return np.heaviside(S-K,0)
            else:
                return np.heaviside(K-S,0)
        else:
            return np.nan
    else:        
        sqrtTau = np.sqrt(tau)
        vol = sigma * sqrtTau
        d1 = ( np.log(S/K) + (b+0.5*sigma*sigma)*tau ) / vol
        d2 = d1 - vol
        if Greek=="Price":
            if CallPut=="Call":
                return S * np.exp((b-r)*tau) * norm.cdf(d1) - K * np.exp(-r*tau) * norm.cdf(d2)
            else:
                return -S * np.exp((b-r)*tau) * norm.cdf(-d1) + K * np.exp(-r*tau) * norm.cdf(-d2)
        elif Greek=="Delta":
            if CallPut=="Call":
                return np.exp((b-r)*tau) * norm.cdf(d1)
            else:
                return np.exp((b-r)*tau) * (norm.cdf(d1)-1)
        elif Greek=="Gamma":
            return np.exp((b-r)*tau) * norm.pdf(d1) / S / vol
        elif Greek=="Vega":
            return S * np.exp((b-r)*tau) * norm.pdf(d1) * sqrtTau
        elif Greek=="Volga":
            return S * np.exp((b-r)*tau) * norm.pdf(d1) * sqrtTau * d1 * d2 / sigma
        elif Greek=="Theta":
            temp = -0.5 * S * np.exp((b-r)*tau) * norm.pdf(d1) * sigma / sqrtTau
            if CallPut=="Call":
                return temp - (b-r) * S * np.exp((b-r)*tau) * norm.cdf(d1) - r * K * np.exp(-r*tau) * norm.cdf(d2)
            else:
                return temp + (b-r) * S * np.exp((b-r)*tau) * norm.cdf(-d1) + r * K * np.exp(-r*tau) * norm.cdf(-d2)
        elif Greek=="Rho":
            if CallPut=="Call":
                return tau * K * np.exp(-r * tau) * norm.cdf(d2)
            else:
                return -tau * K * np.exp(-r * tau) * norm.cdf(-d2)
        elif Greek=="Rho2":
            if CallPut=="Call":
                return tau * S * np.exp((b-r) * tau) * norm.cdf(d1)
            else:
                return -tau * S * np.exp((b-r) * tau) * norm.cdf(-d1)
        elif Greek=="Fwd":
            return S * np.exp(b*tau)



def black_scholes_digital(S, r, b, sigma, t, T, K, CallPut="Call", Greek="Price"):
    """
    Prices the BlackScholes Greeks
    """
    tau = T-t
    if CallPut == 'Call':
        phi = 1
    else:
        phi = -1
    if tau == 0:
        if Greek=="Price":
            return np.heaviside(phi*(S-K),0)
        elif Greek=="Delta":
            return 0
        else:
            return np.nan
    else:        
        sqrtTau = np.sqrt(tau)
        vol = sigma * sqrtTau
        d1 = ( np.log(S/K) + (b+0.5*sigma*sigma)*tau ) / vol
        d2 = d1 - vol
        if Greek=='Price':
            return np.exp(-r*tau) * norm.cdf(phi*d2)
        elif Greek=="Delta":
            return phi * np.exp(-r*tau) * norm.pdf(d2) / S / vol
        elif Greek=="Gamma":
            return -phi*np.exp(-r*tau) * norm.pdf(d2)* d1 / S / S / vol / vol
        elif Greek=="Vega":
            return -phi * np.exp(-r*tau) * norm.pdf(d2) * d1 / sigma
        elif Greek=="Theta":
            return np.exp(-r*tau) * ( 0.5 * phi * norm.pdf(d2) * (d1 - 2* b * sqrtTau / sigma) / tau + 
                 r * norm.cdf(phi*d2) )
        elif Greek=="Rho":
            return np.exp(-r * tau) * ( phi * sqrtTau * norm.pdf(d2) / sigma - tau * norm.cdf(phi*d2) )
        elif Greek=="Fwd":
            return S * np.exp(b*tau)


def black_scholes_onetouch(S, r, b, sigma, t, T, K, CallPut="Call", Greek="Price"):
    """
    Prices the BlackScholes Greeks
    """
    tau = T-t
    if CallPut == 'Call':
        phi = 1
    else:
        phi = -1
    if tau == 0:
        if Greek=="Price":
            return 0
        elif Greek=="Delta":
            return 0
        else:
            return np.nan
    else:        
        sqrtTau = np.sqrt(tau)
        vol = sigma * sqrtTau
        d1 = ( np.log(S/K) + (b+0.5*sigma*sigma)*tau ) / vol
        d2 = d1 - vol
        x1 = ( np.log(K/S) - (b-0.5*sigma*sigma)*tau ) / vol
        x2 = ( np.log(K/S) + (b-0.5*sigma*sigma)*tau ) / vol
        f = (K/S)**(2*b/sigma/sigma -1 )
        if Greek=='Price':
            return np.exp(-r*tau) * ( norm.cdf(x1) + f*norm.cdf(x2))
        elif Greek=="Delta":
            return phi * np.exp(-r*tau) * norm.pdf(d2) / S / vol
        elif Greek=="Gamma":
            return -phi*np.exp(-r*tau) * norm.pdf(d2)* d1 / S / S / vol / vol
        elif Greek=="Vega":
            return -phi * np.exp(-r*tau) * norm.pdf(d2) * d1 / sigma
        elif Greek=="Theta":
            return np.exp(-r*tau) * ( 0.5 * phi * norm.pdf(d2) * (d1 - 2* b * sqrtTau / sigma) / tau + 
                 r * norm.cdf(phi*d2) )
        elif Greek=="Rho":
            return np.exp(-r * tau) * ( phi * sqrtTau * norm.pdf(d2) / sigma - tau * norm.cdf(phi*d2) )
        elif Greek=="Fwd":
            return S * np.exp(b*tau)

def trigger_forward_bsm(S,r,b,sigma,t,T,K,H,CallPit='Call',Greek='Price'):
    tau = T-t
    sqrtTau = np.sqrt(tau)
    vol = sigma * sqrtTau
    d1 = ( np.log(S/H) + (b+0.5*sigma*sigma)*tau ) / vol
    d2 = d1 - vol
    return S * np.exp((b-r)*tau) * norm.cdf(d1) - K * np.exp(-r*tau) * norm.cdf(d2)

def forward_start_bsm(S,r,b,sigma,t,T,alpha,CallPit='Call',Greek='Price'):
    tau = T-t
    sqrtTau = np.sqrt(tau)
    vol = sigma * sqrtTau
    d1 = ( np.log(1/alpha) + (b+0.5*sigma*sigma)*tau ) / vol
    d2 = d1 - vol
    return S * np.exp((b-r)*t) *( np.exp((b-r)*tau) * norm.cdf(d1) - alpha * np.exp(-r*tau) * norm.cdf(d2))


def barrier_option_AB(S, r, b, vol, T, K, x, phi, eta):    
    return phi * S * np.exp((b-r)*T) * norm.cdf(phi*x) - phi*K*np.exp(-r*T)*norm.cdf(phi*x-phi*vol)   
def barrier_option_CD(S, r, b, vol, T, H, K, y, mu, phi, eta):
    return phi * S * np.exp((b-r)*T)* (H/S)**(2*(mu+1))*norm.cdf(eta*y) - phi*K*np.exp(-r*T)*(H/S)**(2*mu) * norm.cdf(eta*y-eta*vol)

def barrier_option_bsm(S, r, b, sigma, t, T, K, H, CallPut="DICall", Greek="Price"):
    tau = T-t
    sqrtTau = np.sqrt(tau)
    vol = sigma * sqrtTau
    sigma2 = sigma * sigma
    
    mu =  ( b - 0.5 * sigma2 ) / sigma2
    lam = np.sqrt( mu*mu + 2 * r / sigma2 )
    
    x1 = np.log(S/K) / vol + ( 1 + mu ) * vol
    x2 = np.log(S/H) / vol + ( 1 + mu ) * vol

    y1 = np.log(H*H/S/K) / vol + ( 1 + mu ) * vol
    y2 = np.log(H/S) / vol + ( 1 + mu ) * vol
    
    z = np.log(H/S) / vol + lam * vol

    if CallPut=='DICall':
        if K > H:
            C = barrier_option_CD(S, r, b, vol, T, H, K, y1, mu, 1, 1)
            return C
        else:
            A = barrier_option_AB(S, r, b, vol, T, K, x1, 1, 1)
            B = barrier_option_AB(S, r, b, vol, T, K, x2, 1, 1)
            D = barrier_option_CD(S, r, b, vol, T, H, K, y2, mu, 1, 1)
            return A - B + D
    if CallPut=='UOCall':
        if K > H:
            return 0
        else:
            eta = -1
            phi = 1
            A = barrier_option_AB(S, r, b, vol, T, K, x1, phi, eta)
            B = barrier_option_AB(S, r, b, vol, T, K, x2, phi, eta)
            C = barrier_option_CD(S, r, b, vol, T, H, K, y1, mu, phi, eta)          
            D = barrier_option_CD(S, r, b, vol, T, H, K, y2, mu, phi, eta)


            return A - B + C - D
        





def perpetual_american_bsm(S, H, r, sigma, greek='Price'):
    if S < H:
        if greek=='Price':
            return S / H
        if greek=='Delta':
            return 1 / H
        if greek=='Gamma':
            return 0
        if greek=='Vega':
            return 0
        if greek=='Theta':
            return 0
        if greek=='Rho':
            return 0
    # S > H
    else:
        if greek=='Price':
            return (S / H)**(-2*r/sigma/sigma) 
        if greek=='Delta':
            return -2*r/sigma/sigma / H * (S / H)**(-2*r/sigma/sigma-1)
        if greek=='Gamma':
            return (2*r/sigma/sigma+1) * 2*r/sigma/sigma / H**2 * (S / H)**(-2*r/sigma/sigma-1)
        if greek=='Theta':
            return 0
        if greek=='Rho':
            return -2/sigma**2*np.log(S/H) * perpetual_american_bsm(S, H, r, sigma, greek='Price')
        if greek=='Vega':
            return 2*r/sigma**3*np.log(S/H) * perpetual_american_bsm(S, H, r, sigma, greek='Price')




#########################################################################################################
# Function to add teh option parameters to the plot
def display_option_params(ax, pricer, display_loc, x_label=None, y_label=None):
    if x_label != None:
        ax.set_xlabel(x_label)
    if y_label != None:
        ax.set_ylabel(y_label)
        
    yval = np.linspace(display_loc[1],display_loc[1]-0.25,5)
    ax.text(display_loc[0], yval[0], 'K='+str(pricer.d.data['Strike']), transform=ax.transAxes, fontsize=10)
    ax.text(display_loc[0], yval[1], '$\sigma$='+'%.3f' % pricer.m.data['Sigma'], transform=ax.transAxes, fontsize=10)
    ax.text(display_loc[0], yval[2], 'T='+'%.3f' % pricer.d.data['Expiry'], transform=ax.transAxes, fontsize=10)
    ax.text(display_loc[0], yval[3], 'r='+str(pricer.m.data['Rate']), transform=ax.transAxes, fontsize=10)
    ax.text(display_loc[0], yval[4], 'q='+str(pricer.m.data['Div']), transform=ax.transAxes, fontsize=10)
    



###############################
# Class for holding market data for a Black Scholes model        
class BlackScholesModel:
    def __init__(self, S=100, r=0.02, q=0.01, sigma=.2, t=0):
        self.data = {}
        self.data['Spot'] = S
        self.data['Rate'] = r
        self.data['Div'] = q
        self.data['Sigma'] = sigma
        self.data['Today'] = t
        self.update_derived_vals()

    def update_derived_vals(self):
        self.drift = self.data['Rate']-self.data['Div']
        self.log_spot = np.log(self.data['Spot']) 
        self.log_drift = self.drift - 0.5 * self.data['Sigma']**2        

       
###############################
# Base class for defining a derivative        
###############################
class Derivative:
    def __init__(self, T=1, K=100, Type="Call"):
        self.data = {}
        self.data['Expiry'] = T
        self.data["Strike"] = K
        self.data["Type"] = Type


###############################
# Returns the undiscounted payoff based on an asset value S        
    def payoff(self,S):
        return S

###############################
# Returns the undiscounted payoff based on an asset value S        
    def black_scholes(self, S, r, b, sigma, t, T, K, Type, greek):
        return S

###############################
# Derived class that implements a vanilla option        
###############################
class Vanilla(Derivative):       
    def payoff(self,S):       
        P = (S - self.data['Strike'])
        P[P<0] = 0
        return P        

    def black_scholes(self, S, r, b, sigma, t, T, K, Type, greek):
        return BlackScholes(S, r, b, sigma, t, T, K, Type, greek)
    

###############################
# Derived class that implements a digital option        
###############################
class Digital(Derivative):
    def __init__(self, T=1, K=100, Type="Call"):
        self.data = {}
        self.data["Expiry"] = T
        self.data["Strike"] = K
        self.data["Type"] = Type

    def payoff(self,S):       
        P = np.heaviside(S - self.data['Strike'],0)
        return P        

    def black_scholes(self, S, r, b, sigma, t, T, K, Type, greek):
        return black_scholes_digital(S, r, b, sigma, t, T, K, Type, greek)


###############################
# Derived class that implements a digital option        
###############################
class Onetouch(Derivative):
    def __init__(self, T=1, K=100, Type="Call"):
        self.data = {}
        self.data["Expiry"] = T
        self.data["Strike"] = K
        self.data["Type"] = Type

    def payoff(self,S):       
        P = np.heaviside(S - self.data['Strike'],0)
        return P        

    def black_scholes(self, S, r, b, sigma, t, T, K, Type, greek):
        return black_scholes_onetouch(S, r, b, sigma, t, T, K, Type, greek)


###############################
# Derived class that implements a perpetual American binary
###############################
class PerpetualAmerican(Derivative):
    def __init__(self, H=90):
        self.data = {}
        self.data['Barrier'] = H
        self.data['Expiry'] = 1.0
        self.data['Strike'] = H
        self.data['Type'] = 'Call'
        
    def payoff(self,S):       
        return 1        

    def black_scholes(self, S, r, b, sigma, t, T, K, Type, greek):
        return perpetual_american_bsm(S, K, r, sigma, greek)


###############################
# Derived class that implements a barrier option        
###############################
class BarrierOption(Derivative):
    def __init__(self, T=1, K=100, H=90, Type='DICall'):
        self.data = {}
        self.data['Barrier'] = H
        self.data['Expiry'] = T
        self.data['Strike'] = K
        self.data['Type'] = Type
        
    def payoff(self,S):       
        return 1        

    def black_scholes(self, S, r, b, sigma, t, T, K, Type, greek):
        return barrier_option_bsm(S, r, b, sigma, t, T, K, self.data['Barrier'], Type, greek)



###############################
# Class for combining a Black-Scholes model and a derivative        
###############################
class BlackScholesPricer():
    sens_dict = { 'Delta': ['Spot'], 'Vega': ['Sigma'], 'Rho': ['Rate'], 'Theta': ['Today']}
    
    def __init__(self, black_scholes_model, derivative):
        self.m = black_scholes_model
        self.d = derivative
        self.update_derived_vals()
        self.greek = "Price"

    def print_greeks(self):
        greeks = {}
        greeks['Fwd'] = self.price(greek='Fwd')
        greeks['Price'] = self.price()
        greeks['Delta'] = self.price(greek='Delta')
        greeks['Gamma'] = self.price(greek='Gamma')
        greeks['Theta'] = self.price(greek='Theta')
        greeks['Vega'] = self.price(greek='Vega')
        greeks['Rho'] = self.price(greek='Rho')
        for g in greeks:
            print(g+' = %.15f' % greeks[g])
        return greeks


    def update_derived_vals(self):
        self.m.update_derived_vals()
        self.tau = self.d.data['Expiry'] - self.m.data['Today']
        self.sigma_root_tau = self.m.data['Sigma'] * np.sqrt(self.tau)
        self.log_drift_tau = self.m.log_drift * self.tau
        self.discount = np.exp(-self.m.data['Rate'] * self.tau)

    def implied_vol(self,fixed_price):
        self.fixed_price = fixed_price
        imp_vol = optimize.root_scalar( self.implied_vol_pricer, bracket=[0.0001,10])
        return imp_vol.root
    
    def implied_vol_pricer(self,sigma):
        return self.price(sigma=sigma) - self.fixed_price        

    def sensitivity(self, S, P, bump):
        return ( self.discount * self.d.payoff(S) - P ) / bump

    def bump(self, key, bump, is_mult_bump=True):
        for x in self.sens_dict[key]:
            if is_mult_bump:
                self.m.data[x] *= bump
            else:
                self.m.data[x] += bump
        self.update_derived_vals()
        
    def get_bump(self, key, bump, is_mult_bump):
        if is_mult_bump:
            x = self.sens_dict[key][0]
            return self.m.data[x] * (bump-1)  
        else:
            return bump                
        
    def price_key(self, key, value):
        self.m.data[key] = value
        return self.price()
    
    def price(self, S=None, r=None, b=None, sigma=None, t=None, T=None, K=None, Type=None, greek=None):
        if S==None:
            S = self.m.data['Spot']
        if r==None:
            r = self.m.data['Rate']
        if b==None:
            b = self.m.drift
        if sigma==None:
            sigma = self.m.data['Sigma']
        if t==None:
            t = self.m.data['Today']
        if T==None:
            T = self.d.data['Expiry']
        if K==None:
            K = self.d.data['Strike']
        if Type==None:
            Type = self.d.data['Type']
        if greek==None:
            greek = self.greek
        return self.d.black_scholes(S, r, b, sigma, t, T, K, Type, greek)
    
    def plot(self, key, xmin, xmax, xpoints, greek="Price", title=None, xlabel=None, ylabel=None, display_vals=False, display_loc=[0.05,0.95], filename=None):
        if greek=='Spread':
            self.greek='Price'
        else:
            self.greek = greek
        if title==None:
            title=greek
        if xlabel==None:
            xlabel=key
        if ylabel==None:
            ylabel="Derivative Value"
        x = np.linspace(xmin, xmax, xpoints)
        y = [ self.price_key(key,val) for val in x ]
        if greek=="Spread":
            strike = self.d.data['Strike']
            self.d.data['Strike'] = strike * 0.95   
            y2 = [ self.price_key(key,val) for val in x ]
            self.d.data['Strike'] = strike
            for i in range(xpoints):
                y[i] = ( y2[i] - y[i] ) / 0.05 / strike            
        ax = pd.DataFrame(index=x, data=y).plot(legend=False, title=title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        if display_vals:
            yval = np.linspace(display_loc[1],display_loc[1]-0.25,5)
            ax.text(0.05, yval[0], 'K='+str(self.d.data['Strike']), transform=ax.transAxes, fontsize=10)
            ax.text(0.05, yval[1], 'sigma='+str(self.m.data['Sigma']), transform=ax.transAxes, fontsize=10)
            ax.text(0.05, yval[2], 'T='+str(self.d.data['Expiry']), transform=ax.transAxes, fontsize=10)
            ax.text(0.05, yval[3], 'r='+str(self.m.data['Rate']), transform=ax.transAxes, fontsize=10)
            ax.text(0.05, yval[4], 'q='+str(self.m.data['Div']), transform=ax.transAxes, fontsize=10)
        if filename!=None:
            plt.savefig(filename+'.pdf')

    def plot2(self, key, xmin, xmax, xpoints, key2, xmin2, xmax2, xpoints2, greek="Price", title=None, xlabel=None, ylabel=None, display_vals=False, display_loc=[0.05,0.95], filename=None):
        if greek=='Spread':
            self.greek='Price'
        else:
            self.greek = greek
        if title==None:
            title=greek
        if xlabel==None:
            xlabel=key
        if ylabel==None:
            ylabel="Derivative Value"
        x = np.linspace(xmin, xmax, xpoints)
        ax = plt.subplot()
        for x2 in np.linspace(xmin2, xmax2, xpoints2):
            self.d.data[key2] = x2
            y = [ self.price_key(key,val) for val in x ]
            pd.DataFrame(index=x, data=y).plot(ax=ax, legend=False, title=title, label=key2)
            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)
        if display_vals:
            yval = np.linspace(display_loc[1],display_loc[1]-0.25,5)
            ax.text(0.05, yval[0], 'K='+str(self.d.data['Strike']), transform=ax.transAxes, fontsize=10)
            ax.text(0.05, yval[1], 'sigma='+str(self.m.data['Sigma']), transform=ax.transAxes, fontsize=10)
            ax.text(0.05, yval[2], 'T='+str(self.d.data['Expiry']), transform=ax.transAxes, fontsize=10)
            ax.text(0.05, yval[3], 'r='+str(self.m.data['Rate']), transform=ax.transAxes, fontsize=10)
            ax.text(0.05, yval[4], 'q='+str(self.m.data['Div']), transform=ax.transAxes, fontsize=10)
        plt.show()
        if filename!=None:
            plt.savefig(filename+'.pdf')

        
# Function to update the pricer based on a change in logS and t
    def update(self, logS, t):
        self.m.data['Spot'] = np.exp(logS)
        self.m.data['Today'] = t
        self.update_derived_vals()
        
        