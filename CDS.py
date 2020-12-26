#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import numpy as np
import pandas as pd

mat = 5
hazard = 0.02
r = 0.05
index = np.array(range(1,6,1))
recovery = 0.4
cpn_dates = index
def_date = index - 0.5


p = pd.DataFrame(index=index)
p['Cumulative Survival'] = [ np.exp(-hazard * index[i]) for i in range(mat) ]
p['Cumulative Default'] = 1 - p['Cumulative Survival']
p['Annual Default'] = p['Cumulative Default'].diff()
p['Annual Default'].iloc[0] = p['Cumulative Default'].iloc[0]
p['Discount'] = [ np.exp(-r * cpn_dates[i]) for i in range(mat) ]
p['Risky Discount']  = p['Discount']   * p['Cumulative Survival']
p['Recovery'] = recovery
p['Default DF'] = [ np.exp(-r * def_date[i]) for i in range(mat ) ]
p['Default PV'] = p['Default DF'] * (1- p['Recovery']) * p['Annual Default']
p['Cpn Accrual'] = p['Annual Default'] * p['Default DF'] * 0.5
print(p.transpose())    

x = p['Risky Discount'].sum()
print(x)
y = p['Cpn Accrual'].sum()
print(y)
z = p['Default PV'].sum()
print(z)  
spd = z / ( x + y )
print(spd)