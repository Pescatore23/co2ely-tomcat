# -*- coding: utf-8 -*-
"""
Created on Mon Apr 24 15:12:25 2023

@author: fische_r
"""

import numpy as np
import scipy as sp
import scipy.optimize
import scipy.interpolate
from matplotlib import cm

# import robpylib
# fun = robpylib.CommonFunctions.functions.power_law
def power_law(x, a,b,c):
    y = a*x**b+c
    return y

fun = power_law

thermopath = r"U:\01_Python\00_playground\water_thermo.txt"

def extract_therm_props(thermopath):
    with open(thermopath) as f:
        lines = f.readlines()
        lines = lines[1:]

    Tx = np.zeros(len(lines))
    pvapx = Tx.copy()
    dHmolx = Tx.copy()
    dHspecx = Tx.copy()
    
    i = 0
    for line in lines:
        test = line.split(' \t')
        Tx[i] = float(test[0])
        pvapx[i] = float(test[1])
        dHmolx[i] = float(test[2])
        dHspecx[i] = float(test[-1][:-1])
        i = i+1
        
    pvap_fun = sp.interpolate.interp1d(Tx, pvapx) #vapor pressure [kPa]
    dHmol_fun = sp.interpolate.interp1d(Tx, dHmolx) #molar latent heat of evaporation [J/mol]
    dHspec_fun = sp.interpolate.interp1d(Tx, dHspecx) #specific latent heat of evap [kJ/kg]
    
    return pvap_fun, dHmol_fun, dHspec_fun



pvap_fun, dHmol_fun, dHspec_fun = extract_therm_props(thermopath) #get the water vapor pressure at a given temperature



#default parameters
N_A = 6.0221415E23 # mol-1 Avogadro constant
R = 8.314472# gas constant m3⋅Pa⋅K−1⋅mol−1
e = 1.602176634E-19 #C C[1]
D2 = 3E-9 # m2/s diffusion coeff of CO2 in H2O  at atmospheric pressure, maybe not reliable value
D1 = 1.06E-5 # m2/s diffusion coeff of CO2 in CO2  at atmospheric pressure, maybe not reliable value
# extrapolated for 45°C from to https://www.engineeringtoolbox.com/diffusion-coefficients-d_1404.html
# https://www.sciencedirect.com/science/article/pii/S0016703713002251 shows a slight decrease for higher pressures and stronger increase for higher temperatures
T = 273+45 #K, temperature 45°C
p_gas = 101325 # Pa, pressure of CO2 at cathode = ambient pressure
H_CO2 = 0.82

#thicknesses
d12 = 8E-6#m , cathode CL
d23 = 60E-6#m, AEM thickness
d34 = 60E-6#m, CEM thickness
d45 = 10e-6#m cathode CL

#volume fractions transpport medium
e12 = 0.5
e23 = 0.36
e34 = 0.36
e45 = 0.39

#tortuosities transport medium
t12 = 1
t23 = 1.5
t34 = 1.5
t45 = 2.8

f_CO2 = 1/3 #1/3 # conversion of electron to CO2, depends on faradeic efficiency, type of ion (CO3/HCO3) and CO2+OH-->HCO3 rate

def calculate_interface_pressure(J, f_CO2 = f_CO2):
    p_vap = pvap_fun(T)*1E-3
    p_CO2 = p_gas-p_vap
    J = J*1E-3/0.01**2 #convert current density from mA/cm2 to A/m2
    n = J/e/N_A*f_CO2 #mol/m2/s
    
    # diffusivities
    D12 = D1*e12/t12/d12
    D23 = D2*e23/t23/d23
    D34 = D2*e34/t34/d34
    D45 = D2*e45/t45/d45
    
    D13 = D12*D23/(D12+D23)
    D35 = D34*D45/(D34+D45)
    
    #interface pressure
    p3 = (R*T*n+D13*p_CO2*H_CO2)/(D13+D35)/H_CO2 #Pa
    
    #cathode flux
    j13 = D13*(p3-p_CO2)/R/T #mol/m2/s
    
    # anode flux
    j35 = D35*p3/R/T #mol/m2/s
    
    #pressure drop across cathode CL
    dp12 = R*T*j13/D12 #Pa
    
    #pressure drop across anode CL
    dp45 = R*T*j35/D45 #Pa
    
    return p3, j13, j35, dp12, dp45


def flux_per_curr(currs, f = f_CO2, varf = False):
    
    j_an = np.zeros(len(currs))
    j_cat = np.zeros(len(currs))
    p_an = np.zeros(len(currs))
    p_cat = np.zeros(len(currs))
    p_cent = np.zeros(len(currs))
    
    f_I = np.array([0.5, 0.5, 0.9*0.5, 0.6*0.5])
    If = np.array([60,100,500,1000])
    pfit, cov = sp.optimize.curve_fit(fun, If, f_I)
    
    for i in range(len(currs)):
        curr = currs[i]
        if varf:
            p, j1, j2, d1, d2 = calculate_interface_pressure(curr, f_CO2 = fun(curr, *pfit))
        else:
            p, j1, j2, d1, d2 = calculate_interface_pressure(curr, f_CO2 = f_CO2)
        j_an[i] = j2
        j_cat[i] = j1
        p_an[i] = d2
        p_cat[i] = d1
        p_cent[i] = p
    
    return j_cat, j_an, p_cat, p_an, p_cent
    
    
import matplotlib.pyplot as plt 
import tikzplotlib  
import os

paperplotpath = '/mpc/homes/fische_r/Paper/05-bpm-delamination/plots'
paperplotpath = r"C:\git_repos\02_Paper\05_BPM_delamination\plots"
currs = np.arange(10,1500)  
j_cat, j_an, d_cat, d_an, p_cent = flux_per_curr(currs)
j_cat2, j_an2, d_cat2, d_an2, p_cent2 = flux_per_curr(currs, varf = True)

NAE = N_A*e

fig, ax = plt.subplots()
ax.plot(currs, j_cat*0.01**2/currs*1000*NAE,'k', label='cathode')
ax.plot(currs, j_an*0.01**2/currs*1000*NAE,'r', label='anode')
ax.set_xlabel('I [$mA/cm^2$]')
ax.set_ylabel('$CO_2/e^-$')


ax.plot(currs, j_cat2*0.01**2/currs*1000*NAE,'k--', label = 'fitted ion composition')
ax.plot(currs, j_an2*0.01**2/currs*1000*NAE,'r--')

ax.legend()

ax2 = ax.twinx()
ax2.plot(currs, j_cat*0.01**2/currs*1000,'k', label='cathode')
ax2.plot(currs, j_an*0.01**2/currs*1000,'r', label='anode')
ax2.set_ylabel('j/I [$mol/As$]')

ax2.plot(currs, j_cat2*0.01**2/currs*1000,'k--', label='cathode')
ax2.plot(currs, j_an2*0.01**2/currs*1000,'r--', label='anode')


ax.set_xlim(0,200)
ax.set_ylim(0)
ax2.set_xlim(0,200)
ax2.set_ylim(0)

# tikzplotlib.save(os.path.join(paperplotpath, 'model_CO2_per_charge_varf.tex'))

# fig, ax = plt.subplots()
# test = []
# fs = np.arange(0.1,1.3,0.2)
# for i in range(len(fs)):
#     j_cat, j_an, d_cat, d_an = flux_per_curr(currs, f=fs[i])
    
#     test.append((j_cat, j_an, d_cat, d_an))
    
#     ax.plot(currs, j_cat*0.01**2/currs*1000*NAE,color=cm.tab10(i), label=fs[i])
#     ax.plot(currs, j_an*0.01**2/currs*1000*NAE,color=cm.tab10(i), linestyle = '--')
#     ax.set_xlabel('I [$mA/cm^2$]')
#     ax.set_ylabel('$CO_2/e^-$')
#     # ax.legend()

#     # ax2 = ax.twinx()
#     # ax2.plot(currs, j_cat*0.01**2/currs*1000,cm.tab10(i), label=fs[i])
#     # ax2.plot(currs, j_an*0.01**2/currs*1000,cm.tab10(i), linestyle = '--')
#     # ax2.set_ylabel('j/I [$mol/As$]')
#     # ax2.set_xlim(0)
#     # ax2.set_ylim(0)
    
# ax.set_xlim(0)
# ax.set_ylim(0)
    
    
    