# -*- coding: utf-8 -*-
"""
Created on Sun Mar 28 23:31:22 2021

@author: kunal

This program creates plots from readings
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.constants import *
plt.rcParams.update({'font.size': 22})

def particle_velocity(kinetic_energy):
    
    #Convert to SI units
    eJoules=kinetic_energy*e
    
    #Calculate the velocity
    velocity=np.sqrt(2*eJoules/m_p)
    
    return velocity

def K_parameter(Lambda):
    
    #log term
    tl=np.log((1+Lambda)/(1-Lambda))
    
    #Term 1
    t1=1/(2*(Lambda**2))
    
    #Term 2
    t2=(1-Lambda**2)/(2*Lambda)
    
    #Calculate the parameter
    K=t1*(t2*tl-1)
    
    return K

def alpha_paramter(scale_length,velocity,cyc_freq):
    
    alpha=velocity/(cyc_freq*scale_length)
    
    return alpha

def cyclotron_freq(magnetic_field):
    
    #For protons
    cyc_freq=-e*magnetic_field/m_p
    
    return cyc_freq

#%%
# =============================================================================
# Calculate the relaxation time
# =============================================================================

#Predetermined parameters
kinetic_energy=20*1e3 #eV
Lambda=1/np.sqrt(2)
A=4
L=0.5

#Derived parameters
K=K_parameter(Lambda)
vel=particle_velocity(kinetic_energy)

#Magnetic Field Array
bArr=np.linspace(0.2,1,100)

#Relaxation Time Array
tArr=[]

for B in bArr:
    
    cyc_freq=cyclotron_freq(B)
    alpha=alpha_paramter(L,vel,cyc_freq)
    
    t=((Lambda*L)/(A*vel))*np.exp(2*K/alpha)
    tArr.append(t)
    
tArr=np.array(tArr)

#%%
# =============================================================================
# Make the plot
# =============================================================================

plt.figure(figsize=(10,12))
plt.plot(bArr,tArr)
plt.grid(True)
plt.xlabel(r'$B_0$ [T]')
plt.ylabel(r'$t_{\mu}$ [s]')
plt.yscale('log')
plt.title('Confinement Time')
plt.show()