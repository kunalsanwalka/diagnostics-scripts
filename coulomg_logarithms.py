# -*- coding: utf-8 -*-
"""
Created on Wed Aug 17 12:06:38 2022

@author: kunal
"""

import numpy as np
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 22})

def lamba_ii_cary(dens,te):
    
    lambda_ii=31-0.5*np.log(dens)+np.log(te)
    
    return lambda_ii

def lambda_ii_nrl(dens,ti):
    
    lambda_ii=23-0.5*np.log(dens)+1.5*np.log(ti)
    
    return lambda_ii

def lambda_ei_1(dens,te):
    
    lambda_ei=23-np.log((dens**0.5)*(te**-1.5))
    
    return lambda_ei

def lambda_ei_2(dens,te):
    
    lambda_ei=24-np.log((dens**0.5)/te)
    
    return lambda_ei

def lambda_ei_3(dens,ti):
    
    lambda_ei=30-np.log((dens**0.5)*(ti**-1.5)/2.5)
    
    return lambda_ei

dens=4*1e14 #cm^{-3}
te=np.linspace(3e4,10e4,100) #eV
ti=np.linspace(3e5,10e5,100) #eV

# =============================================================================
# lambda_ii
# =============================================================================

caryArr=lamba_ii_cary(dens, te)
nrlArr=lambda_ii_nrl(dens, ti)

fig,ax=plt.subplots(figsize=(12,8))
plt.ticklabel_format(axis='x',style='sci',scilimits=(0,0))

ax.plot(te,caryArr,color='Blue')
ax.tick_params(axis='x', labelcolor='Blue')
ax.set_xlabel(r'Cary; Te [eV]',color='Blue')

ax2=ax.twiny()
ax2.plot(ti,nrlArr,color='Red')
ax2.tick_params(axis='x', labelcolor='Red')
ax2.set_xlabel(r'NRL; Ti [eV]',color='Red')

ax.set_ylabel(r'$\lambda_{ii}$')

plt.show()

# =============================================================================
# lambda_ei
# =============================================================================

arr1=lambda_ei_1(dens, te)
arr2=lambda_ei_2(dens, te)
arr3=lambda_ei_3(dens, ti)

fig,ax=plt.subplots(figsize=(12,8))
plt.ticklabel_format(axis='x',style='sci',scilimits=(0,0))

plt.plot(te,arr1,label='1')
plt.plot(te,arr2,label='2')
plt.plot(ti,arr3,label='3')

plt.grid()

plt.show()