# -*- coding: utf-8 -*-
"""
Created on Thu Jul 29 10:33:30 2021

@author: kunal

This programs calculates the full wave solution for the HHFW.

Here, the plasma is treated as a dielectric (cold plasma dispersion relation).

Calculations are heavily inspired by- https://farside.ph.utexas.edu/teaching/jk1/Electromagnetism/node110.html
"""

import cmath as cm
import numpy as np
import scipy.special as special
import scipy.constants as const
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 22})

# =============================================================================
# WHAM Parameters
# =============================================================================

#Antenna frequency
f=120*1e6 #Hz

#Plasma radius
a=0.2 #m

#Plasma length
l=1 #m

#Wave harmonic
n=1

#E field magnitude in vacuum
Ev=1

#H field magnitude in vacuum
Hv=1

# =============================================================================
# Derived WHAM Parameters
# =============================================================================

#Angular antenna frequency
omega=2*np.pi*f

kg=n*np.pi/l

kt=cm.sqrt(kg**2-(omega**2/const.c**2))

#%% Functions

def n_d(x,y,z):
    
    dDens=3e19
    
    return dDens

def n_t(x,y,z):
    
    tDens=3e19
    
    return tDens

def n_e(x,y,z):
    
    eDens=n_d(x,y,z)+n_t(x,y,z)
    
    return eDens

def b_field(x,y,z):
    
    Bx=0
    By=0
    Bz=2
    
    return np.array([Bx,By,Bz])

def b_mag(x,y,z):
    
    Bvec=b_field(x,y,z)
    
    magB=np.sqrt(Bvec[0]**2+Bvec[1]**2+Bvec[2]**2)
    
    return magB

def e_plasma_freq(x,y,z):
    
    freqSquared=n_e(x,y,z)*(const.e**2)/(const.epsilon_0*const.m_e)
    
    return np.sqrt(freqSquared)

def d_plasma_freq(x,y,z):
    
    freqSquared=n_d(x,y,z)*(const.e**2)/(const.epsilon_0*2*const.m_p)
    
    return np.sqrt(freqSquared)

def t_plasma_freq(x,y,z):
    
    freqSquared=n_t(x,y,z)*(const.e**2)/(const.epsilon_0*3*const.m_p)
    
    return np.sqrt(freqSquared)

def e_cyclotron_freq(x,y,z):
    
    freq=-const.e*b_mag(x,y,z)/const.m_e
    
    return freq

def d_cyclotron_freq(x,y,z):
    
    freq=const.e*b_mag(x,y,z)/(2*const.m_p)
    
    return freq

def t_cyclotron_freq(x,y,z):
    
    freq=const.e*b_mag(x,y,z)/(3*const.m_p)
    
    return freq

def p_dielectric(x,y,z):
    
    eTerm=e_plasma_freq(x,y,z)**2/(omega**2)
    dTerm=d_plasma_freq(x,y,z)**2/(omega**2)
    tTerm=t_plasma_freq(x,y,z)**2/(omega**2)
    
    P=1-eTerm-dTerm-tTerm
    
    return P

def r_dielectric(x,y,z):
    
    eTerm=e_plasma_freq(x,y,z)**2/(omega*(omega+e_cyclotron_freq(x,y,z)))
    dTerm=d_plasma_freq(x,y,z)**2/(omega*(omega+d_cyclotron_freq(x,y,z)))
    tTerm=t_plasma_freq(x,y,z)**2/(omega*(omega+t_cyclotron_freq(x,y,z)))
    
    R=1-eTerm-dTerm-tTerm
    
    return R

def l_dielectric(x,y,z):
    
    eTerm=e_plasma_freq(x,y,z)**2/(omega*(omega-e_cyclotron_freq(x,y,z)))
    dTerm=d_plasma_freq(x,y,z)**2/(omega*(omega-d_cyclotron_freq(x,y,z)))
    tTerm=t_plasma_freq(x,y,z)**2/(omega*(omega-t_cyclotron_freq(x,y,z)))
    
    L=1-eTerm-dTerm-tTerm
    
    return L

def s_dielectric(x,y,z):
    
    S=(r_dielectric(x,y,z)+l_dielectric(x,y,z))/2
    
    return S

def d_dielectric(x,y,z):
    
    D=(r_dielectric(x,y,z)-l_dielectric(x,y,z))/2
    
    return D

def dielectric_tensor(x,y,z):
    
    S=s_dielectric(x,y,z)
    D=d_dielectric(x,y,z)
    P=p_dielectric(x,y,z)
    
    epsilon=np.array([[S,1j*D,0],[-1j*D,S,0],[0,0,P]])
    
    return epsilon

def k_s(x,y,z):
    
    ksSquared=(p_dielectric(x,y,z)/const.epsilon_0)*(omega**2/const.c**2)-kg**2
    
    ks=cm.sqrt(ksSquared)
    
    return ks

def E_z(x,y,z):
    
    #Initialize
    Ez=0
    
    r=np.sqrt(x**2+y**2)
    
    #Vacuum Field
    if r>=a:
        
        #Modified Bessel Term
        kTerm=special.kn(0,kt*r)
        
        #Oscillating term
        oscTerm=np.e**(1j*kg*z)
        
        #Ez
        Ez=Ev*kTerm*oscTerm
        
    #Dielectric field
    else:
        
        #E field magnitude in the dielectric (only valid in a constant dielectric)

        #Wavenumber term
        wTerm=const.epsilon_0*k_s(a,0,0)/(s_dielectric(a,0,0)*kt)
        #Bessel term
        bTerm=special.kn(1,kt*a)/special.jv(1,k_s(a,0,0)*a)
        #Ed
        Ed=Ev*wTerm*bTerm
        
        #Bessel Term
        bTerm=special.jv(0,k_s(x,y,z)*r)
        
        #Oscillating term
        oscTerm=np.e**(1j*kg*z)
        
        #Ez
        Ez=Ed*bTerm*oscTerm
    
    return Ez
    
#%% Plotting

xArr=np.linspace(-0.4,0.4,20)
yArr=np.linspace(-0.4,0.4,20)

X,Y=np.meshgrid(xArr,yArr)

EzArr=[]

for x in xArr:
    
    currArr=[]
    
    for y in yArr:
        
        currArr.append(E_z(x,y,0))
        
    EzArr.append(currArr)
    
EzArr=np.array(EzArr)

#Create the plot
fig=plt.figure(figsize=(15,15))
ax=fig.add_subplot(111)
ax.contourf(EzArr,X,Y)
ax.grid(True)
plt.show()

#%% Sandbox

