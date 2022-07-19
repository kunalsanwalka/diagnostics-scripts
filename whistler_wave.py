# -*- coding: utf-8 -*-
"""
Created on Tue Apr 26 15:51:24 2022

@author: kunal
"""

import numpy as np
import scipy.constants as const
import matplotlib.pyplot as plt

plotDest='C:/Users/kunal/OneDrive - UW-Madison/WHAM/Plots/'

#%% Perpendicular wavelength

#Magnetic field strength (T)
B=0.86*2

#Density (m^{-3})
n=1e19

#Plasma length (m)
l=2000

#Antenna frequency (rad/s)
omega=2*const.e*B/(2*const.m_p)

#k_par (m^{-1})
kPar=2*const.pi/(2*l)

#Ion cyclotron frequency (rad/s)
Omega_i=const.e*B/(2*const.m_p)

#Electron cyclotron frequency (rad/s)
Omega_e=-const.e*B/(const.m_e)

#Electron plasma frequency (rad/s)
Pi_e=np.sqrt(n*(const.e**2)/(const.m_e*const.epsilon_0))

#n_par
nPar=const.c*kPar/omega

#Dispersion relation from- https://farside.ph.utexas.edu/teaching/plasma/Plasmahtml/node51.html
#See equation 538

#n_perp^2
nPerpSq=1-(nPar**2)-(Pi_e**2)/((omega+Omega_i)*(omega+Omega_e))

#k_perp (m^{-1})
kPerp=(omega/const.c)*np.sqrt(nPerpSq)

#Perpendicular wavelength (m)
lambdaPerp=2*const.pi/kPerp

print(lambdaPerp)

#%% Minimum density calculator

#Define some constants
pi=const.pi
c=const.c

#Magnetic field strength (T)
B=0.5

#Plasma length (m)
l=4

#Plasma radius (m)
r=0.2

#Antenna frequency (rad/s)
omega=const.e*B/(2*const.m_p)

#Ion cyclotron frequency (rad/s)
Omega_i=const.e*B/(2*const.m_p)

#Electron cyclotron frequency (rad/s)
Omega_e=-const.e*B/(const.m_e)

#k values
k_phi=2*pi/(1*pi*r)
k_r=2*pi/(2*r)
k_par=2*pi/(2*l)

#n values
n_phi=c*k_phi/omega
n_r=c*k_r/omega
n_par=c*k_par/omega

# n_phi=0
n_r=0
# n_par=0

#Plasma frequency squared
Pi_e_sq=(omega+Omega_i)*(omega+Omega_e)*(1-n_phi**2-n_r**2-n_par**2)

#FMSW
Pi_e_sq=(omega-Omega_i)*(omega-Omega_e)*(1-n_phi**2-n_r**2-n_par**2)

#Plasma density
n=Pi_e_sq*const.m_e*const.epsilon_0/(const.e**2)

print(n)

#%% Critical density vs. B field strength

#Define some constants
pi=const.pi
c=const.c

#Plasma length (m)
l=1.2

#Plama radius (m)
r=0.06

#k values
k_phi=2*pi/(1*pi*r)
k_r=0
k_par=2*pi/(2*l)

# k_phi=0

#Magnetic field strength array
bArr=np.linspace(0,5,100)

#Critical density
nCrit=[]

#Go over each field strength value
for i in range(len(bArr)):
    
    #Current field strength
    B=bArr[i]
    
    #Antenna frequency (rad/s)
    omega=const.e*B/(2*const.m_p)
    
    #Ion cyclotron frequency (rad/s)
    Omega_i=const.e*B/(2*const.m_p)
    
    #Electron cyclotron frequency (rad/s)
    Omega_e=-const.e*B/(const.m_e)
    
    #n values
    n_phi=c*k_phi/omega
    n_r=c*k_r/omega
    n_par=c*k_par/omega
    
    #Plasma frequency squared
    Pi_e_sq=(omega+Omega_i)*(omega+Omega_e)*(1-n_phi**2-n_r**2-n_par**2)
    
    #Plasma density
    n=Pi_e_sq*const.m_e*const.epsilon_0/(const.e**2)
    
    #Add to density array
    nCrit.append(n)

#Convert to numpy
nCrit=np.array(nCrit)

#Plot the data
fig,ax=plt.subplots(figsize=(12,8))

#Scientific notation
plt.ticklabel_format(axis='y',style='sci',scilimits=(0,0))

#Plot the data
plt.plot(bArr,nCrit)

#Axes labels
ax.set_xlabel('|B| [T]')
ax.set_ylabel(r'n [m$^{-3}$]')
ax.set_title('m=1 mode')

ax.grid(which='both')
plt.show()
plt.tight_layout()
plt.savefig(plotDest+'gamma10_m_1_critical_dens_vs_bMag.pdf')

#%% Critical density vs. Antenna frequency

#Define some constants
pi=const.pi
c=const.c

#Magnetic field strength (T)
B=0.86*4

#Plasma length (m)
l=1.2

#Plama radius (m)
r=0.06

#k values
k_phi=2*pi/(1*pi*r)
k_r=0
k_par=2*pi/(2*l)

# k_phi=0