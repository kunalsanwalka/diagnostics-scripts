# -*- coding: utf-8 -*-
"""
Created on Thu Mar 16 10:37:51 2023

@author: kunal
"""

import sys
sys.path.insert(1,'C:/Users/kunal/OneDrive - UW-Madison/WHAM/Python Scripts/')

import random
import aurora
import numpy as np
import scipy.constants as const
import matplotlib.pyplot as plt
from scipy.interpolate import RegularGridInterpolator
from scipy.integrate import solve_ivp
plt.rcParams.update({'font.size': 22})

# =============================================================================
# Plot Directory
# =============================================================================
plotDest='C:/Users/kunal/OneDrive - UW-Madison/WHAM/Plots/NBI Source/'

def rhs(l,n):
    """
    This function defines the RHS of the differential equations describing the
    evolution of the fast ion and fast neutral populations in the neutralizer
    of the NBI. These coupled first order ODE's are from Benjamin Hudson's
    thesis (eqn. 4.1).

    Parameters
    ----------
    l : float
        Position along the neutralizer.
    n : np.array
        Current values of the fast ion and fast neutral populations.
        n[0] = fast ions
        n[1] = fast neutrals

    Returns
    -------
    dndl : np.array
        Rate of change of the fast ions and fast neutrals.
        dndl[0] = fast ion change
        dndl[1] = fast neutral change
    """
    
    #Rate of change of fast ions
    dFastIons= - sigmaCX*n[0]*nBackground + sigmaImpact*n[1]*nBackground
    
    #Rate of change of fast neutrals
    dFastNeutrals= + sigmaCX*n[0]*nBackground - sigmaImpact*n[1]*nBackground
    
    #Construct rate of change array
    dndl=np.array([dFastIons,dFastNeutrals])
    
    return dndl

def cx_cross_section(energy,ionSpecies='D',neutralSpecies='D'):
    """
    This function calculates the CX cross section for a given fast ion species
    and background neutral.

    Parameters
    ----------
    energy : float
        Energy of the fast ion [eV].
    ionSpecies : string, optional
        Species name of the fast ion. The default is 'D'.
    neutralSpecies : string, optional
        Species name of the background neutral gas. The default is 'D'.

    Returns
    -------
    sigmaCX : float
        Charge exchange cross section [m^2].
    """
    
    #Define the cross section variable
    sigmaCX=0
    
    if ionSpecies=='D' and neutralSpecies=='D':
        
        #Convert energy to keV/amu for the Janev-Smith rates function
        E=energy/(2*1e3)
        
        sigmaCX=aurora.janev_smith_rates.js_sigma_cx_n1_q1(E)
        
    elif ionSpecies=='D' and neutralSpecies=='Na':
        
        #Convert energy to keV/amu
        E=energy/(2*1e3)
        
        # Coefficients taken from 'Database for inelastic collisions of sodium 
        # atoms with electrons, protons, and multiply charged ions' which can
        # be found here-
        # https://www.sciencedirect.com/science/article/pii/S0092640X08000405
        # Section 3.3
        # This formula is only valid for E>0.5keV
        
        a1=24.21
        a2=1
        a3=0
        a4=1.68e-3
        a5=5.02
        a6=-18.34
        a7=3.342
        a8=0.798
        a9=8.155e-4
        a10=3.25
        a11=1
        a12=0
        
        sigmaCX=1e-16*a1 * (np.exp(-a2*E)*(a12+np.log(a11+a3*E))/E + a4*np.exp(-a5*E)/(E**a6) + a7*np.exp(-a8/E)/(1+a9*E**a10))
    
    #Convert from cm^2 to m^2
    sigmaCX/=1e4
    
    return sigmaCX

def impact_ionization_cross_section(energy,ionSpecies='D',neutralSpecies='D'):
    """
    This function calculates the impact ionization cross section for a given 
    fast ion species and background neutral.

    Parameters
    ----------
    energy : float
        Energy of the fast ion [eV].
    ionSpecies : string, optional
        Species name of the fast ion. The default is 'D'.
    neutralSpecies : string, optional
        Species name of the background neutral gas. The default is 'D'.

    Returns
    -------
    sigmaImpact : float
        Impact ionization cross section [m^2].
    """
    
    #Define the cross section variable
    sigmaImpact=0
    
    if ionSpecies=='D' and neutralSpecies=='D':
        
        #Convert energy to keV/amu for the Janev-Smith rates function
        E=energy/(2*1e3)
        
        # Coefficients taken from 'ATOMIC AND PLASMA-MATERIAL INTERACTION DATA
        # FOR FUSION VOL. 4' which can be found here-
        # https://inis.iaea.org/collection/NCLCollectionStore/_Public/25/024/25024274.pdf
        # Section 2.2.1
        
        a1=12.899
        a2=61.897
        a3=9.2731e+3
        a4=4.9749e-04
        a5=3.989e-2
        a6=-1.59
        a7=3.1834
        a8=-3.7154
        
        sigmaImpact=1e-16*a1 * ( (np.exp(-a2/E)*np.log(1+a3*E)/E) + (a4*np.exp(-a5*E)/(E**a6 + a7*E**a8)) )
        
    elif ionSpecies=='D' and neutralSpecies=='Na':
        
        #Convert energy to keV/amu
        E=energy/(2*1e3)
        
        # Coefficients taken from 'ANALYTIC CROSS SECTIONS FOR CHARGE TRANSFER
        # OF HYDROGEN ATOMS AND IONS COLLIDING WITH METAL VAPORS' which can
        # be found here-
        # https://www.sciencedirect.com/science/article/pii/0168583X88903345
        
        Et=-8.46e-3
        Er=25
        a1=8.16e4
        a2=2
        a3=0.6
        a4=-0.5
        a5=3.21
        a6=2.86
        a7=5.4e-3
        a8=17.5
        
        def f(E0):
            """
            Sub function defined in the Tabata et. al. paper. Same format used
            in the code to make comparison with the paper easier.
            """
            
            E1=E0-Et        
            return a1 * ((E1/Er)**a2) / (1 + ((E1/a3)**(a2+a4)) + ((E1/a5)**(a2+a6)))
        
        sigmaImpact=1e-16 * (f(E) + a7*f(E/a8))
        
    #Convert from cm^2 to m^2
    sigmaImpact/=1e4
    
    return sigmaImpact

# =============================================================================
# Initial Condtions
# =============================================================================

#Beam Current [A]
beamCurr=40
#Beam Radius [m]
beamRad=0.1
#Beam Energy [eV]
beamEnergy=25e3

sigmaCX=cx_cross_section(beamEnergy,neutralSpecies='Na')
sigmaImpact=impact_ionization_cross_section(beamEnergy,neutralSpecies='Na')

#Initial fast ion density based on current and cross section
nFastIonInit=(beamCurr/const.e) * (1/(np.pi*beamRad**2)) * np.sqrt(2*const.m_p/(2*const.e*beamEnergy))
#Initial density of fast neutrals is always 0
nInit=np.array([nFastIonInit,0])

#Array over which to evaluate the answer
#Length of the neutralizer is 80cm per the technical manual
evalPos=np.linspace(0,0.8,1000)

#Background density [m^-3]
nBackground=1.25e20

# =============================================================================
# Solve the equations
# =============================================================================

#Solve the differential equations
soln = solve_ivp(fun=rhs,                       #RHS of the coupled ODE's
                 t_span=(0,evalPos[-1]),        #Start and end position of the neutralizer
                 y0=nInit,                      #Initial conditions
                 t_eval=evalPos)                #Positions at which to save the solutions

#Normalize with respect to the initial fast ion density
solnArr=soln.y/nFastIonInit

# =============================================================================
# Plotting
# =============================================================================

#Plot the solution
fig=plt.figure(figsize=(12,8))
ax=fig.add_subplot(111)

ax.plot(evalPos,solnArr[0],linewidth=5,label='Fast Ions')
ax.plot(evalPos,solnArr[1],linewidth=5,label='Fast Neutrals')

ax.set_xlabel('Position along neutralizer [m]')
ax.set_ylabel(r'Population fraction')
ax.set_title(r'E$_{beam}$= '+str(beamEnergy/1e3)+r'keV; n$_{background}$= '+str(nBackground/1e22)+r'$\cdot 10^{22} m^{-3}$')
# ax.text(0.31,0.65, 'Equilibrium Neutral Fraction= '+str(np.round(solnArr[1,-1],2)), bbox=dict(boxstyle='round', facecolor='white', alpha=0.5))
ax.legend()
plt.savefig(plotDest+'fast_ion_neutralization_sodium.png',bbox_inches='tight')
plt.show()

#%% Na vs D in the neutralizer

# =============================================================================
# Initial Condtions
# =============================================================================

#Beam Current [A]
beamCurr=40
#Beam Radius [m]
beamRad=0.1
#Beam Energy [eV]
beamEnergy=100e3

#Initial fast ion density based on current and cross section
nFastIonInit=(beamCurr/const.e) * (1/(np.pi*beamRad**2)) * np.sqrt(2*const.m_p/(2*const.e*beamEnergy))
#Initial density of fast neutrals is always 0
nInit=np.array([nFastIonInit,0])

#Array over which to evaluate the answer
#Length of the neutralizer is 80cm per the technical manual
evalPos=np.linspace(0,0.8,1000)

#Background density [m^-3]
nBackground=1.25e20

# =============================================================================
# D solution
# =============================================================================

#Cross sections
sigmaCX=cx_cross_section(beamEnergy,neutralSpecies='D')
sigmaImpact=impact_ionization_cross_section(beamEnergy,neutralSpecies='D')

#Solve the differential equations
soln = solve_ivp(fun=rhs,                       #RHS of the coupled ODE's
                 t_span=(0,evalPos[-1]),        #Start and end position of the neutralizer
                 y0=nInit,                      #Initial conditions
                 t_eval=evalPos)                #Positions at which to save the solutions

#Normalize with respect to the initial fast ion density
solnArrD=soln.y/nFastIonInit

# =============================================================================
# Na solution
# =============================================================================

#Cross sections
sigmaCX=cx_cross_section(beamEnergy,neutralSpecies='Na')
sigmaImpact=impact_ionization_cross_section(beamEnergy,neutralSpecies='Na')

#Solve the differential equations
soln = solve_ivp(fun=rhs,                       #RHS of the coupled ODE's
                 t_span=(0,evalPos[-1]),        #Start and end position of the neutralizer
                 y0=nInit,                      #Initial conditions
                 t_eval=evalPos)                #Positions at which to save the solutions

#Normalize with respect to the initial fast ion density
solnArrNa=soln.y/nFastIonInit

# =============================================================================
# Plot the solution
# =============================================================================

fig=plt.figure(figsize=(12,8))
ax=fig.add_subplot(111)

ax.plot(evalPos*1000,solnArrD[1],linewidth=5,label='D Gas')
ax.plot(evalPos*1000,solnArrNa[1],linewidth=5,label='Na Vapor')

ax.set_xlabel('Position along neutralizer [mm]')
ax.set_ylabel(r'Neutral Population fraction')
ax.set_title(r'E$_{beam}$= '+str(beamEnergy/1e3)+r'keV; n$_{background}$= '+str(nBackground/1e22)+r'$\cdot 10^{22} m^{-3}$')
# ax.text(0.31,0.65, 'Equilibrium Neutral Fraction= '+str(np.round(solnArr[1,-1],2)), bbox=dict(boxstyle='round', facecolor='white', alpha=0.5))
ax.legend()
plt.savefig(plotDest+'fast_ion_neutralization_both.png',bbox_inches='tight')
plt.show()

#%% Compare Igenbergs et. al. with Tabata et. al

#Energy array [eV]
energyArr=np.power(10,np.linspace(2,6,100))

#Cross Section Arrays
igenbergsArr=[]
tabataArr=[]

#Go over each energy
for energy in energyArr:
    
    #Igenbergs cross section [cm^2]
    igenbergsArr.append(cx_cross_section(energy,neutralSpecies='Na')*1e4)
    
    #Convert to keV
    energy/=2*1e3
    
    #Tabata cross section
    Et=-8.46e-3
    Er=25
    a1=8.16e4
    a2=2
    a3=0.6
    a4=-0.5
    a5=3.21
    a6=2.86
    a7=5.4e-3
    a8=17.5
    
    def f(E0):
        
        E1=E0-Et        
        return a1 * ((E1/Er)**a2) / (1 + ((E1/a3)**(a2+a4)) + ((E1/a5)**(a2+a6)))
    
    #Calculate the cross section [cm^2]
    sigma=1e-16 * (f(energy) + a7*f(energy/a8))
    
    tabataArr.append(sigma)

#Convert to numpy
igenbergsArr=np.array(igenbergsArr)
tabataArr=np.array(tabataArr)

fig=plt.figure(figsize=(12,8))
ax=fig.add_subplot(111)

ax.loglog(energyArr,igenbergsArr,linewidth=5,label='Igenbergs et. al.')
ax.loglog(energyArr,tabataArr,linewidth=5,label='Tabata et. al.')

ax.set_xlabel('Deuterium Beam energy [eV]')
ax.set_ylabel(r'$\sigma_{CX}$ [cm$^2$]')
ax.legend()
plt.savefig(plotDest+'compare_papers.png',bbox_inches='tight')
plt.show()

































