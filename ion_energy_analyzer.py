# -*- coding: utf-8 -*-
"""
Created on Thu Feb 23 17:17:57 2023

@author: kunal

Script for calculations for the ion energy analyzer.

Eqautions are based on the following papers-
1. Molvik Paper- Review of Scientific Instruments 52, 704 (1981); https://doi.org/10.1063/1.1136655
2. TAE Paper- Review of Scientific Instruments 87, 11D428 (2016); https://doi.org/10.1063/1.4961081
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import constants as const
plt.rcParams.update({'font.size': 22})

def mu(E,B):
    """
    1st adiabatic invariant.
    """
    
    return E*const.e/Bm

def typical_ion_temp(potential):
    """
    Typical ion temperature. 
    Based on the ion energy distribution entering the expander.
    6x the ambipolar potential.
    
    Units- eV
    """
    
    return 6*potential

def max_angle(Bmirror,Bdetector,energy):
    """
    Based on eqn. 3 of the Molvik paper.
    Maximum angle the ions subtend with respect to the background magnetic
    field.
    
    Units- degrees
    """
    
    #mu of the particle in the mirror throat
    muParticle=mu(energy,Bmirror)
    
    #Maximum angle
    theta=np.arctan((Bdetector/(Bmirror-Bdetector-(const.e*phi/muParticle)))**0.5)
    
    #Convert to degrees
    theta=np.rad2deg(theta)
    
    return theta

def max_ion_gyroradius(Bmirror,Bdetector,energy):
    """
    Ion gyroradius for a particle with the maximum angle with respect to the
    background magnetic field at the detector. This will be the particle that
    basrely escapes over the trapped/passing boundary.
    
    Based on eqn. 8 of the Molvik paper.
    
    Units- m
    """
    
    rg=np.sqrt(2*2*const.m_p*const.e*energy/(Bmirror*Bdetector))/const.e
    
    return rg

def max_electron_gyroradius(Bdetector,energy):
    """
    Electron gyroradius for an electron with an angle 90deg with respect to the
    background magnetic field.
    """
    
    rg=np.sqrt(2*const.m_e*const.e*energy)/(Bdetector*const.e)
    
    return rg

def grid_radius(apertureRad,Bmirror,Bdetector,energy):
    """
    Radius of the retarding grids based on the maximum gyroradius of the most
    energetic particles at the largest angles.
    
    Based on eqn. 7 in the Molvik paper.
    
    Units- m
    """
    
    return apertureRad+2*max_ion_gyroradius(Bmirror, Bdetector, energy)

def grid_radius_off_axis(apertureRad,Bmirror,Bdetector,energy,offAxisAngle):
    """
    Radius of the retarding grids, taking into account the off axis B fields.
    
    Units- m
    """
    
    #Maximum ion gyroradius
    maxGyroRad=max_ion_gyroradius(Bmirror, Bdetector, energy)
    
    #Half angle as the detector axis can be betweeen the macine z and the field line
    halfAng=np.deg2rad(offAxisAngle/2)
    
    #Distance between ground and ion screen grid
    gridSep=grid_separation(energy)
    
    #Grid radius
    gridRad=4*gridSep*np.tan(halfAng)+2*maxGyroRad+apertureRad
    
    return gridRad

def current_density(inputCurr,radius):
    """
    We are assuming that the current density is uniform across the radial 
    profile. This is not strictly true, but a good 1st approximation.
    
    Units- A/m^2
    """
    
    #Current out of one end [A]
    #Chi=0.8
    #Shine-Thru=77%
    Iout=0.5*inputCurr*(1-0.8)*(1-0.77)
    
    #Current density at the detector [A/m^2]
    jOut=Iout/(np.pi*radius**2)
    
    return jOut

def plasma_density(inputCurr,radius,potential):
    """
    Assume that the plasma density only consists of the outflow from the
    central cell.
    
    Units- m^{-3}
    """
    
    #Current out of one end [A]
    #Chi=0.8
    #Shine-Thru=77%
    Iout=0.5*inputCurr*(1-0.8)*(1-0.77)
    
    TiTyp=typical_ion_temp(potential)
    
    #Line 1- # of particles per unit time
    #Line 2- 1 / Cross sectional area at the detector location
    #Line 3- 1 / Distance travelled by the ions in 1s
    plasmaDens=(Iout/const.e)*\
        (1/(np.pi*radius**2))*\
        np.sqrt(2*const.m_p/(2*const.e*TiTyp))
    
    return plasmaDens

def debye_length(temp,inputCurr,radius,potential):
    """
    Assume that the electron and ion temperatures are the same.
    Formula from the NRL formulary.
    
    Units- m
    """
    
    #Plasma density
    plasmaDens=plasma_density(inputCurr,radius,potential)
    
    debyeLen=7.43*np.sqrt(temp/(plasmaDens/1e6))
    
    return debyeLen

def grid_separation(maxEnergy):
    """
    Maximum separation between the ground grids and the ion repeller grid.
    
    Units- m
    """
    
    #Voltage standoff distance [V/m]
    standoffDist=0.75*1000*1000     #0.75 kV/mm
    
    return maxEnergy/standoffDist

def child_langmuir(minEnergy,maxEnergy):
    """
    Child-Langmuir limit on the current density in the analyzer.
    
    Units- A/m^2
    """
    
    gridDist=grid_separation(maxEnergy)
    
    #Depends on Emin as that is the minimum voltage we will apply to the ion
    #repeller grid. At arbitrarily low voltages, the jCL goes to 0.
    
    jCL=(4/9)*const.epsilon_0*\
        np.sqrt(2*const.e/(2*const.m_p))*\
        (minEnergy**(3/2))/(gridDist**2)
    
    return jCL

def grid_transparency(minEnergy,maxEnergy,inputCurr,radius):
    """
    How much we need to attenuate the signal before we are no longer space 
    charge limited. Transparency of 0 is no ions go through. Transparency of 1
    is no grid at all.
    """
    
    jCL=child_langmuir(minEnergy,maxEnergy)
    
    jOut=current_density(inputCurr,radius)
    
    return jCL/jOut

def incident_energy(inputCurr,radius,aperture):
    """
    Incident enery on the detector ground grid. Includes a safety factor of 2.
    
    Units- J
    """
    
    #Current density
    jDet=current_density(inputCurr, radius)
    
    #Total current on the detector
    iDet=jDet*np.pi*aperture**2
    
    #Power
    power=iDet*25*1000
    
    #Energy
    energy=power*20*1e-3
    
    return 2*energy

def temp_rise(minEnergy,maxEnergy,inputCurr,radius,aperture,thickness,material='SS'):
    """
    Temperature rise in the stainless steel ground grid per shot.
    
    Units- K
    """
    
    #Energy deposited on the grid
    energyDep=incident_energy(inputCurr, radius, aperture)
    
    #Transparency
    transparency=grid_transparency(minEnergy, maxEnergy, inputCurr, radius)
    
    #Volume of the grid
    vol=np.pi*(aperture**2)*thickness*transparency
    
    if material=='SS':
        
        #Mass of the grid
        #SS density = 8000 kg/m^3
        gridMass=vol*8000
        
        #Temperature change
        #Specific heat capacity of SS = 500 J/(kg K)
        delT=energyDep/(gridMass*500)
        
        return delT
    
    elif material=='Al':
        
        #Mass of the grid
        #Al density = 2710 kg/m^3
        gridMass=vol*2710
        
        #Temperature change
        #Specific heat capacity of Al = 890 J/(kg K)
        delT=energyDep/(gridMass*890)
        
        return delT
    
    elif material=='W':
        
        #Mass of the grid
        #W density = 19300 kg/m^3
        gridMass=vol*19300
        
        #Temperature change
        #Specific heat capacity of W = 134 J/(kg K)
        delT=energyDep/(gridMass*134)
        
        return delT
    
    elif material=='Ti':
        
        #Mass of the grid
        #Ti density = 4506 kg/m^3
        gridMass=vol*4506
        
        #Temperature change
        #Specific heat capacity of Ti = 523 J/(kg K)
        delT=energyDep/(gridMass*523)
        
        return delT

#%% Detector Parameters

#Detector energies [eV]
Emin=500
Emax=10*1000

#Energy resolution [eV]
Eres=50

#Electron temperature [eV]
TeCore=2000
TeEdge=100

#Ambipolar potential [V]
phi=1000

#Field strengths [T]
Bm=17       #Mirror throat
Ba=0.17     #Detector location

#B field angle at plasma edge [deg]
thetaB=14*2

#Injected NBI current [A]
Iinj=40

#Plasma radius at the detector z location [m]
plasmaRad=0.2

#Entrance aperture radius [m]
ra=0.002

#Grid thickness [m]
#38 Gauge SS sheel metal = 157um
gridThickness=157*1e-6

#%% Engineering parameters

# Grid separation
gridSep=grid_separation(Emax)
print('Minimum grid separation- '+str(np.round(gridSep*1000,2))+'mm')

# Grid radius
gridRad=grid_radius(ra,Bm,Ba,Emax)
print('Minimum grid radius- '+str(np.round(gridRad*1000,2))+'mm')

# Grid radius off-axis
gridRadOffAxis=grid_radius_off_axis(ra,Bm,Ba,Emax,thetaB)
print('Minimum off axis grid radius- '+str(np.round(gridRadOffAxis*1000,2))+'mm')

# Ground grid hole spacing
eGyroRad=max_electron_gyroradius(Ba,TeEdge)
print('Maximum spacing between holes for ground grid- '+str(np.round(2*eGyroRad*1000,2))+'mm')

# Grid transparency
gridTransp=grid_transparency(Emin,Emax,Iinj,plasmaRad)
print('Maximum ground grid transparency- '+str(np.round(gridTransp,3)))

# Hole size
#Maximum
debyeLen=debye_length(TeEdge,Iinj,plasmaRad,phi)
print('Maximum hole diameter for repeller grid- '+str(np.round(debyeLen*1000,2))+'mm')
#Minimum
holDia=2*gridThickness*np.tan(np.deg2rad(5+thetaB/2))
print('Minimum hole diameter for the grids- '+str(np.round(holDia*1e6,2))+'um')

# Total current
currDens=current_density(Iinj,plasmaRad)
apertureArea=np.pi*ra**2
totalCurrent=currDens*apertureArea*gridTransp
print('Total current into the detector- '+str(np.round(totalCurrent*1e6,2))+'uA')

# Incident energy on the detector
energy=incident_energy(Iinj,plasmaRad,ra)
print('Incident energy on the detector- '+str(np.round(energy*1000,2))+'mJ')

# Temperature rise of the grounding grid
deltaT=temp_rise(Emin, Emax, Iinj, plasmaRad, ra, gridThickness)
print('Temperature rise in the grounding grid per shot- '+str(np.round(deltaT,2))+'K')

#%% Plots

plotDest='C:/Users/kunal/OneDrive - UW-Madison/WHAM/Plots/Ion Energy Analyzer/'

#Grid thickness array [m]
gridThicknessArr=1e-6*np.linspace(10,300,100)

#Temperature rise arrays
tempRiseArrSS=temp_rise(Emin, Emax, Iinj, plasmaRad, ra, gridThicknessArr, material='SS')
tempRiseArrAl=temp_rise(Emin, Emax, Iinj, plasmaRad, ra, gridThicknessArr, material='Al')
tempRiseArrW=temp_rise(Emin, Emax, Iinj, plasmaRad, ra, gridThicknessArr, material='W')
tempRiseArrTi=temp_rise(Emin, Emax, Iinj, plasmaRad, ra, gridThicknessArr, material='Ti')

fig=plt.figure(figsize=(12,8))
ax=fig.add_subplot(111)

#Plot the temperature rise arrays
ax.plot(gridThicknessArr*1e6, tempRiseArrSS, linewidth=5, label='316L SS')
ax.plot(gridThicknessArr*1e6, tempRiseArrAl, linewidth=5, label='Aluminium')
ax.plot(gridThicknessArr*1e6, tempRiseArrW, linewidth=5, label='Tungsten')
ax.plot(gridThicknessArr*1e6, tempRiseArrTi, linewidth=5, label='Titanium', linestyle='dotted')

ax.set_xlim(0,np.max(gridThicknessArr)*1e6)
ax.set_ylim(0,200)
ax.set_xlabel(r'Grid Thickness [$\mu$m]')
ax.set_ylabel('Temperature Rise [K]')
ax.set_title(r'Ground Grid $\Delta$T after a shot')
ax.legend()
ax.grid()

plt.show()
plt.savefig(plotDest+'temp_rise.png',bbox_inches='tight')