# -*- coding: utf-8 -*-
"""
Created on Fri Feb 26 16:33:48 2021

@author: kunalsanwalka

This program contains all the routines used to analyze outputs from CQL3D using
the routines written in cql3d_analysis_functions

"""

import sys
sys.path.insert(1,'C:/Users/kunal/OneDrive - UW-Madison/WHAM/Python Scripts/')

import csv
import numpy as np
import netCDF4 as nc
import scipy.constants as const
import matplotlib.pyplot as plt
import eqdsk_analysis_toolbox as eqTools
from tabulate import tabulate
from scipy.interpolate import RegularGridInterpolator
plt.rcParams.update({'font.size': 22})

# =============================================================================
# Plot Directory
# =============================================================================
plotDest='C:/Users/kunal/OneDrive - UW-Madison/WHAM/Plots/'

# =============================================================================
# Processed Data Directory
# =============================================================================
dataDest='C:/Users/kunal/OneDrive - UW-Madison/WHAM/Processed Data/'

# =============================================================================
# CQL3D Data Directory
# =============================================================================
cql3dDest='C:/Users/kunal/OneDrive - UW-Madison/WHAM/Data/CQL3D/'

#Import all the analysis functions and packages
from cql3d_analysis_functions import *

#%% Analysis

#Location of the CQL3D output
filename=cql3dDest+'230222_newFusDiag.nc'

#Location of the eqdsk
filenameEqdsk='C:/Users/kunal/OneDrive - UW-Madison/WHAM/Data/eqdsk/WHAM_Phase_2_eqdsk'

#Get the netCDF4 data
ds=nc.Dataset(filename)

# Get the 0D Parameters
# zeroDParams=zero_d_parameters(filename,longParams=True)

#Get the distribution function data
# distData,vPar,vPerp=dist_func(filename,makeplot=True,saveplot=True,fluxsurfplot=0,species=0,vMax=4e6)

#Plot all distribution functions
# plot_dist_funcs(filename,saveplot=True,species=1,vMax=6e6)

#Get the derivatives of the distribution function
# dist_func_derivatives(filename,makeplot=True)

#Get the ion densities
# ndwarmz,ndfz,ndtotz,solrz,solzz=ion_dens(filename,makeplot=True,saveplot=True,savedata=True,species=0)

#Get the radial density profile
# radArr,radDens=radial_density_profile(filename,makeplot=True,saveplot=True)

#Get the axial density profile
# axialArr,axialDens=axial_density_profile(filename,makeplot=True,saveplot=True)

#Get the pressure profiles for a given species
# pressparz_d,pressprpz_d,pressz_d,solrz,solzz=pressure(filename,makeplot=True,saveplot=False,savedata=True,species=0)

#Get the total pressure
# pressparz,pressprpz,pressz,solrz,solzz=total_pressure(filename,makeplot=True,saveplot=True,savedata=True)

#Get the plasma beta
# betaArr,solrz,solzz=beta(filename,makeplot=True,saveplot=True)

#Get the AIC growthrate
# gammaNormArr,solrz,solzz=aic_growthrate(filename,makeplot=True,saveplot=True)

# Get the axial fusion neutron flux
# fusArr,zArr=axial_neutron_flux(filename,makeplot=True,saveplot=True)

#Get the radial fusion flux
# rya,fusPower=radial_fusion_power(filename,makeplot=True,saveplot=True)

# Get the fusion reaction rate 
# fusrxrt,tArr=fusion_rx_rate(filename,makeplot=True,saveplot=True)

#Get the radial Q_fus profile
# rya,qFus=radial_q_profile(filename,makeplot=True,saveplot=True)

#Get the radial input power density
# rya,sorpw_nbi,sorpw_rf,sorpw_tot=source_power_dens(filename,makeplot=True,saveplot=True)

#Get the radial integrated input power density
# rya,sourceRate,integratedRF,totalIntegrated=integrated_power_density(filename,makeplot=True,saveplot=True)

# Get the radial input power profile
# rya,rfProfile,nbiProfile,totProfile=radial_input_power(filename,makeplot=True,saveplot=True)

#Get the radial integrated input power profile
# rya,integratedNBI,integratedRF,integrateTot=integrated_power(filename,makeplot=True,saveplot=True)

#Get the fast ion confinement time
# rya,tau_i=fast_ion_confinement_time(filename,makeplot=True,saveplot=True)

# Get the average energy
# rya,energyLastT=average_energy_final_timestep(filename,makeplot=True,saveplot=True)

#Get the average energy time evolution
# rya,time,avgEnergy=average_energy(filename,species=0,makeplot=True,saveplot=True)

filenameFreya=cql3dDest+'freya_points_radProf.txt'

#Get the NBI bith points
# xArr,yArr,zArr,rArr,vxArr,vyArr,vzArr=nbi_birth_points(filenameFreya=filenameFreya,filenameEqdsk=filenameEqdsk,withFields=True,makeplot=True,saveplot=True)

#Get the NBI initial and final bounce fields
# Binit,Bfinal=nbi_bounce_field(filenameFreya=filenameFreya,filenameEqdsk=filenameEqdsk,makeplot=True,saveplot=True)

#Get the RF ray data
# freq,delpwr,sdpwr,sbtot=ray_power_absorption(filename,makeplot=True,species=1)

#%% Run Scan (Setup)

# =============================================================================
# Location of CQL3D outputs
# =============================================================================

#NBI Power Scan
# filenameArr=[cql3dDest+'220321_pwr_100kW.nc',
#              cql3dDest+'220321_pwr_200kW.nc',
#              cql3dDest+'220321_pwr_300kW.nc',
#              cql3dDest+'220321_pwr_400kW.nc',
#              cql3dDest+'220321_pwr_500kW.nc']

# #Density scan
# filenameArr=[cql3dDest+'220321_dt_1e13.nc',
#              cql3dDest+'220321_dt_2e13.nc',
#              cql3dDest+'220321_dt_3e13.nc',
#              cql3dDest+'220321_dt_4e13.nc',
#              cql3dDest+'220321_dt_5e13.nc']

# filenameArr=[cql3dDest+'220627_genDens_1e13_Te_1000eV.nc',
#              cql3dDest+'220627_genDens_2e13_Te_1000eV.nc',
#              cql3dDest+'220627_genDens_3e13_Te_1000eV.nc',
#              cql3dDest+'220627_genDens_4e13_Te_1000eV.nc',
#              cql3dDest+'220627_genDens_5e13_Te_1000eV.nc']

#NBI Energy scan
# filenameArr=[cql3dDest+'220428_only3rd_10keV.nc',
#               cql3dDest+'220428_only3rd_15keV.nc',
#               cql3dDest+'220428_only3rd_25keV.nc',
#               cql3dDest+'220428_only3rd_35keV.nc',
#               cql3dDest+'220428_only3rd_45keV.nc',
#               cql3dDest+'220428_only3rd_55keV.nc',
#               cql3dDest+'220428_only3rd_65keV.nc',
#               cql3dDest+'220428_only3rd_75keV.nc',
#               cql3dDest+'220428_only3rd_85keV.nc',
#               cql3dDest+'220428_only3rd_95keV.nc',
#               cql3dDest+'220428_only3rd_105keV.nc',
#               cql3dDest+'220428_only3rd_115keV.nc',
#               cql3dDest+'220428_only3rd_125keV.nc']

#RF Power scan
# filenameArr=[cql3dDest+'220404_rf_0kW.nc',
#                cql3dDest+'220404_rf_10kW.nc',
#                cql3dDest+'220404_rf_20kW.nc',
#                cql3dDest+'220404_rf_30kW.nc',
#                cql3dDest+'220404_rf_40kW.nc',
#                cql3dDest+'220404_rf_50kW.nc',
#                cql3dDest+'220404_rf_60kW.nc',
#                cql3dDest+'220404_rf_70kW.nc',
#                cql3dDest+'220404_rf_80kW.nc',
#                cql3dDest+'220404_rf_90kW.nc',
#                cql3dDest+'220404_rf_100kW.nc',
#                cql3dDest+'220404_rf_125kW.nc',
#                cql3dDest+'220404_rf_150kW.nc',
#                cql3dDest+'220404_rf_175kW.nc']

#Misc Scans
# filenameArr=[cql3dDest+'220512_maxDens_10e12.nc',
#              cql3dDest+'220512_maxDens_9e12.nc',
#              cql3dDest+'220512_maxDens_8e12.nc',
#              cql3dDest+'220512_maxDens_7e12.nc',
#              cql3dDest+'220512_maxDens_6e12.nc',
#              cql3dDest+'220512_maxDens_5e12.nc',
#              cql3dDest+'220512_maxDens_4e12.nc',
#              cql3dDest+'220512_maxDens_3e12.nc',
#              cql3dDest+'220512_maxDens_2e12.nc',
#              cql3dDest+'220512_maxDens_1e12.nc']

# filenameArr=[cql3dDest+'220513_resChange_maxTemp_10eV.nc',
#              cql3dDest+'220513_resChange_maxTemp_20eV.nc',
#              cql3dDest+'220513_resChange_maxTemp_30eV.nc',
#              cql3dDest+'220513_resChange_maxTemp_40eV.nc']

# filenameArr=[cql3dDest+'220518_zeff_maxTemp_10eV.nc',
#              cql3dDest+'220518_zeff_maxTemp_20eV.nc',
#              cql3dDest+'220518_zeff_maxTemp_30eV.nc',
#              cql3dDest+'220518_zeff_maxTemp_40eV.nc',
#              cql3dDest+'220518_zeff_maxTemp_50eV.nc']

# filenameArr=[cql3dDest+'220520_dens_1e12.nc',
#              cql3dDest+'220520_dens_2e12.nc',
#              cql3dDest+'220520_dens_5e12.nc',
#              cql3dDest+'220520_dens_10e12.nc']

filenameArr=[cql3dDest+'220912_noNBI_1ms.nc',
             cql3dDest+'220912_noNBI_2ms.nc',
             cql3dDest+'220912_noNBI_5ms.nc',
             cql3dDest+'220912_noNBI_10ms.nc',
             cql3dDest+'220912_noNBI_20ms.nc']

# =============================================================================
# Labels for each file
# =============================================================================

#NBI Power Scan
# labelArr=['0kW','100kW','200kW','300kW','400kW','500kW','600kW','700kW','800kW','900kW','1000kW']
# labelArr=['100kW','200kW','300kW','400kW','500kW']
# labelArr=['0kW','10kW','25kW','50kW','100kW','250kW','500kW','1000kW']

#Density Scan
# labelArr=[r'$1\cdot10^{13}cm^{-3}$',r'$2\cdot10^{13}cm^{-3}$',r'$3\cdot10^{13}cm^{-3}$',r'$4\cdot0^{13}cm^{-3}$',r'$5\cdot0^{13}cm^{-3}$']
# labelArr=[r'$1\cdot10^{13}cm^{-3}$',r'$2\cdot10^{13}cm^{-3}$',r'$3\cdot10^{13}cm^{-3}$',r'$4\cdot0^{13}cm^{-3}$',r'$5\cdot0^{13}cm^{-3}$']

#NBI Energy Scan
# labelArr=['10keV','25keV','50keV','75keV','100keV','125keV','150keV','200keV']
# labelArr=['1keV','5keV','10keV','15keV','20keV','25keV']
# labelArr=['10keV','15keV','17keV','19keV','20keV','21keV','22keV','25keV','27keV','29keV','30keV','31keV','33keV','35keV','45keV','55keV','65keV']
# labelArr=['10keV','15keV','17keV','19keV','20keV','21keV','22keV','25keV','35keV','45keV','55keV','65keV']

#RF Power Scan
# labelArr=['0kW','10kW','25kW','50kW','75kW','100kW','125kW','175kW','225kW','250kW']
# labelArr=['0kW','10kW','20kW','30kW','40kW','50kW','60kW','70kW','80kW','90kW','100kW','125kW','150kW','175kW']

# =============================================================================
# Parameter values in each file
# =============================================================================

#NBI Power Scan
# paramArr=np.array([1e-3,10,25,50,100,250,500,1000])
# paramArr=np.array([100,200,300,400,500])

#Density Scan
# paramArr=np.array([1e13,2e13,3e13,4e13,5e13])

#NBI Energy Scan
# paramArr=np.array([10,15,25,35,45,55,65,75,85,95,105,115,125])

#RF Power Scan
# paramArr=np.array([0,10,20,30,40,50,60,70,80,90,100,125,150,175])

#Misc Scan
# paramArr=np.array([10,20,30,40,50])
# paramArr=np.array([1,2,5,10])
# paramArr=np.array([9,10,11,12,13])
paramArr=np.array([1,2,5,10,20])

# =============================================================================
# Plot name
# =============================================================================

#Savefile Prefix
# plotName='WHAM_rf_pwrScan'
# plotName='WHAM_nbi_pwrScan'
# plotName='220428_nbi_energyScan_only3rd'
# plotName='220627_genDens_scan'
plotName='220912_noHeating'

#Plot title
# plotTitle='Background Density Scan'
# plotTitle='RF Power Scan'
# plotTitle='NBI Power Scan'
# plotTitle='NBI Energy Scan'
# plotTitle='Maxwellian Temperature Scan'
# plotTitle='General Density Scan'
plotTitle=r'No Heating, WHAM 0.86T, $T_e$=1keV'

#x-axis label
# xAxLabel='RF Power [kW]'
# xAxLabel='NBI Power [kW]'
# xAxLabel=r'Initial General Density (n$_i$) [cm$^{-3}$]'
# xAxLabel='NBI Energy [keV]'
# xAxLabel=r'Density [$10^{x} cm^{-3}$]'
# xAxLabel='Temperature [eV]'
xAxLabel='Simulation Time [ms]'

#Number of runs
numRuns=len(filenameArr)

#%% Scan- DD reaction rates

ddRateArr=[]

#Go over each file
for i in range(len(filenameArr)):
    
    #Get the filename
    filename=filenameArr[i]
    
    #Get the fusion reactivity
    fusrxrt,time=fusion_rx_rate(filename)
    
    #Get the DD reaction rate at the last timestep
    ddRate=fusrxrt[2,-1]+fusrxrt[3,-1]
    
    #Append to the array
    ddRateArr.append(ddRate)

#Initialize the plot
fig,ax=plt.subplots(figsize=(12,8))

#Plot the data
ax.plot(paramArr,ddRateArr)
ax.scatter(paramArr,ddRateArr)

#Add labels
ax.set_xlabel(xAxLabel)
ax.set_ylabel(r'Peak DD reaction rate [s$^{-1}$]')
ax.set_title(plotTitle)

ax.set_xlim(min(paramArr),max(paramArr))
ax.grid(which='both')
plt.show()
plt.savefig(plotDest+plotName+'_dd_rx_rate_scan.pdf')

#%% Scan- DT reation rates

dtRateArr=[]

#Go over each file
for i in range(len(filenameArr)):
    
    #Get the filename
    filename=filenameArr[i]
    
    #Get the fusion reactivity
    fusrxrt,time=fusion_rx_rate(filename)
    
    #Get the DT reaction rate at the last timestep
    dtRate=fusrxrt[0,-1]
    
    #Append to the array
    dtRateArr.append(dtRate)

#Initialize the plot
fig,ax=plt.subplots(figsize=(12,8))

#Plot the data
ax.plot(paramArr,dtRateArr)

#Add labels
ax.set_xlabel(xAxLabel)
ax.set_ylabel(r'Peak DT reaction rate [s$^{-1}$]')
ax.set_title(plotTitle)

ax.set_xlim(min(paramArr),max(paramArr))
ax.grid(which='both')
plt.show()
plt.savefig(plotDest+plotName+'_dt_rx_rate_scan.pdf')

#%% Scan- Maximum plasma beta

betaArr=[]

#Go over each file
for i in range(len(filenameArr)):
    
    #Get the filename
    filename=filenameArr[i]
    
    #Get the plasma beta
    betaMap,solrz,solzz=beta(filename)
    
    #Find the maximum beta value
    betaMax=np.max(betaMap)
    
    #Append to the array
    betaArr.append(betaMax)
    
#Initialize the plot
fig,ax=plt.subplots(figsize=(12,8))

#Plot the data
ax.plot(paramArr,betaArr)

#Add labels
ax.set_xlabel(xAxLabel)
ax.set_ylabel(r'Maximum $\beta$')
ax.set_title(plotTitle)

ax.set_xlim(min(paramArr),max(paramArr))
ax.grid(which='both')
plt.show()
plt.savefig(plotDest+plotName+'_beta_scan.pdf')

#%% Scan- Maximum particle energy

maxEnergyArr=[]

#Go over each file
for i in range(len(filenameArr)):
    
    #Get the filename
    filename=filenameArr[i]
    
    #Get the average energy
    rya,energyLastT=average_energy(filename)
    
    #Find the maximum energy
    eMax=np.max(energyLastT)
    
    #Append to the array
    maxEnergyArr.append(eMax)

#Convert to numpy
maxEnergyArr=np.array(maxEnergyArr)
    
#Initialize the plot
fig,ax=plt.subplots(figsize=(12,8))

#Scientific notation
plt.ticklabel_format(axis='y',style='sci',scilimits=(0,0))

#Plot the data
ax.plot(paramArr,maxEnergyArr/1000)

#Add labels
ax.set_xlabel(xAxLabel)
ax.set_ylabel(r'Maximum Particle Energy [keV]')
ax.set_title(plotTitle)

ax.set_xlim(min(paramArr),max(paramArr))
ax.grid(which='both')
plt.show()
plt.savefig(plotDest+plotName+'_max_energy_scan.pdf')

#%% Scan- NBI Absorption Fraction

#NBI power per run
# nbiPower=np.array([1e-6,10,25,50,100,250,500,1000]) #kW
# nbiPower=np.array([100,110,150,170,190,200,210,220,250,350,450,550,650]) #kW
nbiPower=np.array([100]*numRuns) #kW

absorbFracArr=[]

#Go over each file
for i in range(len(filenameArr)):
    
    #Get the filename
    filename=filenameArr[i]
    
    #Get the amount of NBI absorbed
    zeroDParams=zero_d_parameters(filename)
    
    #Calculate the absorption fraction
    absorbFrac=100*zeroDParams['nbiPow']/(nbiPower[i]*1000)
    
    #Append to the array
    absorbFracArr.append(absorbFrac)
    
#Convert to numpy
absorbFracArr=np.array(absorbFracArr)

#Initialize the plot
fig,ax=plt.subplots(figsize=(12,8))

#Scientific notation
# plt.ticklabel_format(axis='y',style='sci',scilimits=(0,0))

#Plot the data
ax.plot(paramArr,absorbFracArr)
ax.scatter(paramArr,absorbFracArr)

#Add labels
ax.set_xlabel(xAxLabel)
ax.set_ylabel(r'NBI Absorption Percentage [%]')
ax.set_title(plotTitle)

ax.set_xlim(min(paramArr),max(paramArr))
ax.set_ylim(0,100)
ax.grid(which='both')
plt.show()
plt.savefig(plotDest+plotName+'_nbi_absorption_fraction.pdf')

#%% Scan- RF Absorption Fraction

#RF power per run
# rfPower=np.array([0,10,20,30,40,50,60,70,80,90,100,125,150,175]) #kW
rfPower=np.array([300]*numRuns) #kW

absorbFracArr=[]

#Go over each file
for i in range(len(filenameArr)):
    
    #Get the filename
    filename=filenameArr[i]
    
    #Get the amount of NBI absorbed
    zeroDParams=zero_d_parameters(filename)
    
    #Calculate the absorption fraction
    absorbFrac=100*zeroDParams['rfPow']/(rfPower[i]*1000)
    
    #Append to the array
    absorbFracArr.append(absorbFrac)
    
#Convert to numpy
absorbFracArr=np.array(absorbFracArr)

#Initialize the plot
fig,ax=plt.subplots(figsize=(12,8))

#Scientific notation
plt.ticklabel_format(axis='y',style='sci',scilimits=(0,0))

#Plot the data
ax.plot(paramArr,absorbFracArr)
ax.scatter(paramArr,absorbFracArr)

#Add labels
ax.set_xlabel(xAxLabel)
ax.set_ylabel(r'RF Absorption Percentage [%]')
ax.set_title(plotTitle)

ax.set_xlim(min(paramArr),max(paramArr))
ax.set_ylim(0,100)
ax.grid(which='both')
plt.show()
# plt.savefig(plotDest+plotName+'_rf_absorption_fraction.pdf')

#%% Scan- Distribution Function

#Maximum velocity on the plots
vMax=2.5e6

#Flux surface index
fluxSurf=0

#Initialize the plot
fig,axs=plt.subplots(numRuns,1,figsize=(21,numRuns*9))

#Go over each file
for i in range(0,numRuns):
    
    #Get the filename
    filename=filenameArr[i]
    
    #Get the distribution function data
    distData,vPar,vPerp=dist_func(filename)
    
    #Convert the data to log
    logData=np.log10(distData)
    
    #Max and min of the distribution function
    maxDist=np.max(logData)
    minDist=maxDist-15
    
    #Create the distribution function plot
    ax=axs[i]
    
    pltObj=ax.contourf(vPar[fluxSurf],vPerp[fluxSurf],logData[fluxSurf],levels=np.linspace(minDist,maxDist,30))
    ax.contour(pltObj,colors='black')
    
    ax.set_xlabel(r'$v_{||}$ [m/s]')
    ax.set_xlim(-vMax,vMax)
    ax.set_xticks(np.linspace(-vMax,vMax,11))
    
    ax.set_ylabel(r'$v_{\perp}$ [m/s]')
    ax.set_ylim(0,vMax)
    
    # ax.set_title(r'Maxwellian Density='+str(paramArr[i])+r'$\cdot 10^{12} cm^{-3}$')
    # ax.set_title(r'Maxwellian Temperature='+str(paramArr[i])+' eV')
    # ax.set_title(r'General Density= $10^{'+str(paramArr[i])+'} cm^{-3}$')
    # ax.set_title(r'General Density= '+str(paramArr[i]/1e13)+r'$\cdot 10^{13}$ cm$^{-3}$')
    ax.set_title(r'Time= '+str(paramArr[i])+'ms')
    
    ax.grid(True)
    
    cbar=fig.colorbar(pltObj,ax=ax)
    cbar.set_label(r'log$_{10}$(v$^{-3}$)')
    
plt.savefig(plotDest+plotName+'_dist_func_fluxsurf_'+str(fluxSurf)+'.pdf',bbox_inches='tight')
plt.close()
# plt.show()

#%% Scan- Maximum Density

densArr=[]

#Go over each file
for i in range(len(filenameArr)):
    
    #Get the filename
    filename=filenameArr[i]
    
    #Get the plasma densities
    ndwarmz,ndfz,ndtotz,solrz,solzz=ion_dens(filename)
    
    #Find the maximum density
    densMax=np.max(ndtotz)
    
    #Append to the array
    densArr.append(densMax)
    
#Initialize the plot
fig,ax=plt.subplots(figsize=(12,8))

#Plot the data
ax.plot(paramArr,densArr)

#Add labels
ax.set_xlabel(xAxLabel)
ax.set_ylabel(r'Maximum Density [m$^{-3}$]')
ax.set_title(plotTitle)

ax.set_xlim(min(paramArr),max(paramArr))
ax.grid(which='both')
plt.show()
plt.savefig(plotDest+plotName+'_maxDens_scan.pdf')

#%% Contour- DD reaction rate

# =============================================================================
# Location of CQL3D outputs
# =============================================================================

filenameArr=[[cql3dDest+'220627_genDens_1e13_Te_200eV.nc',
             cql3dDest+'220627_genDens_1e13_Te_400eV.nc',
             cql3dDest+'220627_genDens_1e13_Te_600eV.nc',
             cql3dDest+'220627_genDens_1e13_Te_800eV.nc',
             cql3dDest+'220627_genDens_1e13_Te_1000eV.nc',
             cql3dDest+'220627_genDens_1e13_Te_1200eV.nc',
             cql3dDest+'220627_genDens_1e13_Te_1400eV.nc',
             cql3dDest+'220627_genDens_1e13_Te_1600eV.nc',
             cql3dDest+'220627_genDens_1e13_Te_1800eV.nc',
             cql3dDest+'220627_genDens_1e13_Te_2000eV.nc'],
             [cql3dDest+'220627_genDens_2e13_Te_200eV.nc',
             cql3dDest+'220627_genDens_2e13_Te_400eV.nc',
             cql3dDest+'220627_genDens_2e13_Te_600eV.nc',
             cql3dDest+'220627_genDens_2e13_Te_800eV.nc',
             cql3dDest+'220627_genDens_2e13_Te_1000eV.nc',
             cql3dDest+'220627_genDens_2e13_Te_1200eV.nc',
             cql3dDest+'220627_genDens_2e13_Te_1400eV.nc',
             cql3dDest+'220627_genDens_2e13_Te_1600eV.nc',
             cql3dDest+'220627_genDens_2e13_Te_1800eV.nc',
             cql3dDest+'220627_genDens_2e13_Te_2000eV.nc'],
             [cql3dDest+'220627_genDens_3e13_Te_200eV.nc',
             cql3dDest+'220627_genDens_3e13_Te_400eV.nc',
             cql3dDest+'220627_genDens_3e13_Te_600eV.nc',
             cql3dDest+'220627_genDens_3e13_Te_800eV.nc',
             cql3dDest+'220627_genDens_3e13_Te_1000eV.nc',
             cql3dDest+'220627_genDens_3e13_Te_1200eV.nc',
             cql3dDest+'220627_genDens_3e13_Te_1400eV.nc',
             cql3dDest+'220627_genDens_3e13_Te_1600eV.nc',
             cql3dDest+'220627_genDens_3e13_Te_1800eV.nc',
             cql3dDest+'220627_genDens_3e13_Te_2000eV.nc'],
             [cql3dDest+'220627_genDens_4e13_Te_200eV.nc',
             cql3dDest+'220627_genDens_4e13_Te_400eV.nc',
             cql3dDest+'220627_genDens_4e13_Te_600eV.nc',
             cql3dDest+'220627_genDens_4e13_Te_800eV.nc',
             cql3dDest+'220627_genDens_4e13_Te_1000eV.nc',
             cql3dDest+'220627_genDens_4e13_Te_1200eV.nc',
             cql3dDest+'220627_genDens_4e13_Te_1400eV.nc',
             cql3dDest+'220627_genDens_4e13_Te_1600eV.nc',
             cql3dDest+'220627_genDens_4e13_Te_1800eV.nc',
             cql3dDest+'220627_genDens_4e13_Te_2000eV.nc'],
             [cql3dDest+'220627_genDens_5e13_Te_200eV.nc',
             cql3dDest+'220627_genDens_5e13_Te_400eV.nc',
             cql3dDest+'220627_genDens_5e13_Te_600eV.nc',
             cql3dDest+'220627_genDens_5e13_Te_800eV.nc',
             cql3dDest+'220627_genDens_5e13_Te_1000eV.nc',
             cql3dDest+'220627_genDens_5e13_Te_1200eV.nc',
             cql3dDest+'220627_genDens_5e13_Te_1400eV.nc',
             cql3dDest+'220627_genDens_5e13_Te_1600eV.nc',
             cql3dDest+'220627_genDens_5e13_Te_1800eV.nc',
             cql3dDest+'220627_genDens_5e13_Te_2000eV.nc']]

# filenameArr=[[cql3dDest+'220707_genDens_1e13_Te_100eV.nc',
#              cql3dDest+'220707_genDens_1e13_Te_200eV.nc',
#              cql3dDest+'220707_genDens_1e13_Te_300eV.nc',
#              cql3dDest+'220707_genDens_1e13_Te_400eV.nc',
#              cql3dDest+'220707_genDens_1e13_Te_500eV.nc',
#              cql3dDest+'220707_genDens_1e13_Te_600eV.nc',
#              cql3dDest+'220707_genDens_1e13_Te_700eV.nc',
#              cql3dDest+'220707_genDens_1e13_Te_800eV.nc',
#              cql3dDest+'220707_genDens_1e13_Te_900eV.nc',
#              cql3dDest+'220707_genDens_1e13_Te_1000eV.nc'],
#              [cql3dDest+'220707_genDens_2e13_Te_100eV.nc',
#              cql3dDest+'220707_genDens_2e13_Te_200eV.nc',
#              cql3dDest+'220707_genDens_2e13_Te_300eV.nc',
#              cql3dDest+'220707_genDens_2e13_Te_400eV.nc',
#              cql3dDest+'220707_genDens_2e13_Te_500eV.nc',
#              cql3dDest+'220707_genDens_2e13_Te_600eV.nc',
#              cql3dDest+'220707_genDens_2e13_Te_700eV.nc',
#              cql3dDest+'220707_genDens_2e13_Te_800eV.nc',
#              cql3dDest+'220707_genDens_2e13_Te_900eV.nc',
#              cql3dDest+'220707_genDens_2e13_Te_1000eV.nc'],
#              [cql3dDest+'220707_genDens_3e13_Te_100eV.nc',
#              cql3dDest+'220707_genDens_3e13_Te_200eV.nc',
#              cql3dDest+'220707_genDens_3e13_Te_300eV.nc',
#              cql3dDest+'220707_genDens_3e13_Te_400eV.nc',
#              cql3dDest+'220707_genDens_3e13_Te_500eV.nc',
#              cql3dDest+'220707_genDens_3e13_Te_600eV.nc',
#              cql3dDest+'220707_genDens_3e13_Te_700eV.nc',
#              cql3dDest+'220707_genDens_3e13_Te_800eV.nc',
#              cql3dDest+'220707_genDens_3e13_Te_900eV.nc',
#              cql3dDest+'220707_genDens_3e13_Te_1000eV.nc'],
#              [cql3dDest+'220707_genDens_4e13_Te_100eV.nc',
#              cql3dDest+'220707_genDens_4e13_Te_200eV.nc',
#              cql3dDest+'220707_genDens_4e13_Te_300eV.nc',
#              cql3dDest+'220707_genDens_4e13_Te_400eV.nc',
#              cql3dDest+'220707_genDens_4e13_Te_500eV.nc',
#              cql3dDest+'220707_genDens_4e13_Te_600eV.nc',
#              cql3dDest+'220707_genDens_4e13_Te_700eV.nc',
#              cql3dDest+'220707_genDens_4e13_Te_800eV.nc',
#              cql3dDest+'220707_genDens_4e13_Te_900eV.nc',
#              cql3dDest+'220707_genDens_4e13_Te_1000eV.nc'],
#              [cql3dDest+'220707_genDens_5e13_Te_100eV.nc',
#              cql3dDest+'220707_genDens_5e13_Te_200eV.nc',
#              cql3dDest+'220707_genDens_5e13_Te_300eV.nc',
#              cql3dDest+'220707_genDens_5e13_Te_400eV.nc',
#              cql3dDest+'220707_genDens_5e13_Te_500eV.nc',
#              cql3dDest+'220707_genDens_5e13_Te_600eV.nc',
#              cql3dDest+'220707_genDens_5e13_Te_700eV.nc',
#              cql3dDest+'220707_genDens_5e13_Te_800eV.nc',
#              cql3dDest+'220707_genDens_5e13_Te_900eV.nc',
#              cql3dDest+'220707_genDens_5e13_Te_1000eV.nc']]

#Major axis
majAx=np.array([1e13,2e13,3e13,4e13,5e13])
majAxName=r'Density [cm$^{-3}$]'

#Minor axis
# minAx=np.array([200,400,600,800,1000,1200,1400,1600,1800,2000])
minAx=np.array([100,200,300,400,500,600,700,800,900,1000])
minAxName=r'T$_e$ [eV]'

#NBI Power
nbiPow=np.array([100e3,200e3,300e3,400e3,500e3]) #W

#Plot savename
plotSavename='dd_rx_rate_2d_param_scan_changing_NBI'

#Reaction rate
rxRateArr=[]

#Adjusted reaction rate
adjRxRateArr=[]

#Maximum beta array
maxBetaArr=[]

#Go over the major axis
for i in range(len(majAx)):
    
    #Reaction rate for each major axis value
    rxRateArrTemp=[]
    
    #Adjusted reaction rate for each major axis value
    adjRxRateArrTemp=[]
    
    #Maxmimum beta for each major axis value
    maxBetaArrTemp=[]
    
    #Go over the minor axis
    for j in range(len(minAx)):
        
        print(majAxName+' = '+str(majAx[i]))
        print(minAxName+' = '+str(minAx[j]))
        print('**********************')
        
        #Get the filename
        filename=filenameArr[i][j]
        
        #Get the 0D parameters
        zeroDParams=zero_d_parameters(filename,longParams=True)
        
        #Total DD reaction rate
        ddRate=zeroDParams['rxRate'][2]+zeroDParams['rxRate'][3]
        
        #Adjusted DD reaction rate
        adjDDRate=ddRate/nbiPow[i]
        
        #Maximum beta
        maxBeta=zeroDParams['maxBeta']
        
        #Append to the array
        rxRateArrTemp.append(ddRate)
        adjRxRateArrTemp.append(adjDDRate)
        maxBetaArrTemp.append(maxBeta)
        
    #Append to the array
    rxRateArr.append(rxRateArrTemp)
    adjRxRateArr.append(adjRxRateArrTemp)
    maxBetaArr.append(maxBetaArrTemp)
    
#Convert to numpy
rxRateArr=np.array(rxRateArr)
adjRxRateArr=np.array(adjRxRateArr)
maxBetaArr=np.array(maxBetaArr)

#%% Contour Plotting

#Initialize the plot
fig,ax=plt.subplots(figsize=(12,8))

#Plot the contour
# contourObj=ax.contourf(minAx,majAx,rxRateArr,levels=20)
contourObj=ax.contourf(minAx,majAx,adjRxRateArr,levels=20)
#Add black lines to distinguish contour lines
ax.contour(contourObj,colors='black')
#Add a colorbar
cbar=fig.colorbar(contourObj,ax=ax)
# cbar.set_label(r'DD reaction rate [s$^{-1}]')
cbar.set_label(r'DD reaction rate per W [s$^{-1} \cdot$ W$^{-1}$] ')

#Max beta contour values
maxBetaContourVals=[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1]
#Contour labels
maxBetaContourLabels={}
for i in range(len(maxBetaContourVals)):
    maxBetaContourLabels[maxBetaContourVals[i]]=r'$\beta$='+str(maxBetaContourVals[i])
#Plot the max beta contours
maxBetaContour=ax.contour(minAx,majAx,maxBetaArr,levels=maxBetaContourVals,
                          colors='white',linewidths=4,linestyles='dashed')
#Label the contours
ax.clabel(maxBetaContour,fmt=maxBetaContourLabels)

ax.set_ylabel(majAxName)
ax.set_xlabel(minAxName)
plt.title('NBI Power scaled DD reaction rate',pad=30)

ax.grid(True)
plt.show()

plt.savefig(plotDest+plotSavename+'.pdf',bbox_inches='tight')

#%% Run Scan (RF Power and Beam Power)

# #Location of CQL3D outputs
# rffilenameArr=['C:/Users/kunal/OneDrive - UW-Madison/WHAM/Data/CQL3D/211028_powerScan_0kW.nc',
#               'C:/Users/kunal/OneDrive - UW-Madison/WHAM/Data/CQL3D/211028_powerScan_100kW.nc',
#               'C:/Users/kunal/OneDrive - UW-Madison/WHAM/Data/CQL3D/211028_powerScan_200kW.nc',
#               'C:/Users/kunal/OneDrive - UW-Madison/WHAM/Data/CQL3D/211028_powerScan_300kW.nc',
#               'C:/Users/kunal/OneDrive - UW-Madison/WHAM/Data/CQL3D/211028_powerScan_400kW.nc',
#               'C:/Users/kunal/OneDrive - UW-Madison/WHAM/Data/CQL3D/211028_powerScan_500kW.nc',
#               'C:/Users/kunal/OneDrive - UW-Madison/WHAM/Data/CQL3D/211028_powerScan_600kW.nc',
#               'C:/Users/kunal/OneDrive - UW-Madison/WHAM/Data/CQL3D/211028_powerScan_700kW.nc',
#               'C:/Users/kunal/OneDrive - UW-Madison/WHAM/Data/CQL3D/211028_powerScan_800kW.nc',
#               'C:/Users/kunal/OneDrive - UW-Madison/WHAM/Data/CQL3D/211028_powerScan_900kW.nc',
#                 'C:/Users/kunal/OneDrive - UW-Madison/WHAM/Data/CQL3D/211028_powerScan_1000kW.nc',
#                 'C:/Users/kunal/OneDrive - UW-Madison/WHAM/Data/CQL3D/211028_powerScan_2000kW.nc',
#                 'C:/Users/kunal/OneDrive - UW-Madison/WHAM/Data/CQL3D/211028_powerScan_3000kW.nc',
#                 'C:/Users/kunal/OneDrive - UW-Madison/WHAM/Data/CQL3D/211028_powerScan_4000kW.nc',
#                 'C:/Users/kunal/OneDrive - UW-Madison/WHAM/Data/CQL3D/211028_powerScan_5000kW.nc',
#                 'C:/Users/kunal/OneDrive - UW-Madison/WHAM/Data/CQL3D/211028_powerScan_6000kW.nc',
#                 'C:/Users/kunal/OneDrive - UW-Madison/WHAM/Data/CQL3D/211028_powerScan_7000kW.nc',
#                 'C:/Users/kunal/OneDrive - UW-Madison/WHAM/Data/CQL3D/211028_powerScan_8000kW.nc',
#                 'C:/Users/kunal/OneDrive - UW-Madison/WHAM/Data/CQL3D/211028_powerScan_9000kW.nc',
#                 'C:/Users/kunal/OneDrive - UW-Madison/WHAM/Data/CQL3D/211028_powerScan_10000kW.nc']
# nbifilenameArr=['C:/Users/kunal/OneDrive - UW-Madison/WHAM/Data/CQL3D/211101_beamScan_0kW.nc',
#                 'C:/Users/kunal/OneDrive - UW-Madison/WHAM/Data/CQL3D/211101_beamScan_100kW.nc',
#                 'C:/Users/kunal/OneDrive - UW-Madison/WHAM/Data/CQL3D/211101_beamScan_200kW.nc',
#                 'C:/Users/kunal/OneDrive - UW-Madison/WHAM/Data/CQL3D/211101_beamScan_300kW.nc',
#                 'C:/Users/kunal/OneDrive - UW-Madison/WHAM/Data/CQL3D/211101_beamScan_400kW.nc',
#                 'C:/Users/kunal/OneDrive - UW-Madison/WHAM/Data/CQL3D/211101_beamScan_500kW.nc',
#                 'C:/Users/kunal/OneDrive - UW-Madison/WHAM/Data/CQL3D/211101_beamScan_600kW.nc',
#                 'C:/Users/kunal/OneDrive - UW-Madison/WHAM/Data/CQL3D/211101_beamScan_700kW.nc',
#                 'C:/Users/kunal/OneDrive - UW-Madison/WHAM/Data/CQL3D/211101_beamScan_800kW.nc',
#                 'C:/Users/kunal/OneDrive - UW-Madison/WHAM/Data/CQL3D/211101_beamScan_900kW.nc',
#                 'C:/Users/kunal/OneDrive - UW-Madison/WHAM/Data/CQL3D/211101_beamScan_1000kW.nc',
#                 'C:/Users/kunal/OneDrive - UW-Madison/WHAM/Data/CQL3D/211101_beamScan_2000kW.nc',
#                 'C:/Users/kunal/OneDrive - UW-Madison/WHAM/Data/CQL3D/211101_beamScan_3000kW.nc',
#                 'C:/Users/kunal/OneDrive - UW-Madison/WHAM/Data/CQL3D/211101_beamScan_4000kW.nc',
#                 'C:/Users/kunal/OneDrive - UW-Madison/WHAM/Data/CQL3D/211101_beamScan_5000kW.nc',
#                 'C:/Users/kunal/OneDrive - UW-Madison/WHAM/Data/CQL3D/211101_beamScan_6000kW.nc',
#                 'C:/Users/kunal/OneDrive - UW-Madison/WHAM/Data/CQL3D/211101_beamScan_7000kW.nc',
#                 'C:/Users/kunal/OneDrive - UW-Madison/WHAM/Data/CQL3D/211101_beamScan_8000kW.nc',
#                 'C:/Users/kunal/OneDrive - UW-Madison/WHAM/Data/CQL3D/211101_beamScan_9000kW.nc',
#                 'C:/Users/kunal/OneDrive - UW-Madison/WHAM/Data/CQL3D/211101_beamScan_10000kW.nc']

# #RF Power in each file
# rfPower=np.array([0,100,200,300,400,500,600,700,800,900,1000,2000,3000,4000,5000,6000,7000,8000,9000,10000])
# #NBI Power for RF Scan
# nbiNorm=1000

# #NBI Power in each file
# nbiPower=np.array([0,100,200,300,400,500,600,700,800,900,1000,2000,3000,4000,5000,6000,7000,8000,9000,10000])
# #RF Power for NBI Scan
# rfNorm=500

# #DT reaction rates
# rfDTRateArr=[]
# nbiDTRateArr=[]

# #Number of runs
# numRuns=len(rffilenameArr)

# #Initialize the plot
# fig,ax=plt.subplots(figsize=(12,8))

# #Go over each file
# for i in range(len(rffilenameArr)):
    
#     #Get the filenames
#     rffilename=rffilenameArr[i]
#     nbifilename=nbifilenameArr[i]
    
#     #RF Scan
#     #Get the fusion reactivity
#     fusrxrt,time=fusion_rx_rate(rffilename)
#     #Get the DT reaction rate at the last timestep
#     dtRate=fusrxrt[0,-1]
#     #Append to the array
#     rfDTRateArr.append(dtRate)
    
#     #NBI Scan
#     #Get the fusion reactivity
#     fusrxrt,time=fusion_rx_rate(nbifilename)
#     #Get the DT reaction rate at the last timestep
#     dtRate=fusrxrt[0,-1]
#     #Append to the array
#     nbiDTRateArr.append(dtRate)
    
# #Convert to numpy array
# rfDTRateArr=np.array(rfDTRateArr)
# nbiDTRateArr=np.array(nbiDTRateArr)

# #Plot the data
# ax.plot((rfPower+nbiNorm)/1000,rfDTRateArr,label=r'RF Scan (P$_{NBI}$=1MW)') #Convert power units to MW
# ax.plot((nbiPower+rfNorm)/1000,nbiDTRateArr,label=r'NBI Scan (P$_{RF}$=0.5MW)') #Convert power units to MW

# #Add labels
# ax.set_xlabel('Total Power [MW]')
# ax.set_ylabel(r'DT reaction rate [s$^{-1}$]')
# ax.set_title('Power Scan')

# # ax.set_xlim(min(rfPower/1000),max(rfPower/1000))
# ax.legend()
# ax.grid(which='both')
# plt.show()
# plt.savefig(plotDest+'tot_power_scan.png',bbox_to_inches='tight')

# #Plot Q_fus for each run
# #Calculate Q_fus
# pOut=nbiDTRateArr*17.6*1.602*1e-13*1e-6
# pIn=(rfPower+nbiNorm)/1000
# qFus=pOut/pIn

# #Plot the data
# plt.figure(figsize=(12,8))
# plt.plot(pIn,qFus)

# #Add labels
# plt.xlabel('Total Input Power [MW]')
# plt.ylabel(r'Q$_{fus}$')

# #Show the plot
# plt.grid()
# plt.show()

#%% Run Scan (Source Power Density)

# #Location of CQL3D outputs
# filenameArr=['C:/Users/kunal/OneDrive - UW-Madison/WHAM/Data/CQL3D/210625_100keV.nc',
#              'C:/Users/kunal/OneDrive - UW-Madison/WHAM/Data/CQL3D/210628_150keV.nc',
#              'C:/Users/kunal/OneDrive - UW-Madison/WHAM/Data/CQL3D/210628_200keV.nc',
#              'C:/Users/kunal/OneDrive - UW-Madison/WHAM/Data/CQL3D/210628_250keV.nc',
#              'C:/Users/kunal/OneDrive - UW-Madison/WHAM/Data/CQL3D/210628_300keV.nc']

# #Labels for each file (used in plotting)
# labelArr=['100keV','150keV','200keV','250keV','300keV']

# #Number of runs
# numRuns=len(filenameArr)

# #Initialize the plot
# fig,ax=plt.subplots(figsize=(12,8))

# #Evenly spaced array for changing colors in the plot
# colorArr=np.linspace(0,1,numRuns)

# #Go over each file
# for i in range(len(filenameArr)):
    
#     #Get the filename
#     filename=filenameArr[i]
    
#     #Get the power deposition profiles
#     rya,sorpw_nbi,sorpw_rf,sorpw_tot=source_power_dens(filename)
    
#     #Plot the NBI deposition profile
#     ax.plot(rya,sorpw_nbi[1],label=labelArr[i],color=(colorArr[i],0,0))
    
# #Add labels
# ax.set_xlabel('Normalized Radius (r/a)')
# ax.set_ylabel(r'Power Density [W/m$^3$]')
# ax.set_title('Source Power Density')

# ax.grid(True)
# ax.set_xlim(0,1)
# ax.legend(bbox_to_anchor=(1,1))
# plt.show()

# plt.savefig(plotDest+'NBI_power_scan_source_power_dens.png',bbox_to_anchor='tight')

#%% Run Scan (Average Ion Energy)

# #Location of CQL3D outputs
# filenameArr=['C:/Users/kunal/OneDrive - UW-Madison/WHAM/Data/CQL3D/210625_100keV.nc',
#              'C:/Users/kunal/OneDrive - UW-Madison/WHAM/Data/CQL3D/210628_150keV.nc',
#              'C:/Users/kunal/OneDrive - UW-Madison/WHAM/Data/CQL3D/210628_200keV.nc',
#              'C:/Users/kunal/OneDrive - UW-Madison/WHAM/Data/CQL3D/210628_250keV.nc',
#              'C:/Users/kunal/OneDrive - UW-Madison/WHAM/Data/CQL3D/210628_300keV.nc']

# #Labels for each file (used in plotting)
# labelArr=['100keV','150keV','200keV','250keV','300keV']

# #Number of runs
# numRuns=len(filenameArr)

# #Initialize the plot
# fig,ax=plt.subplots(figsize=(12,8))

# #Evenly spaced array for changing colors in the plot
# colorArr=np.linspace(0,1,numRuns)

# #Go over each file
# for i in range(len(filenameArr)):
    
#     #Get the filename
#     filename=filenameArr[i]
    
#     #Get the power deposition profiles
#     rya,aveEnergy=average_energy(filename)
    
#     #Plot the NBI deposition profile
#     ax.plot(rya,aveEnergy[1],label=labelArr[i],color=(colorArr[i],0,0))
    
# #Add labels
# ax.set_xlabel('Normalized Radius (r/a)')
# ax.set_ylabel('Particle Energy [eV]')
# ax.set_title('Average Particle Energy')

# ax.grid(True)
# ax.set_xlim(0,1)
# ax.legend(bbox_to_anchor=(1,1))
# plt.show()

# plt.savefig(plotDest+'NBI_power_scan_average_ion_energy.png',bbox_to_anchor='tight')

#%% Run Scan (Fast Ion Confinement Time)

# #Location of CQL3D outputs
# filenameArr=['C:/Users/kunal/OneDrive - UW-Madison/WHAM/Data/CQL3D/210625_100keV.nc',
#              'C:/Users/kunal/OneDrive - UW-Madison/WHAM/Data/CQL3D/210628_150keV.nc',
#              'C:/Users/kunal/OneDrive - UW-Madison/WHAM/Data/CQL3D/210628_200keV.nc',
#              'C:/Users/kunal/OneDrive - UW-Madison/WHAM/Data/CQL3D/210628_250keV.nc',
#              'C:/Users/kunal/OneDrive - UW-Madison/WHAM/Data/CQL3D/210628_300keV.nc']

# #Labels for each file (used in plotting)
# labelArr=['100keV','150keV','200keV','250keV','300keV']

# #Number of runs
# numRuns=len(filenameArr)

# #Initialize the plot
# fig,ax=plt.subplots(figsize=(12,8))

# #Evenly spaced array for changing colors in the plot
# colorArr=np.linspace(0,1,numRuns)

# #Go over each file
# for i in range(len(filenameArr)):
    
#     #Get the filename
#     filename=filenameArr[i]
    
#     #Get the power deposition profiles
#     rya,tauI=fast_ion_confinement_time(filename)
    
#     #Plot the NBI deposition profile
#     ax.plot(rya,tauI[1],label=labelArr[i],color=(colorArr[i],0,0))
    
# #Add labels
# ax.set_xlabel('Normalized Radius (r/a)')
# ax.set_ylabel('Time [s]')
# ax.set_title('Fast Ion Confinement Time')

# ax.grid(True)
# ax.set_xlim(0,1)
# ax.legend(bbox_to_anchor=(1,1))
# plt.show()

# plt.savefig(plotDest+'NBI_power_scan_fast_ion_confinement_time.png',bbox_to_anchor='tight')

#%% Run Scan (Antenna Position)

# #Location of CQL3D outputs
# filenameArr=['C:/Users/kunal/OneDrive - UW-Madison/WHAM/Data/CQL3D/210823_thgrill_60.nc',
#              'C:/Users/kunal/OneDrive - UW-Madison/WHAM/Data/CQL3D/210823_thgrill_61.nc',
#              'C:/Users/kunal/OneDrive - UW-Madison/WHAM/Data/CQL3D/210823_thgrill_62.nc',
#              'C:/Users/kunal/OneDrive - UW-Madison/WHAM/Data/CQL3D/210823_thgrill_63.nc',
#              'C:/Users/kunal/OneDrive - UW-Madison/WHAM/Data/CQL3D/210823_thgrill_64.nc',
#              'C:/Users/kunal/OneDrive - UW-Madison/WHAM/Data/CQL3D/210823_thgrill_65.nc',
#              'C:/Users/kunal/OneDrive - UW-Madison/WHAM/Data/CQL3D/210823_thgrill_66.nc',
#              'C:/Users/kunal/OneDrive - UW-Madison/WHAM/Data/CQL3D/210823_thgrill_67.nc',
#              'C:/Users/kunal/OneDrive - UW-Madison/WHAM/Data/CQL3D/210823_thgrill_68.nc',
#              'C:/Users/kunal/OneDrive - UW-Madison/WHAM/Data/CQL3D/210823_thgrill_69.nc',
#              'C:/Users/kunal/OneDrive - UW-Madison/WHAM/Data/CQL3D/210823_thgrill_70.nc',
#              'C:/Users/kunal/OneDrive - UW-Madison/WHAM/Data/CQL3D/210823_thgrill_71.nc',
#              'C:/Users/kunal/OneDrive - UW-Madison/WHAM/Data/CQL3D/210823_thgrill_72.nc',
#              'C:/Users/kunal/OneDrive - UW-Madison/WHAM/Data/CQL3D/210823_thgrill_73.nc',
#              'C:/Users/kunal/OneDrive - UW-Madison/WHAM/Data/CQL3D/210823_thgrill_74.nc']

# #Labels for each file (used in plotting)
# labelArr=[r'$\theta$=60',r'$\theta$=61',r'$\theta$=62',r'$\theta$=63',r'$\theta$=64',
#           r'$\theta$=65',r'$\theta$=66',r'$\theta$=67',r'$\theta$=68',r'$\theta$=69',
#           r'$\theta$=70',r'$\theta$=71',r'$\theta$=72',r'$\theta$=73',r'$\theta$=74',]

# #Number of runs
# numRuns=len(filenameArr)

# #Initialize the plot
# fig,ax=plt.subplots(figsize=(12,8))

# #Evenly spaced array for changing colors in the plot
# colorArr=np.linspace(0,1,numRuns)

# #Go over each file
# for i in range(len(filenameArr)):
    
#     #Get the filename
#     filename=filenameArr[i]
    
#     #Get the fusion reaction rate
#     fusrxrt,tArr=fusion_rx_rate(filename)
    
#     #Compare the DT branch
#     ax.plot(tArr,fusrxrt[0],label=labelArr[i],color=(colorArr[i],0,0))
    
# #Add labels
# ax.set_xlabel('Time [s]')
# ax.set_ylabel(r'Fusion Reaction Rate [s$^{-1}$]')
# ax.set_title('DT Reaction Rate')

# ax.legend(bbox_to_anchor=(1.01,0.27),loc=3,borderaxespad=0)
# ax.grid(True)

# plt.show()

# plt.savefig(plotDest+'Antenna_Position_Scan.png',bbox_inches='tight')

#%% Density vs Beam Current

# #Location of CQL3D outputs
# filenameArr=['C:/Users/kunal/OneDrive - UW-Madison/WHAM/Data/CQL3D/210621_dtr_10ms.nc',
#              'C:/Users/kunal/OneDrive - UW-Madison/WHAM/Data/CQL3D/210621_dtr_5ms.nc',
#              'C:/Users/kunal/OneDrive - UW-Madison/WHAM/Data/CQL3D/210621_dtr_2ms.nc',
#              'C:/Users/kunal/OneDrive - UW-Madison/WHAM/Data/CQL3D/210621_dtr_1ms.nc']

# #Labels for each file (used in plotting)
# labelArr=['10ms','5ms','2ms','1ms']

# #Generate a table with densities
# table=[['Timestep','D + T --> n + 4He','D + 3He --> p + 4He','D + D --> n + 3He','D + D --> p + T']]

# beamCurrent=np.array([1,2,5,10])

# fusPwr=[]

# fig,ax=plt.subplots()

# #Go over each file
# for i in range(len(filenameArr)):
    
#     filename=filenameArr[i]
    
#     currPwr=list(total_fusion_power(filename))
    
#     fusPwr.append(currPwr)
    
#     table.append([labelArr[i]]+currPwr)
    
# fusPwr=np.array(fusPwr)

# ax.plot(beamCurrent,fusPwr[:,0])
# ax.scatter(beamCurrent,fusPwr[:,0])
# ax.grid()
# ax.set_xlim(0,10)
# ax.set_ylim(0,1.3e6)
# plt.show()
    
# print(tabulate(table,headers='firstrow'))

# fig,axs=plt.subplots(3,1)

# #Total ion count
# warmArr=[]
# fastArr=[]
# totArr=[]

# #Go over each file
# for i in range(len(filenameArr)):
    
#     filename=filenameArr[i]
    
#     #Get the ion densities
#     ndwarmz,ndfz,ndtotz,solrz,solzz=ion_dens(filename,species=0)
    
#     #Radial array
#     rArr=solrz[:,0]
    
#     #Sum densities over z
#     warmZSum=[]
#     fastZSum=[]
#     totZSum=[]
    
#     #Go over each flux surface
#     for j in range(len(solrz)):
        
#         #zArr
#         zArr=solzz[i,:]
        
#         #Integrate over z
#         warmZSum.append(np.trapz(ndwarmz[j],zArr))
#         fastZSum.append(np.trapz(ndfz[j],zArr))
#         totZSum.append(np.trapz(ndtotz[j],zArr))
        
#     #Convert to numpy array
#     warmZSum=np.array(warmZSum)
#     fastZSum=np.array(fastZSum)
#     totZSum=np.array(totZSum)
     
#     #Integrate over r
#     warmArr.append(np.trapz(2*np.pi*warmZSum*rArr,rArr))
#     fastArr.append(np.trapz(2*np.pi*fastZSum*rArr,rArr))
#     totArr.append(np.trapz(2*np.pi*totZSum*rArr,rArr))
    
#     #Plot each density profile
#     #Warm ions
#     ax=axs[0]
#     ax.plot(rArr,warmZSum,label=labelArr[i])
#     #Fast ions
#     ax=axs[1]
#     ax.plot(rArr,fastZSum,label=labelArr[i])
#     #Total ions
#     ax=axs[2]
#     ax.plot(rArr,totZSum,label=labelArr[i])
    
# #Convert to numpy array
# warmArr=np.array(warmArr)
# fastArr=np.array(fastArr)
# totArr=np.array(totArr)
    
# #Add titles
# axs[0].set_title('Warm Ions')
# axs[1].set_title('Fast Ions')
# axs[2].set_title('Total Ions')

# #Add xlabel
# axs[0].set_xlabel('r [m]')
# axs[1].set_xlabel('r [m]')
# axs[2].set_xlabel('r [m]')

# #Add ylabel
# axs[0].set_ylabel(r'$\int \rho \cdot dz$')
# axs[1].set_ylabel(r'$\int \rho \cdot dz$')
# axs[2].set_ylabel(r'$\int \rho \cdot dz$')

# #Add legends
# axs[0].legend()
# axs[1].legend()
# axs[2].legend()

# #Add grid
# axs[0].grid()
# axs[1].grid()
# axs[2].grid()

# plt.show()

# #Generate a table with densities
# table=[['Timestep','Fast Ions','Warm Ions','Total Ions']]

# #Go over each file
# for i in range(len(filenameArr)):
    
#     table.append([labelArr[i],fastArr[i],warmArr[i],totArr[i]])
    
# #Print the table
# print(tabulate(table,headers='firstrow'))

#%% Plasma Group Talk (7th April, 2022)

# =============================================================================
# NBI Only
# =============================================================================
filenameArrNBI=[cql3dDest+'220329_nbi_0kW.nc',
                cql3dDest+'220329_nbi_10kW.nc',
                cql3dDest+'220329_nbi_25kW.nc',
                cql3dDest+'220329_nbi_50kW.nc',
                cql3dDest+'220329_nbi_100kW.nc',
                cql3dDest+'220329_nbi_250kW.nc',
                cql3dDest+'220329_nbi_500kW.nc',
                cql3dDest+'220329_nbi_1000kW.nc']

#NBI DD reaction rates with absorbed power
ddRateArrNBI=[]
#NBI Power absorbed
NBIabsorbed=[]

#Go over each file
for i in range(len(filenameArrNBI)):
    
    #Get the filename
    filename=filenameArrNBI[i]
    
    #Get the fusion reactivity
    fusrxrt,time=fusion_rx_rate(filename)
    
    #Get the DD reaction rate at the last timestep
    ddRate=fusrxrt[2,-1]+fusrxrt[3,-1]
    
    #Append to the array
    ddRateArrNBI.append(ddRate) #s^{-1}
    
    #Get the 0D parameters
    zeroDParams=zero_d_parameters(filename)
    
    #Get the absorbed NBI power
    NBIabsorbed.append(zeroDParams['nbiPow']/1000) #kW

# =============================================================================
# RF Only
# =============================================================================
filenameArrRF=[cql3dDest+'220404_rf_0kW.nc',
               cql3dDest+'220404_rf_10kW.nc',
               cql3dDest+'220404_rf_20kW.nc',
               cql3dDest+'220404_rf_30kW.nc',
               cql3dDest+'220404_rf_40kW.nc',
               cql3dDest+'220404_rf_50kW.nc',
               cql3dDest+'220404_rf_60kW.nc',
               cql3dDest+'220404_rf_70kW.nc',
               cql3dDest+'220404_rf_80kW.nc',
               cql3dDest+'220404_rf_90kW.nc',
               cql3dDest+'220404_rf_100kW.nc',
               cql3dDest+'220404_rf_125kW.nc',
               cql3dDest+'220404_rf_150kW.nc',
               cql3dDest+'220404_rf_175kW.nc']
    
#NBI DD reaction rates with absorbed power
ddRateArrRF=[]
#NBI Power absorbed
RFabsorbed=[]

#Go over each file
for i in range(len(filenameArrRF)):
    
    #Get the filename
    filename=filenameArrRF[i]
    
    #Get the fusion reactivity
    fusrxrt,time=fusion_rx_rate(filename)
    
    #Get the DD reaction rate at the last timestep
    ddRate=fusrxrt[2,-1]+fusrxrt[3,-1]
    
    #Append to the array
    ddRateArrRF.append(ddRate) #s^{-1}
    
    #Get the 0D parameters
    zeroDParams=zero_d_parameters(filename)
    
    #Get the absorbed RF power
    RFabsorbed.append(zeroDParams['rfPow']/1000) #kW
    
# =============================================================================
# NBI+RF
# =============================================================================
filenameArrBoth=[cql3dDest+'220404_rf_50kW_nbi_980kW.nc',
                 cql3dDest+'220404_rf_100kW_nbi_870kW.nc',
                 cql3dDest+'220404_rf_150kW_nbi_760kW.nc',
                 cql3dDest+'220404_rf_200kW_nbi_650kW.nc',
                 cql3dDest+'220404_rf_250kW_nbi_540kW.nc',
                 cql3dDest+'220404_rf_300kW_nbi_430kW.nc']

#NBI DD reaction rates with absorbed power
ddRateArrBoth=[]
#Total Power absorbed
TOTabsorbed=[]
#RF to total ratio
rfNBIRatio=[]

#Go over each file
for i in range(len(filenameArrBoth)):
    
    #Get the filename
    filename=filenameArrBoth[i]
    
    #Get the fusion reactivity
    fusrxrt,time=fusion_rx_rate(filename)
    
    #Get the DD reaction rate at the last timestep
    ddRate=fusrxrt[2,-1]+fusrxrt[3,-1]
    
    #Append to the array
    ddRateArrBoth.append(ddRate) #s^{-1}
    
    #Get the 0D parameters
    zeroDParams=zero_d_parameters(filename)
    
    #Get the absorbed total power
    TOTabsorbed.append((zeroDParams['rfPow']+zeroDParams['nbiPow'])/1000) #kW
    
    #Get the RF to total ratio
    rfNBIRatio.append(zeroDParams['rfPow']/(zeroDParams['rfPow']+zeroDParams['nbiPow']))    

# =============================================================================
# Plotting
# =============================================================================
#Initialize the plot
fig,ax=plt.subplots(figsize=(12,8))

#Plot the data
ax.plot(NBIabsorbed,ddRateArrNBI,label='NBI Only',color='blue')
ax.scatter(NBIabsorbed,ddRateArrNBI,color='blue')
ax.plot(RFabsorbed,ddRateArrRF,label='RF Only',color='red')
ax.scatter(RFabsorbed,ddRateArrRF,color='red')
ax.plot(TOTabsorbed,ddRateArrBoth,label='NBI+RF',color='green')
ax.scatter(TOTabsorbed,ddRateArrBoth,color='green')

#Add labels
ax.set_xlabel('Absorbed Power [kW]')
ax.set_ylabel(r'Peak DD reaction rate [s$^{-1}$]')
ax.set_title('Power Scan')

#Add beta values text
ax.text(260,18.0e12,r'$\beta$=0.54, $r_{RF}$='+str(np.round(rfNBIRatio[5][0],2)))
ax.text(240,13.5e12,r'$\beta$=0.49, $r_{RF}$='+str(np.round(rfNBIRatio[4][0],2)))
ax.text(215,9.75e12,r'$\beta$=0.45, $r_{RF}$='+str(np.round(rfNBIRatio[3][0],2)))
ax.text(195,7.22e12,r'$\beta$=0.38, $r_{RF}$='+str(np.round(rfNBIRatio[2][0],2)))
ax.text(172,5.00e12,r'$\beta$=0.31, $r_{RF}$='+str(np.round(rfNBIRatio[1][0],2)))
ax.text(150,3.26e12,r'$\beta$=0.22, $r_{RF}$='+str(np.round(rfNBIRatio[0][0],2)))
ax.text(35,0.95e13,r'$\beta$=0.66')

# ax.set_xlim(min(NBIabsorbed),max(NBIabsorbed))
ax.grid(which='both')
ax.legend(loc='best')
plt.show()
plt.savefig(plotDest+'dd_rx_rate_compare.pdf')

#%% Compare RF absorption fractions

filenameArr1=[cql3dDest+'220414_nbi_10keV_100kW.nc',
              cql3dDest+'220414_nbi_15keV_150kW.nc',
              cql3dDest+'220414_nbi_17keV_170kW.nc',
              cql3dDest+'220414_nbi_19keV_190kW.nc',
              cql3dDest+'220414_nbi_20keV_200kW.nc',
              cql3dDest+'220414_nbi_21keV_210kW.nc',
              cql3dDest+'220414_nbi_22keV_220kW.nc',
              cql3dDest+'220414_nbi_25keV_250kW.nc',
              cql3dDest+'220414_nbi_27keV_270kW.nc',
              cql3dDest+'220414_nbi_29keV_290kW.nc',
              cql3dDest+'220414_nbi_30keV_300kW.nc',
              cql3dDest+'220414_nbi_31keV_310kW.nc',
              cql3dDest+'220414_nbi_33keV_330kW.nc',
              cql3dDest+'220414_nbi_35keV_350kW.nc',
              cql3dDest+'220414_nbi_45keV_450kW.nc',
              cql3dDest+'220414_nbi_55keV_550kW.nc',
              cql3dDest+'220414_nbi_65keV_650kW.nc']
filenameArr2=[cql3dDest+'220418_only2nd_10keV.nc',
              cql3dDest+'220418_only2nd_15keV.nc',
              cql3dDest+'220418_only2nd_17keV.nc',
              cql3dDest+'220418_only2nd_19keV.nc',
              cql3dDest+'220418_only2nd_20keV.nc',
              cql3dDest+'220418_only2nd_21keV.nc',
              cql3dDest+'220418_only2nd_22keV.nc',
              cql3dDest+'220418_only2nd_25keV.nc',
              cql3dDest+'220418_only2nd_35keV.nc',
              cql3dDest+'220418_only2nd_45keV.nc',
              cql3dDest+'220418_only2nd_55keV.nc',
              cql3dDest+'220418_only2nd_65keV.nc']
filenameArr3=[cql3dDest+'220428_only3rd_10keV.nc',
              cql3dDest+'220428_only3rd_20keV.nc',
              cql3dDest+'220428_only3rd_30keV.nc',
              cql3dDest+'220428_only3rd_40keV.nc',
              cql3dDest+'220428_only3rd_50keV.nc',
              cql3dDest+'220428_only3rd_60keV.nc',
              cql3dDest+'220428_only3rd_70keV.nc',
              cql3dDest+'220428_only3rd_80keV.nc',
              cql3dDest+'220428_only3rd_90keV.nc',
              cql3dDest+'220428_only3rd_100keV.nc',
              cql3dDest+'220428_only3rd_110keV.nc',
              cql3dDest+'220428_only3rd_125keV.nc']

numRuns1=len(filenameArr1)
numRuns2=len(filenameArr2)
numRuns3=len(filenameArr3)

paramArr1=np.array([10,15,17,19,20,21,22,25,27,29,30,31,33,35,45,55,65])
paramArr2=np.array([10,15,17,19,20,21,22,25,35,45,55,65])
paramArr3=np.array([10,20,30,40,50,60,70,80,90,100,110,125])

plotName='rf_absorption_comparison'

plotTitle='RF Absorption Comparison'

xAxLabel='NBI Energy [keV]'

#Go over all harmonic runs
rfPower=np.array([300]*numRuns1) #kW

absorbFracArr1=[]

#Go over each file
for i in range(len(filenameArr1)):
    
    #Get the filename
    filename=filenameArr1[i]
    
    #Get the amount of NBI absorbed
    zeroDParams=zero_d_parameters(filename)
    
    #Calculate the absorption fraction
    absorbFrac=100*zeroDParams['rfPow']/(rfPower[i]*1000)
    
    #Append to the array
    absorbFracArr1.append(absorbFrac)
    
#Convert to numpy
absorbFracArr1=np.array(absorbFracArr1)

#Go over 2nd harmonic runs
rfPower=np.array([300]*numRuns2) #kW

absorbFracArr2=[]

#Go over each file
for i in range(len(filenameArr2)):
    
    #Get the filename
    filename=filenameArr2[i]
    
    #Get the amount of NBI absorbed
    zeroDParams=zero_d_parameters(filename)
    
    #Calculate the absorption fraction
    absorbFrac=100*zeroDParams['rfPow']/(rfPower[i]*1000)
    
    #Append to the array
    absorbFracArr2.append(absorbFrac)
    
#Convert to numpy
absorbFracArr2=np.array(absorbFracArr2)

#Go over 2nd harmonic runs
rfPower=np.array([300]*numRuns3) #kW

absorbFracArr3=[]

#Go over each file
for i in range(len(filenameArr3)):
    
    #Get the filename
    filename=filenameArr3[i]
    
    #Get the amount of NBI absorbed
    zeroDParams=zero_d_parameters(filename)
    
    #Calculate the absorption fraction
    absorbFrac=100*zeroDParams['rfPow']/(rfPower[i]*1000)
    
    #Append to the array
    absorbFracArr3.append(absorbFrac)
    
#Convert to numpy
absorbFracArr3=np.array(absorbFracArr3)

#Initialize the plot
fig,ax=plt.subplots(figsize=(12,8))

#Scientific notation
plt.ticklabel_format(axis='y',style='sci',scilimits=(0,0))

#Plot the data
ax.semilogy(paramArr1,absorbFracArr1,label='All harmonics')
ax.scatter(paramArr1,absorbFracArr1)
ax.semilogy(paramArr2,absorbFracArr2,label='Only 2nd')
ax.scatter(paramArr2,absorbFracArr2)
ax.semilogy(paramArr3,absorbFracArr3,label='Only 3rd')
ax.scatter(paramArr3,absorbFracArr3)

#Add labels
ax.set_xlabel(xAxLabel)
ax.set_ylabel(r'RF Absorption Percentage [%]')
ax.set_title(plotTitle)

ax.legend(loc='best')
ax.set_xlim(min(min(paramArr1),min(paramArr2),min(paramArr3)),max(max(paramArr1),max(paramArr2),max(paramArr3)))
ax.set_ylim(0,100)
ax.grid(which='both')
plt.show()
plt.savefig(plotDest+plotName+'_rf_absorption_fraction.pdf')

#%% T_e scan

rxRates=[]

#Initialize the plot
fig,axs=plt.subplots(numRuns,1,figsize=(21,numRuns*9))

#Go over each file
for i in range(numRuns):
    
    #Get the filename
    filename=filenameArr[i]
    
    #Get the fusion reactivity
    fusrxrt,tArr=fusion_rx_rate(filename)
    
    #Get the total fusion reactivity at the last timestep
    rxRates.append(sum(fusrxrt[:,-1]))
    
    #Get the distribution function data
    distData,vPar,vPerp=dist_func(filename)
    
    #Convert the data to log
    logData=np.log10(distData)
    
    #Max and min of the distribution function
    maxDist=np.max(logData)
    minDist=maxDist-15
    
    #Create the distribution function plot
    ax=axs[i]
    pltObj=ax.contourf(vPar[0],vPerp[0],logData[0],levels=np.linspace(minDist,maxDist,30))
    ax.contour(pltObj,colors='black')
    ax.set_xlabel(r'$v_{||}$ [m/s]')
    ax.set_xlim(-2e6,2e6)
    ax.set_xticks(np.linspace(-2e6,2e6,17))
    ax.set_ylabel(r'$v_{\perp}$ [m/s]')
    ax.set_ylim(0,2e6)
    ax.set_title(r'Maxwellian $T_i$='+str(paramArr[i])+'keV')
    ax.grid(True)
    cbar=fig.colorbar(pltObj,ax=ax)
    cbar.set_label(r'log$_{10}$(v$^{-3}$)')
    
plt.savefig(plotDest+'dist_funcs_maxTiScan.pdf',bbox_inches='tight')
plt.close()
# plt.show()

#Reaction rate plot
fig=plt.figure(figsize=(8,8))
ax=fig.add_subplot(111)
ax.plot(paramArr,rxRates)
ax.set_xlabel(r'$T_e$ [keV]')
ax.set_ylabel(r'Reaction rate [$s^{-1}$]')
ax.grid(True)

plt.savefig(plotDest+'rx_rate_maxTiScan.pdf',bbox_inches='tight')
plt.close()

#%% Plots for paper (Get data)

#With RF simulation
filenameWithRF=cql3dDest+'230131_withRF2.nc'
#Without RF simulation
filenameWithoutRF=cql3dDest+'230207_withoutRF2.nc'

#Plot savenames
radSavename='radial_profiles.png'
axialSavename='axial_profiles.png'

# =============================================================================
# Get the data
# =============================================================================

#Without RF

#Get the radial density profile
radArr,radDensWithoutRF=radial_density_profile(filenameWithoutRF)
#Get the radial fusion flux
rya,fusPowerWithoutRF=radial_fusion_power(filenameWithoutRF)

#Get the axial density profile
axialArrWithoutRF,axialDensWithoutRF=axial_density_profile(filenameWithoutRF)
# Get the axial fusion neutron flux
fusArrWithoutRF,zArr=axial_neutron_flux(filenameWithoutRF)

#With RF

#Get the radial density profile
radArr,radDensWithRF=radial_density_profile(filenameWithRF)
#Get the radial fusion flux
rya,fusPowerWithRF=radial_fusion_power(filenameWithRF)

#Get the axial density profile
axialArrWithRF,axialDensWithRF=axial_density_profile(filenameWithRF)
# Get the axial fusion neutron flux
fusArrWithRF,zArr=axial_neutron_flux(filenameWithRF)

# =============================================================================

#%% Plots for paper (Plot data)

#Plot the radial profiles
fig,ax=plt.subplots(figsize=(16,8))

ax.plot(rya,fusPowerWithoutRF[2]+fusPowerWithoutRF[3],label='Fusion Flux (no RF)', color='red', linewidth=3)
ax.plot(rya,fusPowerWithRF[2]+fusPowerWithRF[3],label='Fusion Flux (with RF)', color='red', linestyle='dashed', linewidth=3)

ax2=ax.twinx()

ax2.plot(rya,radDensWithoutRF,label='Density (no RF)', color='blue', linewidth=3)
ax2.plot(rya,radDensWithRF,label='Density (with RF)', color='blue', linestyle='dashed', linewidth=3)

ax.set_xlabel('Normalized Radius')
ax.set_xlim(0,1)
ax.yaxis.set_ticks([])
ax2.yaxis.set_ticks([])

ax.legend(bbox_to_anchor=(1,1))
ax2.legend(bbox_to_anchor=(0.955,0.82))

plt.show()
plt.savefig(plotDest+radSavename,bbox_inches='tight',dpi=300)

#Plot the axial profiles
fig,ax=plt.subplots(figsize=(16,8))

ax.plot(zArr,fusArrWithoutRF,label='Fusion Flux (no RF)', color='red', linewidth=3)
ax.plot(zArr,fusArrWithRF,label='Fusion Flux (with RF)', color='red', linestyle='dashed', linewidth=3)

ax2=ax.twinx()

ax2.plot(axialArrWithoutRF,axialDensWithoutRF,label='Density (no RF)', color='blue', linewidth=3)
ax2.plot(axialArrWithRF,axialDensWithRF,label='Density (with RF)', color='blue', linestyle='dashed', linewidth=3)

ax.set_xlabel('Z [m]')
ax.set_xlim(0,1)
ax.yaxis.set_ticks([])
ax2.yaxis.set_ticks([])

ax.legend(bbox_to_anchor=(1,1))
ax2.legend(bbox_to_anchor=(0.955,0.82))

plt.show()
plt.savefig(plotDest+axialSavename,bbox_inches='tight',dpi=300)