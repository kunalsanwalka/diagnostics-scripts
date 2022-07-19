# -*- coding: utf-8 -*-
"""
Created on Fri Apr  2 10:39:12 2021

@author: kunal

This program is used to analyse results from MST.

NOTE- Before running this program, make sure to connect via GlobalProtect else
      this program will not be able access files on the MST database.
"""

import numpy as np
import MDSplus as mds
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
plt.rcParams.update({'font.size': 22})

def variable_data(shotnum,nodepath):
    """
    This function returns the data and associated time array for a given data
    stream for a particular shot.

    Parameters
    ----------
    shotnum : int
        Shot number.
    nodepath : string
        Name of the data stream.

    Returns
    -------
    dataArr : np.array
        Array with the values of the data.
    tArr : np.array
        Associated time array.
    """
    
    #Connect to the server
    conn=mds.Connection(conn_name)
    
    #Open the connection
    conn.openTree(tree_name,shotnum)
    
    #Get the node for the data
    node=conn.get(nodepath)
    
    #Get the time array
    tArr=np.array(conn.get('dim_of({}, 0)'.format(nodepath)))
    
    #Get the data array
    dataArr=node.data()
    
    return dataArr,tArr

def time_trim(valArr,tArr,startTime,stopTime):
    """
    This function trims the data and time arrays based on a given start and
    stop time. Useful when doing integratations etc.

    Parameters
    ----------
    valArr : np.array
        Array with the data values.
    tArr : np.array
        Time array.
    startTime : float
        Start time.
    stopTime : float
        End time.

    Returns
    -------
    valArr : np.array
        Trimmed value array.
    tArr : np.array
        Trimmed time array.
    """
    
    #Get the index of the start and stop times
    startInd=np.where(tArr>=startTime)[0][0]
    stopInd=np.where(tArr<=stopTime)[0][-1]
    
    #Trim the arrays
    tArr=tArr[startInd:stopInd]
    valArr=valArr[startInd:stopInd]
    
    return valArr,tArr

def align_time(rdbkArr,rdbkTArr,beamCurrTArr,beamVoltTArr):
    """
    This function aligns the time arrays for the neutral beam with the MST
    clock. This is required since the neutral beam runs on its own clock.

    Parameters
    ----------
    rdbkArr : np.array
        Indicates the start and stop time of the neutral beam.
    rdbkTArr : np.array
        Time array of the readback which runs on the MST clock.
    beamCurrTArr : np.array
        Time array for the beam current.
    beamVoltTArr : np.array
        Time array for the beam voltage.

    Returns
    -------
    beamCurrTArr : np.array
        Corrected current timing array.
    beamVoltTArr : np.array
        Corrected voltage timing array.
    """
    
    #Get the index of the NBI start time
    beamStartInd=np.where(rdbkArr>3.5)[0][0]
    
    #Get the actual start time
    beamStartTime=rdbkTArr[beamStartInd]
    
    #Subtract 100ms from the NBI timing arrays
    beamCurrTArr-=0.1
    beamVoltTArr-=0.1
    
    #Add the start time based on the readback
    beamCurrTArr+=beamStartTime
    beamVoltTArr+=beamStartTime
    
    return beamCurrTArr,beamVoltTArr

def neutron_dc_offset(signalArr,tArr,startTime):
    
    #Get the index of the start time
    startTimeInd=np.where(tArr>=startTime)[0][0]
    
    #Get the neutron signal before the start time
    dcSignal=signalArr[:startTimeInd]
    
    #Calculate the DC offset
    dcOffset=np.average(dcSignal)
    
    #Subtract the offset from the raw data
    newSignal=signalArr-dcOffset
    
    return newSignal

#%% Analysis Variables

#Global variables
tree_name='mst'
conn_name='dave.physics.wisc.edu'

#%% Neutron Flux vs. Total Beam Charge

#Shot Numbers
shotnumArr=[1210329012,\
            1210329013,\
            1210329014,\
            1210329015,\
            1210329016,\
            1210329017,\
            1210329018,\
            1210329019,\
            1210329020,\
            1210329021,\
            1210329022,\
            1210329023,\
            1210329024,\
            1210329025,\
            1210329026,\
            1210329027,\
            1210329028,\
            1210329029,\
            1210329030,\
            1210329031,\
            1210329032]

#Neutron flux array
nFluxArr=[]
#Fast ion array
fIonArr=[]

#Go over each shot
for shotnum in shotnumArr:
    
    print('Analyzing shot number- '+str(shotnum))

    #Get the neutron flux data
    nodepath_neutron='\mraw_misc::wham_1n_det'
    nFlux,nFluxTArr=variable_data(shotnum,nodepath_neutron)
    #Flip the neutron flux signal because of the PMT bias
    nFlux=-nFlux
    
    #Get the beam current
    nodepath_beamCurr='\mraw_nbi::top:nbi_i_beam'
    beamCurr,beamCurrTArr=variable_data(shotnum,nodepath_beamCurr)
    #Convert to s
    beamCurrTArr/=1000
    
    #Get the beam voltage
    nodepath_beamCurr='\mraw_nbi::top:nbi_u_beam'
    beamVolt,beamVoltTArr=variable_data(shotnum,nodepath_beamCurr)
    #Convert to s
    beamVoltTArr/=1000
    
    #Beam readback
    nodepath_nbi_start='\mraw_nbi::nbi_rdbk'
    rdbkArr,rdbkTArr=variable_data(shotnum,nodepath_nbi_start)
    
    #Correct the NBI timing arrays
    beamCurrTArr,beamVoltTArr=align_time(rdbkArr,rdbkTArr,beamCurrTArr,beamVoltTArr)
    
    #Trim the neutron detector time interval
    timeInterval=0.001
    #Get the start time based on the readback
    startTimeInd=np.where(rdbkArr>3.5)[0][0]
    startTime=rdbkTArr[startTimeInd]+0.003
    stopTime=startTime+timeInterval
    
    #Remove the DC offset from the neutron detector signal
    nFlux=neutron_dc_offset(nFlux,nFluxTArr,startTime)
    
    #Trim the signal arrays
    #Neutron singal
    nFluxTrimmed,nFluxTArrTrimmed=time_trim(nFlux,nFluxTArr,startTime,stopTime)
    #Beam Current
    beamCurrTrimmed,beamCurrTArrTrimmed=time_trim(beamCurr,beamCurrTArr,startTime,stopTime)
    
    #Get the total detected neutron count
    neutronFlux=np.trapz(nFluxTrimmed,nFluxTArrTrimmed)
    
    #Get the total number of fast ions
    fastIonTot=np.trapz(beamCurrTrimmed,beamCurrTArrTrimmed)
    
    #Append data to the arrays
    nFluxArr.append(neutronFlux)
    fIonArr.append(fastIonTot)
    
#Conver the arrays to numpy
nFluxArr=np.array(nFluxArr)
fIonArr=np.array(fIonArr)

#%% Fusion Cross Section

#Shot Numbers
shotnumArr=[1210326019,\
            1210326020,\
            # 1210326021,\
            1210326022,\
            1210326023,\
            1210326024,\
            1210326025,\
            1210326026,\
            1210326027]

#Cross section array
crossArr=[]
#Particle energy array
pEnergyArr=[]

#Go over each shot
for shotnum in shotnumArr:
    
    print('Analyzing shot number- '+str(shotnum))

    #Get the neutron flux data
    nodepath_neutron='\mraw_misc::wham_1n_det'
    nFlux,nFluxTArr=variable_data(shotnum,nodepath_neutron)
    #Flip the neutron flux signal because of the PMT bias
    nFlux=-nFlux
    
    #Get the beam current
    nodepath_beamCurr='\mraw_nbi::top:nbi_i_beam'
    beamCurr,beamCurrTArr=variable_data(shotnum,nodepath_beamCurr)
    #Convert to s
    beamCurrTArr/=1000
    
    #Get the beam voltage
    nodepath_beamCurr='\mraw_nbi::top:nbi_u_beam'
    beamVolt,beamVoltTArr=variable_data(shotnum,nodepath_beamCurr)
    #Convert to s
    beamVoltTArr/=1000
    
    #Beam readback
    nodepath_nbi_start='\mraw_nbi::nbi_rdbk'
    rdbkArr,rdbkTArr=variable_data(shotnum,nodepath_nbi_start)
    
    #Correct the NBI timing arrays
    beamCurrTArr,beamVoltTArr=align_time(rdbkArr,rdbkTArr,beamCurrTArr,beamVoltTArr)
    
    #Trim the neutron detector time interval
    timeInterval=0.001
    #Get the start time based on the readback
    startTimeInd=np.where(rdbkArr>3.5)[0][0]
    startTime=rdbkTArr[startTimeInd]+0.00
    stopTimeInd=np.where(rdbkArr>3.5)[0][-1]
    stopTime=rdbkTArr[stopTimeInd]
    
    #Remove the DC offset from the neutron detector signal
    nFlux=neutron_dc_offset(nFlux,nFluxTArr,startTime)
    
    #Trim the signal arrays
    #Neutron singal
    nFluxTrimmed,nFluxTArrTrimmed=time_trim(nFlux,nFluxTArr,startTime,stopTime)
    #Beam Current
    beamCurrTrimmed,beamCurrTArrTrimmed=time_trim(beamCurr,beamCurrTArr,startTime,stopTime)
    #Beam Voltage
    beamVoltTrimmed,beamVoltTArrTrimmed=time_trim(beamVolt,beamVoltTArr,startTime,stopTime)
        
    #Get the total detected neutron count
    neutronFlux=np.trapz(nFluxTrimmed,nFluxTArrTrimmed)
    
    #Get the average beam current
    beamCurrAvg=np.average(beamCurrTrimmed)
    print(beamCurrAvg)
    
    #Get the average beam voltage
    beamVoltAvg=np.average(beamVoltTrimmed)
    print(beamVoltAvg)
    
    #Normalize the neutron count
    nFluxNorm=neutronFlux/beamCurrAvg
    
    #Append data to the arrays
    crossArr.append(nFluxNorm)
    pEnergyArr.append(beamVoltAvg)
    
#Convert to numpy arrays
crossArr=np.array(crossArr)
pEnergyArr=np.array(pEnergyArr)

#%% All 4 data streams

#Good shot number
shotnum=1210329020

#Get the neutron flux data
nodepath_neutron='\mraw_misc::wham_1n_det'
nFlux,nFluxTArr=variable_data(shotnum,nodepath_neutron)
#Flip the neutron flux signal because of the PMT bias
nFlux=-nFlux

#Smooth the data
nFluxSmooth=savgol_filter(nFlux,19,1)

#Get the beam current
nodepath_beamCurr='\mraw_nbi::top:nbi_i_beam'
beamCurr,beamCurrTArr=variable_data(shotnum,nodepath_beamCurr)
#Convert to s
beamCurrTArr/=1000

#Get the beam voltage
nodepath_beamCurr='\mraw_nbi::top:nbi_u_beam'
beamVolt,beamVoltTArr=variable_data(shotnum,nodepath_beamCurr)
#Convert to s
beamVoltTArr/=1000

#Beam readback
nodepath_nbi_start='\mraw_nbi::nbi_rdbk'
rdbkArr,rdbkTArr=variable_data(shotnum,nodepath_nbi_start)

#Plasma current
nodepath_plasmaCurr='\ip'
plasmaArr,plasmaTArr=variable_data(shotnum,nodepath_plasmaCurr)

#Correct the NBI timing arrays
beamCurrTArr,beamVoltTArr=align_time(rdbkArr,rdbkTArr,beamCurrTArr,beamVoltTArr)

#Neutron Detector Signal
plt.figure(figsize=(15,8))
plt.title('Neutron Detector Data')
plt.plot(nFluxTArr,nFlux,label='Raw Signal')
plt.plot(nFluxTArr,nFluxSmooth,label='Smoothed Signal')
plt.plot([0.013,0.013],[-1,10],linewidth=4,color='black',label='Beam')
plt.plot([0.0361,0.0361],[-1,10],linewidth=4,color='black')
plt.xlabel('Time [s]')
plt.xlim(0,0.06)
plt.ylabel('Signal [V]')
plt.ylim(0,5)
plt.legend()
plt.grid()
plt.savefig('C:/Users/kunal/OneDrive - UW-Madison/WHAM/Plots/neutron_detector_signal_'+str(shotnum)+'.png'\
            ,dpi=600)
plt.show()

#Beam Voltage
plt.figure(figsize=(15,8))
plt.title('Beam Voltage')
plt.plot(beamVoltTArr,beamVolt)
plt.xlabel('Time [s]')
plt.xlim(0,0.06)
plt.ylabel('Signal [kV]')
plt.grid()
plt.savefig('C:/Users/kunal/OneDrive - UW-Madison/WHAM/Plots/beam_voltage_'+str(shotnum)+'.png'\
            ,dpi=600)
plt.show()

#Beam Current
plt.figure(figsize=(15,8))
plt.title('Beam Current')
plt.plot(beamCurrTArr,beamCurr)
plt.xlabel('Time [s]')
plt.xlim(0,0.06)
plt.ylabel('Signal [A]')
plt.grid()
plt.savefig('C:/Users/kunal/OneDrive - UW-Madison/WHAM/Plots/beam_current_'+str(shotnum)+'.png'\
            ,dpi=600)
plt.show()

#Plasma Current
plt.figure(figsize=(15,8))
plt.title('Plasma Current')
plt.plot(plasmaTArr,plasmaArr)
plt.xlabel('Time [s]')
plt.xlim(0,0.06)
plt.ylabel('Signal [A]')
plt.grid()
plt.savefig('C:/Users/kunal/OneDrive - UW-Madison/WHAM/Plots/plasma_current_'+str(shotnum)+'.png'\
            ,dpi=600)
plt.show()

#%% Plotting

plt.figure(figsize=(15,8))
plt.scatter(fIonArr,nFluxArr)
plt.title('Total Neutron Flux')
plt.ylabel('Total Neutron Flux [Vs]')
plt.ylim(0,0.0006)
plt.xlabel('Fast Ion Charge [C]')
plt.xlim(0,0.035)
plt.grid()
plt.savefig('C:/Users/kunal/OneDrive - UW-Madison/WHAM/Plots/beam_current_scan.png',dpi=600)
plt.show()

plt.figure(figsize=(15,8))
plt.scatter(pEnergyArr,crossArr)
plt.title('Fusion Cross Section')
plt.ylabel('Normalized Neutron Flux [Vs/A]')
plt.ylim(0,3.5e-5)
# plt.yscale('log')
plt.xlabel('Particle Energy [keV]')
plt.xlim(0,25)
plt.grid()
plt.savefig('C:/Users/kunal/OneDrive - UW-Madison/WHAM/Plots/beam_voltage_scan.png',dpi=600)
plt.show()

# plt.figure()
# plt.plot(beamCurrTArr,beamCurr)
# plt.title('Beam Current')
# plt.show()

# plt.figure()
# plt.plot(beamVoltTArr,beamVolt)
# plt.title('Beam Voltage')
# plt.show()

# plt.figure()
# plt.plot(rdbkTArr,rdbkArr)
# plt.title('Beam Readback')
# plt.show()

# plt.figure()
# plt.plot(nFluxTArr,nFlux)
# plt.title('Neutron Detector')
# plt.show()