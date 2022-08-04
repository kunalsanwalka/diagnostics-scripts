# -*- coding: utf-8 -*-
"""
Created on Wed Jun 15 10:29:32 2022

@author: kunal

This program analyzes the date from a charge sensitive detector designed by
O. Anderson, M. Yu and K. Sanwalka
"""

import abel
import time
import numpy as np
import pandas as pd
import scipy as sc
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 22})

# =============================================================================
# Plot Directory
# =============================================================================
plotDest='C:/Users/kunal/OneDrive - UW-Madison/WHAM/Plots/'

# =============================================================================
# Raw Data Directory
# =============================================================================
dataDest='C:/Users/kunal/OneDrive - UW-Madison/WHAM/Data/Proton Detector Testing/'

# =============================================================================
# Processed Data Directory
# =============================================================================
processedDataDest='C:/Users/kunal/OneDrive - UW-Madison/WHAM/Processed Data/'

def open_red_pitaya_csv(filename):
    """
    This function returns the time and data arrays for a trace taken from a
    red pitaya and stored in the csv format.

    Parameters
    ----------
    filename : string
        Location of the csv file.

    Returns
    -------
    timeArr : np.array
        Time array [s].
    dataArr : np.array
        Value array [V].
    """
    
    #Open the file
    file=open(filename,'rb')
    
    #Skip the 1st line since it has array labels
    next(file)
    
    #Convert data to numpy
    data=np.loadtxt(file,delimiter=',')
    
    #Separate time data and value data into 2 arrays
    timeArr=data[:,0]
    dataArr=data[:,1]
    
    #Convert time to s
    timeArr/=1e3
    
    return timeArr,dataArr

def open_red_pitaya_binary(filename):
    """
    This function returns the time and data arrays for a trace taken from a
    red pitaya and stored in a binary file format.

    Parameters
    ----------
    filename : string
        Location of the binary file.

    Returns
    -------
    timeArr : np.array
        Time array [s].
    dataArr : np.array
        Value array [arb u.].
    """
    
    #Sampling rate
    sampRate=125e6
    
    #Timestep per data point
    timeStep=1/sampRate
    
    #Integer array
    dataArr=[]
    
    #Open the binary file
    with open(filename,mode='rb') as file:
        
        data=file.read()
        
        dataArr=np.fromstring(data,dtype=np.int16)
            
    #Convert to numpy array
    dataArr=np.array(dataArr)
    
    #Construct the time array
    timeArr=np.linspace(0,timeStep*len(dataArr),len(dataArr))
    
    return timeArr,dataArr

def open_data(filename):
    """
    This function allows for abstraction of the specific function used to open
    the data file. This allows any data handling function to be used and the
    change only needs to be made here.

    Parameters
    ----------
    filename : string
        Location of the data file.

    Returns
    -------
    timeArr : np.array
        Time array [s].
    dataArr : np.array
        Value array [V].
    """
    
    #Get data from the red pitaya when stored as a csv file
    # timeArr,dataArr=open_red_pitaya_csv(filename)
    
    #Get data from the red pitaya when stored as a binary file
    timeArr,dataArr=open_red_pitaya_binary(filename)
    
    return timeArr,dataArr

def rolling_avg(timeArr,dataArr,window=5):
    """
    This function takes a rolling average of the data with the specified window
    size.

    Parameters
    ----------
    timeArr : np.array
        Time array [s].
    dataArr : np.array
        Raw data array.
    window : int, optional
        Window size for smoothing. The default is 5.

    Returns
    -------
    timeAvg : np.array
        Time array [s].
    dataAvg : np.array
        Averaged data array.
    """
    
    #Construct a pandas data structure
    dataStruct=pd.DataFrame(dataArr)
    
    #Take a rolling average
    dataStruct['rollingAvg']=dataStruct.rolling(window).mean()
    
    #Get the rolling average array
    dataAvg=dataStruct['rollingAvg']
    
    #Remove the nan values
    dataAvg=dataAvg[window:]
    
    #Convert to numpy array
    dataAvg=np.array(dataAvg)
    
    #Adjust the time array to match
    timeAvg=timeArr[window:]
    
    return timeAvg,dataAvg

def peak_timestamps(filename):
    """
    This function calculates the time stamps when a proton hits the detector.

    Parameters
    ----------
    filename : string
        Location of the data file.

    Returns
    -------
    peakTimes : np.array
        Time stamps of each proton detection.
    """
    
    #Open the file
    timeArr,dataArr=open_data(filename)
    
    #Convolve the data with a gaussian
    convData=sc.signal.convolve(dataArr,sc.signal.windows.gaussian(50,     #Width of gaussian (in number of data points)
                                                                   10))    #Standard deviation
    
    #Differentiate the data to find the peaks
    diffData=np.diff(convData)
    
    #Find the peaks
    peaks,prop=sc.signal.find_peaks(-diffData,prominence=400)
    
    #Remove peaks with index greater than length of timeArr (processing artifact)
    peaks=peaks[peaks<(len(timeArr)-1)]
    
    #Time stamps of the peaks
    peakTimes=timeArr[peaks]
    
    #Remove the 1st detection (processing artifact)
    peakTimes=peakTimes[1:]
    
    return peakTimes

def reaction_rate(filename,timeInterval=1e-5):
    """
    This function calculates the reaction rate as seen by the detector.
    
    It measures the number of hits in the given time interval and scales that
    to the number that would be seen in 1s.
    
    This window of the size timeInterval is moved forward by 1us and the
    process is repeated.

    Parameters
    ----------
    filename : string
        Location of the data file.
    timeInterval : float, optional
        Time interval to measure the reaction rate. The default is 1e-5.

    Returns
    -------
    rxRate : np.array
        Fusion reaction rate as seen by the detector.
    rxTimeArr : np.array
        Associated time array.
    """
    
    #Get the raw data
    timeArr,dataArr=open_data(filename)
    
    #Get the time stamps of each detection
    peakTimes=peak_timestamps(filename)
    
    #Array to store the reaction rate
    rxRate=[]
    
    #Time of final detection
    finalTime=peakTimes[-1]
    
    #Time array
    rxTimeArr=np.arange(0,finalTime-timeInterval,1e-6)
    
    #Go over each time interval
    for i in range(len(rxTimeArr)):
        
        #Hits in given time interval
        hitsArr=peakTimes[np.logical_and(peakTimes>rxTimeArr[i],
                                         peakTimes<(rxTimeArr[i]+timeInterval))]
        
        #Number of hits
        numHits=len(hitsArr)
        
        #Scale to convert rate to s^[-1]
        rate=numHits/timeInterval
        
        #Append to the array
        rxRate.append(rate)
        
    #Convert to numpy
    rxRate=np.array(rxRate)
    
    return rxRate,rxTimeArr

def perform_analysis(filename,
                     dataDirectory='C:/Users/kunal/OneDrive - UW-Madison/WHAM/Data/Proton Detector Testing/',
                     processedDataDirectory='C:/Users/kunal/OneDrive - UW-Madison/WHAM/Processed Data/',
                     plotDirectory='C:/Users/kunal/OneDrive - UW-Madison/WHAM/Plots/',
                     makeplot=False,
                     saveplot=False):
    """
    This function reads the raw data from the detector and calculates the
    timestamps for each reaction as well as the reaction rate.

    Parameters
    ----------
    filename : string
        Name of the binary file.
    dataDirectory : string, optional
        Location of the binary file.
    processedDataDirectory : TYPE, optional
        Location to store the processed data.
    plotDirectory : string, optional
        Location to store the plots.
    makeplot : boolean
        Make a plot of the data.
    saveplot : boolean
        Save the plot.

    Returns
    -------
    rxTimeArr : np.array
        Timestamps of each detection.
    rxRate : np.array
        Reaction rate.
    rxTimeArr : np.array
        Time array for the reaction rate.
    """
    
    #Add directory path to the filename
    filenameDir=dataDirectory+filename
    
    # =========================================================================
    # Perform the analysis
    # =========================================================================
    
    #Get the raw data
    timeArr,dataArr=open_data(filenameDir)
    
    #Get the timestamps of the peaks
    peakTimes=peak_timestamps(filenameDir)
    
    #Get the reaction rate
    rxRate,rxTimeArr=reaction_rate(filenameDir,timeInterval=5e-3)
    
    # =========================================================================
    # Save the data
    # =========================================================================
    
    #Get the filename with the .bin
    filenameNoBin=filename[:-4]
    
    #Generate the savenames
    savenamePeakTimes=filenameNoBin+'_peakTimes.npy'
    savenameRxRate=filenameNoBin+'_rxRate.npy'
    savenameRxRateTimeArr=filenameNoBin+'_rxRateTimeArr.npy'
    
    #Save the data
    np.save(processedDataDirectory+savenamePeakTimes,peakTimes)
    np.save(processedDataDirectory+savenameRxRate,rxRate)
    np.save(processedDataDirectory+savenameRxRateTimeArr,rxTimeArr)
    
    # =========================================================================
    # Plot the data
    # =========================================================================
    
    if makeplot==True:
        
        #Plot the raw data
        fig=plt.figure(figsize=(15,8))
        ax=fig.add_subplot(111)
        
        ax.plot(timeArr,dataArr)
        
        ax.set_xlabel('Time [s]')
        ax.set_ylabel('Signal [arb. u.]')
        ax.set_title('Raw Signal')
        
        ax.grid()
        plt.show()
        
        if saveplot==True:
            
            #Generate the savename
            savename=filename[:-4]+'_raw_data.pdf'
            
            #Save the plot
            plt.savefig(plotDirectory+savename,bbox_inches='tight')
        
        #Plot the reaction rate
        fig=plt.figure(figsize=(15,8))
        ax=fig.add_subplot(111)
        
        ax.plot(rxTimeArr,rxRate)
        
        ax.set_xlim(min(rxTimeArr),max(rxTimeArr))
        ax.set_ylim(0,max(rxRate))
        
        ax.set_xlabel('Time [s]')
        ax.set_ylabel(r'Reaction rate [s$^{-1}$]')
        ax.set_title('Reaction Rate')
        
        ax.grid()
        plt.show()
        
        if saveplot==True:
            
            #Generate the savename
            savename=filename[:-4]+'_reaction_rate.pdf'
            
            #Save the plot
            plt.savefig(plotDirectory+savename,bbox_inches='tight')
    
    return rxTimeArr,rxRate,rxTimeArr

def abel_inversion(integratedSignalArr,closestApproachArr,makeplot=False,saveplot=False):
    """
    This function performs the inverse abel transform for a given integrated
    radial profile.

    Parameters
    ----------
    integratedSignalArr : np.array
        Line of slight integrated data.
    closestApproachArr : np.array
        The minimum r value traced through by the protons.
    makeplot : boolean
        Make a plot of the data.
    saveplot : boolean
        Save the plot.

    Returns
    -------
    radialProfile : TYPE
        Radial profile of the reaction rate.
    """
    
    #Perform the inverse abel transform
    radialProfile=abel.direct.direct_transform(integratedSignalArr,
                                               r=closestApproachArr,
                                               direction='inverse')
    
    # =========================================================================
    # Plot the data
    # =========================================================================
    
    if makeplot==True:
        
        #Create the plot
        fig=plt.figure(figsize=(12,8))
        ax=fig.add_subplot(111)
        
        #Plot the integrated data
        sig1=ax.plot(closestApproachArr,integratedSignalArr,color='red',label='Integrated Signal')
        ax.set_ylabel(r'Integrated reaction rate [m/s]',color='red')
        
        #Plot the radial profile
        ax1=ax.twinx()
        sig2=ax1.plot(closestApproachArr,radialProfile,color='blue',label='Reconstructed Profile')
        ax1.set_ylabel(r'Reaction rate [s$^{-1}$]',color='blue')
        
        ax.set_xlabel(r'Radial Position [m]')
        ax.set_title(r'Radial reaction rate profiles')
        
        sigs=sig1+sig2
        names=[l.get_label() for l in sigs]
        ax.legend(sigs,names,loc='lower left')
        plt.show()
        
        if saveplot==True:
            
            #Generate the savename
            savename='reconstructed_radial_profile.pdf'
            
            #Save the plot
            plt.savefig(plotDest+savename,bbox_inches='tight')
    
    return radialProfile

#%% Analysis

#This function takes around 30s to process 10^8 data points.

#Name of the file
filename='2e8_samples_1.bin'

#Process the data
rxTimeArr,rxRate,rxTimeArr=perform_analysis(filename,makeplot=True,saveplot=True)

#%% Processing by hand

filename=dataDest+'1e6_samples_2(1).bin'

#Open the file
timeArr,dataArr=open_data(filename)

#Convolve the data with a gaussian
convData=sc.signal.convolve(dataArr,sc.signal.windows.gaussian(50,    #Width of gaussian (in number of data points)
                                                               10))    #Standard deviation

#Differentiate the data to find the peaks
diffData=np.diff(convData)

#Find the peaks
peaks,prop=sc.signal.find_peaks(-diffData,prominence=400)

#Remove peaks with index greater than length of timeArr (processing artifact)
peaks=peaks[peaks<(len(timeArr)-1)]

#Remove the 1st detection (processing artifact)
peaks=peaks[1:]

#Time stamps of the peaks
peakTimes=timeArr[peaks]

rxRate,rxTimeArr=reaction_rate(filename,timeInterval=1e-2)

#%% Plotting

#Plot the processed data
fig=plt.figure(figsize=(15,8))
ax=fig.add_subplot(111)

ax.plot(timeArr[1:],diffData[1:-48])
ax.scatter(timeArr[peaks],diffData[peaks],color='red')

ax.set_xlabel('Time [s]')
ax.set_ylabel('Signal [arb. u.]')
ax.set_title('Processed Signal')

ax.grid()
plt.show()

#Plot the raw data
fig=plt.figure(figsize=(15,8))
ax=fig.add_subplot(111)

ax.plot(timeArr,dataArr)

ax.set_xlabel('Time [s]')
ax.set_ylabel('Signal [arb. u.]')
ax.set_title('Raw Signal')

ax.grid()
plt.show()

#Plot the reaction rate
fig=plt.figure(figsize=(15,8))
ax=fig.add_subplot(111)

ax.plot(rxTimeArr,rxRate)
ax.set_xlabel('Time [s]')
ax.set_ylabel(r'Reaction rate [s$^{-1}$]')
ax.set_title('Reaction Rate')

ax.grid()
plt.show()

#%% Abel inversion testing

closestApproachArr=np.arange(0,1.5,0.1)

#Construct an integrated array for a constant density function
integratedSignalArr=np.array([2,1.98997,1.95959,1.90788,1.83303,1.73205,1.6,1.42829,1.2,0.87178,0,0,0,0,0])

#Perform the inverse abel transform
radialProfile=abel.direct.direct_transform(integratedSignalArr,
                                           r=closestApproachArr,
                                           direction='inverse')
    
# =========================================================================
# Plot the data
# =========================================================================

makeplot=True
saveplot=True

if makeplot==True:
    
    #Create the plot
    fig=plt.figure(figsize=(12,8))
    ax=fig.add_subplot(111)
    
    #Plot the integrated data
    sig1=ax.plot(closestApproachArr,integratedSignalArr,color='red',label='Integrated Signal')
    ax.set_ylabel(r'Integrated reaction rate [m/s]',color='red')
    
    #Plot the radial profile
    ax1=ax.twinx()
    sig2=ax1.plot(closestApproachArr,radialProfile,color='blue',label='Reconstructed Profile')
    ax1.set_ylabel(r'Reaction rate [s$^{-1}$]',color='blue')
    
    #Plot the ideal radial profile
    sig3=ax1.plot([0,1,1,1.4],[1,1,0,0],color='green',label='Ideal Profile')
    
    ax.set_xlabel(r'Radial Position [m]')
    ax.set_title(r'Radial reaction rate profiles')
    
    sigs=sig1+sig2+sig3
    names=[l.get_label() for l in sigs]
    ax.legend(sigs,names,loc='lower left')
    plt.show()
    
    if saveplot==True:
        
        #Generate the savename
        savename='reconstructed_radial_profile.pdf'
        
        #Save the plot
        plt.savefig(plotDest+savename,bbox_inches='tight')