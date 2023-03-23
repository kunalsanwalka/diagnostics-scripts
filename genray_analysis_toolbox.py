# -*- coding: utf-8 -*-
"""
Created on Thu Apr 29 12:04:37 2021

@author: kunal

This program contains a list of functions used to extract useful information
from the netCDF4 files output by Genray.

Most functions are written to be standalone, requiring only the location of the
file. However, each function has its own docstring and one should refer to that
to understand function behaviour.

Packages needed to run functions in this file-
1. numpy as np
2. netCDF4 as nc
3. matplotlib.pyplot as plt
4. scipy.constants as constants
5. math
"""

import math
import warnings
import numpy as np
import netCDF4 as nc
import matplotlib.pyplot as plt
from scipy import constants
from scipy.interpolate import RegularGridInterpolator
plt.rcParams.update({'font.size': 22})
warnings.filterwarnings("ignore", category=DeprecationWarning)

# =============================================================================
# Destination of the plots
# =============================================================================
plotDest='C:/Users/kunal/OneDrive - UW-Madison/WHAM/Plots/'

# =============================================================================
# Genray Data Directory
# =============================================================================
genrayDest='C:/Users/kunal/OneDrive - UW-Madison/WHAM/Data/Genray/'

def electron_density_XZ(filename,makeplot=False,saveplot=False):
    """
    This function returns the electron density profile along with the
    associated coordinate arrays.
    
    To get the 1D from Xmesh and and Zmesh, use the following code-
    x1d=Xmesh[0]
    z1d=Zmesh[:,0]    

    Parameters
    ----------
    filename : string
        Location of the Genray output file.
    makeplot : boolean
        Make a plot of the data.
    saveplot : boolean
        Save the plot.

    Returns
    -------
    Xmesh : np.array
        X position.
    Zmesh : np.array
        Z position.
    densprofxz : np.array
        Electron density.
    """
    
    #Open the file
    ds=nc.Dataset(filename)
    
    # =========================================================================
    # Get the raw data
    # =========================================================================
    
    #x-array for equilibrium B
    eqdsk_x=np.array(ds['eqdsk_x'][:])
    
    #z-array for equilibrium B
    eqdsk_z=np.array(ds['eqdsk_z'][:])
    
    #Density of electrons on the XZ plane
    densprofxz=np.array(ds['densprofxz'][:])
    
    #Create a mesh from eqdsk_x and eqdsk_y
    Xmesh,Zmesh=np.meshgrid(eqdsk_x,eqdsk_z)
    
    #Convert to particles/m^3
    densprofxz*=10**-6
    
    # =========================================================================
    # Plot the data
    # =========================================================================
    
    if makeplot==True:
        
        #Generate the savename of the plot
        #Get the name of the .nc file
        ncName=filename.split('/')[-1]
        #Remove the .nc part
        ncName=ncName[0:-3]
        savename=ncName+'_electron_density_XZ.svg'
        
        fig=plt.figure(figsize=(21,8))
        ax=fig.add_subplot(111)
        pltobj=ax.contourf(Zmesh,Xmesh,densprofxz,levels=30)
        ax.contour(pltobj,colors='black')
        ax.set_xlabel('Z [m]')
        ax.set_ylabel('R [m]')
        ax.set_title('Electron Density')
        ax.grid(True)
        cbar=fig.colorbar(pltobj)
        cbar.set_label(r'Number of Particles [m$^{-3}$]')
        if saveplot==True:
            plt.savefig(plotDest+savename,bbox_inches='tight')
        plt.show()
            
    return Xmesh,Zmesh,densprofxz

def magnetic_field_RZ(filename,makeplot=False,saveplot=False):
    """
    This function returns the magnetic field profile along with the associated
    coordinate arrays.
    
    NOTE- The field values are only for r>0. If you are making a XZ-plot. Refer
    to the plots in the function to see how to handle that case.
    
    To get the 1D from Rmesh and and Zmesh, use the following code-
    r1d=Rmesh[0]
    z1d=Zmesh[:,0]

    Parameters
    ----------
    filename : string
        Location of the Genray output file.
    makeplot : boolean
        Make a plot of the data.
    saveplot : boolean
        Save the plot.

    Returns
    -------
    Rmesh : np.array
        R position.
    Zmesh : np.array
        Z position.
    Br : np.array
        Radial component of B.
    Bz : np.array
        Axial component of B.
    Bmag : np.array
        Magnetic field strength.
    """
    
    #Open the file
    ds=nc.Dataset(filename)
    
    # =========================================================================
    # Get the raw data
    # =========================================================================
    
    #r-array for equilibrium B
    eqdsk_r=np.array(ds['eqdsk_r'][:])
    
    #z-array for equilibrium B
    eqdsk_z=np.array(ds['eqdsk_z'][:])
    
    #Poloidal flux / 2*pi
    eqdsk_psi=np.array(ds['eqdsk_psi'][:])
    
    # =========================================================================
    # Process the data
    # =========================================================================

    #Create the R,Z mesh (useful for plotting)
    Rmesh,Zmesh=np.meshgrid(eqdsk_r,eqdsk_z)
    
    #dr and dz to take the gradient of the flux
    dr=eqdsk_r[1]-eqdsk_r[0]
    dz=eqdsk_z[1]-eqdsk_z[0]
    
    #Br and Bz
    Bz=np.gradient(eqdsk_psi,dr,axis=1)/(eqdsk_r)
    Br=np.gradient(eqdsk_psi,dz,axis=0)/(eqdsk_r)
    
    #Correct for the / by 0 error when R=0
    Bz[:,0]=-Bz[:,2]+2*Bz[:,1]
    Br[:,0]=0
    
    #|B| (since we are axisymmetric, B_phi=0)
    Bmag=np.sqrt(Bz**2+Br**2)
    
    # =========================================================================
    # Plot the data
    # =========================================================================
    
    if makeplot==True:
        
        #Generate the savename of the plot
        #Get the name of the .nc file
        ncName=filename.split('/')[-1]
        #Remove the .nc part
        ncName=ncName[0:-3]
        savenameStrength=ncName+'_magnetic_field_strength.svg'
        savenameBr=ncName+'_magnetic_field_r.svg'
        savenameBz=ncName+'_magnetic_field_z.svg'
        
        # =====================================================================
        # Magnetic Field Strength
        # =====================================================================
        
        fig=plt.figure(figsize=(21,8))
        ax=fig.add_subplot(111)
        pltobj=ax.contourf(Zmesh,Rmesh,Bmag,levels=30)
        ax.contour(pltobj,colors='black')
        pltobj=ax.contourf(Zmesh,-Rmesh,Bmag,levels=30)
        ax.contour(pltobj,colors='black')
        ax.set_xlabel('Z [m]')
        ax.set_ylabel('R [m]')
        ax.set_title('Magnetic Field Strength')
        ax.grid(True)
        cbar=fig.colorbar(pltobj)
        cbar.set_label('|B| [T]')
        if saveplot==True:
            plt.savefig(plotDest+savenameStrength,bbox_inches='tight')
        plt.show()
        
        # =====================================================================
        # B_r
        # =====================================================================
        
        fig=plt.figure(figsize=(21,8))
        ax=fig.add_subplot(111)
        pltobj=ax.contourf(Zmesh,Rmesh,Br,levels=30)
        ax.contour(pltobj,colors='black')
        pltobj=ax.contourf(Zmesh,-Rmesh,Br,levels=30)
        ax.contour(pltobj,colors='black')
        ax.set_xlabel('Z [m]')
        ax.set_ylabel('R [m]')
        ax.set_title(r'Magnetic Field ($B_r$)')
        ax.grid(True)
        cbar=fig.colorbar(pltobj)
        cbar.set_label(r'$B_r$ [T]')
        if saveplot==True:
            plt.savefig(plotDest+savenameBr,bbox_inches='tight')
        plt.show()
        
        # =====================================================================
        # B_z
        # =====================================================================
        
        fig=plt.figure(figsize=(21,8))
        ax=fig.add_subplot(111)
        pltobj=ax.contourf(Zmesh,Rmesh,Bz,levels=30)
        ax.contour(pltobj,colors='black')
        pltobj=ax.contourf(Zmesh,-Rmesh,Bz,levels=30)
        ax.contour(pltobj,colors='black')
        ax.set_xlabel('Z [m]')
        ax.set_ylabel('R [m]')
        ax.set_title(r'Magnetic Field ($B_z$)')
        ax.grid(True)
        cbar=fig.colorbar(pltobj)
        cbar.set_label(r'$B_z$ [T]')
        if saveplot==True:
            plt.savefig(plotDest+savenameBz,bbox_inches='tight')
        plt.show()
        
    return Rmesh,Zmesh,Br,Bz,Bmag

def flux_surfaces(filename,makeplot=False,saveplot=False):
    """
    This function returns the flux surfaces from Pleiades as stored in the
    Genray .nc
    
    To get the 1D from Rmesh and and Zmesh, use the following code-
    r1d=Rmesh[0]
    z1d=Zmesh[:,0]

    Parameters
    ----------
    filename : string
        Location of the Genray output file.
    makeplot : boolean
        Make a plot of the data.
    saveplot : boolean
        Save the plot.

    Returns
    -------
    Rmesh : np.array
        R position.
    Zmesh : np.array
        Z position.
    eqdsk_psi : np.array
        Flux surfaces.
    """
    
    #Open the file
    ds=nc.Dataset(filename)
    
    # =========================================================================
    # Get the raw data
    # =========================================================================
    
    #r-array for equilibrium B
    eqdsk_r=np.array(ds['eqdsk_r'][:])
    
    #z-array for equilibrium B
    eqdsk_z=np.array(ds['eqdsk_z'][:])
    
    #Poloidal flux / 2*pi
    eqdsk_psi=np.array(ds['eqdsk_psi'][:])
    
    # =========================================================================
    # Process the data
    # =========================================================================

    #Create the R,Z mesh (useful for plotting)
    Rmesh,Zmesh=np.meshgrid(eqdsk_r,eqdsk_z)
    
    # =========================================================================
    # Plot the data
    # =========================================================================
    
    if makeplot==True:
        
        #Generate the savename of the plot
        #Get the name of the .nc file
        ncName=filename.split('/')[-1]
        #Remove the .nc part
        ncName=ncName[0:-3]
        savename=ncName+'_flux_surfaces.png'
        
        fig=plt.figure(figsize=(21,8))
        ax=fig.add_subplot(111)
        
        #Plot the poloidal flux
        #Plot limits
        psimag=ds.variables['psimag'].getValue()  #getValue() for scalar
        psimag=psimag.item()
        psilim=ds.variables['psilim'].getValue()  #getValue() for scalar
        psilim=psilim.item()
        PSImin=psimag
        PSImax=psilim+0.30*(psilim-psimag) #plot a few more surfaces than the LCFS
        #Plot levels
        levels=np.arange(PSImin,PSImax,(PSImax-PSImin)/50)
        
        pltobj=ax.contour(Zmesh,Rmesh,eqdsk_psi,levels=levels)
        pltobj=ax.contour(Zmesh,-Rmesh,eqdsk_psi,levels=levels)
        cbar=fig.colorbar(pltobj)
        cbar.set_label(r'$\Psi_B$ [$T \cdot m^2$]')
        
        ax.set_xlabel('Z [m]')
        ax.set_ylabel('R [m]')
        
        ax.grid(True)
        
        if saveplot==True:
            plt.savefig(plotDest+savename,bbox_inches='tight')
            
        plt.show()
    
    return Rmesh,Zmesh,eqdsk_psi

def field_along_flux_surface(filename,fluxValue,makeplot=False,saveplot=False):
    """
    This function returns the position and value of the magnetic field strength
    along a given flux surface.

    Parameters
    ----------
    filename : string
        Location of the Genray output file.
    fluxValue : float
        Value of the flux through a given surface.
    makeplot : boolean
        Make a plot of the data.
    saveplot : boolean
        Save the plot.

    Returns
    -------
    zPosArr : np.array
        Z position.
    rPosArr : np.array
        R position.
    Bflux : np.array
        Strength of the field along the flux surface.
    """
    
    # =========================================================================
    # Get the raw data
    # =========================================================================
    
    #Flux surface data
    Rmesh,Zmesh,eqdsk_psi=flux_surfaces(filename)
    
    #Magnetic field data
    Rmesh,Zmesh,Br,Bz,Bmag=magnetic_field_RZ(filename)
    
    # =========================================================================
    # Process the data
    # =========================================================================
    
    #Interpolation function for magnetic field data
    field_interpolator=RegularGridInterpolator((Zmesh[:,0],Rmesh[0]),Bmag)
    
    #Points for a given contour
    #Make a contour object
    contourObj=plt.contour(Zmesh,Rmesh,eqdsk_psi,levels=[fluxValue])
    #Get the vertex data
    vertexData=contourObj.collections[-1].get_paths()[0].vertices
    #z positions
    zPosArr=vertexData[::-1,0]
    #r positions
    rPosArr=vertexData[::-1,1]
    plt.close()
    
    #Get the field strength at the given points
    #Initialize array
    Bflux=[]
    for i in range(len(zPosArr)):
        Bflux.append(field_interpolator([zPosArr[i],rPosArr[i]])[0])
    #Convert to numpy
    Bflux=np.array(Bflux)
    
    #Calculate the number of maxima of |B| along the flux surface
    epsilon=0.3
    #1st derivative
    dBdz=np.gradient(Bflux,zPosArr)
    #2nd derivative
    d2Bdz2=np.gradient(dBdz,zPosArr)
    #Number of maxima
    maxima=0
    for val in d2Bdz2:
        if val<epsilon and val>-epsilon:
            maxima+=1
    #Print the result
    print('==================================================================')
    print('The number of maxima along this flux surface is- '+str(maxima))
    print('==================================================================')
    
    # =========================================================================
    # Plot the data
    # =========================================================================
    
    if makeplot==True:
        
        #Generate the savename of the plot
        #Get the name of the .nc file
        ncName=filename.split('/')[-1]
        #Remove the .nc part
        ncName=ncName[0:-3]
        savename=ncName+'_field_along_flux_surface_Psi_'+str(fluxValue)+'.png'
        
        fig=plt.figure(figsize=(21,12))
        fig.suptitle(r'$\Psi_B$='+str(fluxValue)+' T/m$^2$')
        
        ax=fig.add_subplot(211)
        
        ax.plot(zPosArr,Bflux)
        
        ax.set_ylabel('|B| [T]')
        ax.set_title('Field Strength |B|')
        ax.grid(True)
        plt.setp(ax.get_xticklabels(),visible=False)
        
        ax=fig.add_subplot(212)
        
        ax.plot(zPosArr,dBdz)
        
        ax.set_xlabel('Z [m]')
        ax.set_ylabel(r'dB/dz [T/m]')
        ax.set_title('dB/dz')
        ax.grid(True)
        
        if saveplot==True:
            plt.savefig(plotDest+savename,bbox_inches='tight')
            
        plt.show()
    
    return zPosArr,rPosArr,Bflux

def refractive_indices(filename):
    """
    This function returns the refractive index of the ray as it propagates 
    through the plasma.
    
    The 1st index specifies the ray number.
    The 2nd index is the value of the refractive index.    

    Parameters
    ----------
    filename : string
        Location of the Genray output file.

    Returns
    -------
    wn_x : np.array
        n_x refractive index
    wn_z : np.array
        n_z refractive index
    wn_phi : np.array
        n_phi refractive index
    wnper : np.array
        n_per refractive index
    wnpar : np.array
        n_par refractive index
    wntot : np.array
        n_tot refractive index
    """
    
    #Open the file
    ds=nc.Dataset(filename)
    
    # =========================================================================
    # Get the raw data
    # =========================================================================
    
    #n_x refractive index
    wn_x=np.array(ds['wn_x'])
    
    #n_z refractive index
    wn_z=np.array(ds['wn_z'])
    
    #n_phi refractive index
    wn_phi=np.array(ds['wn_phi'])
    
    #n_per refractive index
    wnper=np.array(ds['wnper'])
    
    #n_par refractive index
    wnpar=np.array(ds['wnpar'])
    
    # =========================================================================
    # Calculate n_tot
    # =========================================================================
    
    wntot=np.sqrt(wnper**2+wnpar**2)
    
    return wn_x,wn_z,wn_phi,wnper,wnpar,wntot

def ray_paths(filename):
    """
    This function returns the coordiates of the ray as it propagates through
    the plasma.
    
    The 1st index specifies the ray number.
    The 2nd index is the value of the coordinate.

    Parameters
    ----------
    filename : string
        Location of the Genray output file.

    Returns
    -------
    wr : np.array
        R position.
    wx : np.array
        X position.
    wy : np.array
        Y position.
    wz : np.array
        Z position.
    ws : np.array
        Distance.
    """
    
    #Open the file
    ds=nc.Dataset(filename)
    
    # =========================================================================
    # Get the raw data
    # =========================================================================
    
    #r positions of each ray
    wr=np.array(ds['wr'])
    
    #x positions of each ray
    wx=np.array(ds['wx'])
    
    #y positions of each ray
    wy=np.array(ds['wy'])
    
    #z positions of each ray
    wz=np.array(ds['wz'])
    
    #Distance along each ray
    ws=np.array(ds['ws'])
    
    # =========================================================================
    # Convert to m
    # =========================================================================
    
    wr/=100
    wx/=100
    wy/=100
    wz/=100
    ws/=100
    
    return wr,wx,wy,wz,ws

def ray_power_deposition(filename):
    """
    This function returns the power deposited by each ray into the plasma as a 
    function of the total path length it has covered. The 1st index in each
    output variable selects for a given ray.
    
    The inital power in each ray is evenly divided based on the total initial
    power. Therefore 2 runs with the same initial parameters except the number
    of rays will have different initial powers in each ray.
    
    NOTE- When the ray terminates in genray, all subsequent values in ws and delpwr
    are stored as 0.

    Parameters
    ----------
    filename : string
        Location of the Genray output file.

    Returns
    -------
    ws : np.array
        Path length.
    delpwr : np.array
        Power deposited along the path.
    """
    
    #Open the file
    ds=nc.Dataset(filename)
    
    # =========================================================================
    # Get the raw data
    # =========================================================================
    
    ws=np.array(ds['ws'][:])
    
    delpwr=np.array(ds['delpwr'][:])
    
    #Convert to m
    ws/=100
    
    #Convert to W
    delpwr*=10**-7
    
    return ws,delpwr

def start_point(filename):
    """
    This function returns the start positions of the ray launch in Genray.

    Parameters
    ----------
    filename : string
        Location of the Genray output file.

    Returns
    -------
    startPos : np.array
        Start positons (r,x,y,z).
    """
    
    #Open the file
    ds=nc.Dataset(filename)
    
    # =========================================================================
    # Get the raw data
    # =========================================================================
    
    #r positions of each ray
    wr=np.array(ds['wr'][:])
    
    #x positions of each ray
    wx=np.array(ds['wx'][:])
    
    #y positions of each ray
    wy=np.array(ds['wy'][:])
    
    #z positions of each ray
    wz=np.array(ds['wz'][:])
    
    # =========================================================================
    # Coordinates of the start point (r,x,y,z)
    # =========================================================================
    
    #Coordinates of all start points
    rStart=wr[:,0]
    xStart=wx[:,0]
    yStart=wy[:,0]
    zStart=wz[:,0]
    
    #Convert to m
    rStart/=100
    xStart/=100
    yStart/=100
    zStart/=100
    
    startPos=np.array([rStart,xStart,yStart,zStart])
    
    return startPos

def resonant_field_strengths(filename,species1,species2):
    """
    This function calculates the magentic field strength for the 1st 5
    harmonics of the D and T cyclotron frequency given the antenna frequency.

    Parameters
    ----------
    filename : string
        Location of the Genray output file.
    species1 : string
        Tag for the ion species.
    species2 : string
        Tag for the ion species.

    Returns
    -------
    Bs1Arr : np.array
        Species 1 |B| array.
    Bs2Arr : np.array
        Species 2 |B| array.
    """
    
    #Open the file
    ds=nc.Dataset(filename)
    
    # =========================================================================
    # Get the raw data
    # =========================================================================
    
    #Antenna frequency (Hz)
    freq=int(ds['freqcy'][:])
    
    # =========================================================================
    # Magnetic fields for resonances
    # =========================================================================
    
    #Fundamental frequency field for species1
    if species1=='D':
        B0s1=2*np.pi*2*constants.m_p*freq/constants.e
    elif species1=='T':
        B0s1=2*np.pi*3*constants.m_p*freq/constants.e
    elif species1=='He3':
        B0s1=2*np.pi*3*constants.m_p*freq/(2*constants.e)
    
    #Fundamental frequency field for species2
    if species2=='D':
        B0s2=2*np.pi*2*constants.m_p*freq/constants.e
    elif species2=='T':
        B0s2=2*np.pi*3*constants.m_p*freq/constants.e
    elif species2=='He3':
        B0s2=2*np.pi*3*constants.m_p*freq/(2*constants.e)
    
    #Array with the 1st 9 harmonics
    Bs1Arr=[]
    Bs2Arr=[]
    for i in reversed(range(1,8)): #range() does not include the last number
        Bs1Arr.append(B0s1/i)
        Bs2Arr.append(B0s2/i)
    
    #Convert to numpy
    Bs1Arr=np.array(Bs1Arr)
    Bs2Arr=np.array(Bs2Arr)
    
    return Bs1Arr,Bs2Arr

def plot_power_in_ray_channel(filename,saveplot=False):
    """
    This function plots the power in each ray as a function of the distance it
    has propagated through the plasma.

    Parameters
    ----------
    filename : string
        Location of the Genray output file.
    saveplot : boolean
        Save the plot.
        
    Returns
    -------
    None.
    """
    
    # =========================================================================
    # Get the raw data
    # =========================================================================
    
    ws,delpwr=ray_power_deposition(filename)
    
    # =========================================================================
    # Create the plot
    # =========================================================================
    
    #Generate the savename of the plot
    #Get the name of the .nc file
    ncName=filename.split('/')[-1]
    #Remove the .nc part
    ncName=ncName[0:-3]
    savename=ncName+'_power_in_ray_channel.png'
    
    fig=plt.figure(figsize=(21,8))
    ax=fig.add_subplot(111)
    
    #Plot each ray
    #Number of rays
    numRays=np.shape(ws)[0]
    #Evenly spaced array for chaning colors
    colorArr=np.linspace(0,1,numRays)
    #Plot each ray
    for i in range(numRays):
        
        #Take the data for each ray
        wsCurr=ws[i,:]
        delpwrCurr=delpwr[i,:]
        
        #Indetify indices that have delpwr!=0
        delpwr0=np.where(delpwrCurr!=0)[0]
        
        #If no value meet that criteria, plot the whole array
        if len(delpwr0)==0:
            ax.plot(wsCurr,delpwrCurr,linewidth=2,color=(colorArr[i],0,0))
            
        #Else plot only when the condition delpwr!=0 condition is True
        else:
            #Get subarrays where that condition is not true
            wsSub=wsCurr[delpwr0]
            delpwrSub=delpwrCurr[delpwr0]
            
            ax.plot(wsSub,delpwrSub,linewidth=2,color=(colorArr[i],0,0))
        
    #Axes labels and plot title
    ax.set_xlabel('Z [m]')
    ax.set_ylabel('Ray Power [W]')
    ax.set_ylim(-0.05*np.max(delpwr),1.05*np.max(delpwr))
    ax.set_title('Power in each ray channel')
        
    #Add a grid
    ax.grid(True)
    
    #Save the plot
    if saveplot==True:
        plt.savefig(plotDest+savename,bbox_inches='tight')
    
    plt.show()
    
    return

def plot_ray_path_with_B_flux_surfaces_XZ(filename,species1='D',species2='T',saveplot=False):
    """
    This function plots the ray wavepaths and magnetic flux surfaces on the XZ
    Plane. It also highlights field strengths that are at the resonances for
    D and T given the antenna frequency.

    Parameters
    ----------
    filename : string
        Location of the Genray output file.
    species1 : string
        Tag for the ion species.
    species2 : string
        Tag for the ion species.
    saveplot : boolean
        Save the plot.
        
    Returns
    -------
    None.
    """
    
    # =========================================================================
    # Get the raw data
    # =========================================================================
    
    #Antenna frequency
    freq=int(ds['freqcy'][:])
    
    #Flux surfaces
    eqdsk_psi=np.array(ds['eqdsk_psi'])
    
    #Get the magnetic field data
    Rmesh,Zmesh,Br,Bz,Bmag=magnetic_field_RZ(filename)
    
    #Get the resonance field values
    BDArr,BTArr=resonant_field_strengths(filename,species1,species2)
    
    #Get the ray path data
    wr,wx,wy,wz,ws=ray_paths(filename)
    
    #Get the start position data
    startPos=start_point(filename)
    
    # =========================================================================
    # Analysis
    # =========================================================================
    
    if species1=='D':
        label1='Deuterium'
    elif species1=='T':
        label1='Tritium'
    elif species1=='He3':
        label1='Helium-3'
        
    if species2=='D':
        label2='Deuterium'
    elif species2=='T':
        label2='Tritium'
    elif species2=='He3':
        label2='Helium-3'
    
    # =========================================================================
    # Create the plot
    # =========================================================================
    
    #Generate the savename of the plot
    #Get the name of the .nc file
    ncName=filename.split('/')[-1]
    #Remove the .nc part
    ncName=ncName[0:-3]
    savename=ncName+'_ray_path_with_B_field_strength_XZ.png'
    
    fig=plt.figure(figsize=(21,8))
    ax=fig.add_subplot(111)
    
    #Plot the poloidal flux
    #Plot limits
    psimag=ds.variables['psimag'].getValue()  #getValue() for scalar
    psimag=psimag.item()
    psilim=ds.variables['psilim'].getValue()  #getValue() for scalar
    psilim=psilim.item()
    PSImin=psimag
    PSImax=psilim+0.30*(psilim-psimag) #plot a few more surfaces than the LCFS
    #Plot levels
    levels=np.arange(PSImin,PSImax,(PSImax-PSImin)/50)
    
    print(PSImin)
    print(PSImax)
    
    pltobj=ax.contour(Zmesh,Rmesh,eqdsk_psi,levels=levels)
    pltobj=ax.contour(Zmesh,-Rmesh,eqdsk_psi,levels=levels)
    
    #Flux surface colorbar
    # cbar=fig.colorbar(pltobj)
    # cbar.set_label(r'$\Psi_B$ [$T \cdot m^2$]')
    
    #Plot the D resonances
    dCont=ax.contour(Zmesh,Rmesh,Bmag,levels=BDArr,colors='black',linewidths=3)
    ax.contour(Zmesh,-Rmesh,Bmag,levels=BDArr,colors='black',linewidths=3)
    #Plot the T resonances
    tCont=ax.contour(Zmesh,Rmesh,Bmag,levels=BTArr,colors='dodgerblue',linestyles='dashed',linewidths=3,zorder=10)
    ax.contour(Zmesh,-Rmesh,Bmag,levels=BTArr,colors='dodgerblue',linestyles='dashed',linewidths=3,zorder=10)
    
    #Plot limits
    ax.set_ylim(-0.2,0.2)
    
    #Add labels for each contour
    DArrDict=dict()
    TArrDict=dict()
    for i in range(len(BTArr)):
        DArrDict[BDArr[i]]=str(int(BDArr[-1]/BDArr[i]))
        TArrDict[BTArr[i]]=str(int(BTArr[-1]/BTArr[i]))
    #Label locations (Assumes we only have 4 labels)
    # #For WHAM VNS (D)
    # DLabelLocs=np.array([(-1.1,0.58),(-0.82,0.67),(-0.48,0.2),(-0.58,0.74)])
    # TLabelLocs=np.array([(-1.25,0.4),(-0.94,0.5),(-0.75,0.56),(-0.57,0.6)])
    #For WHAM VNS (T)
    # DLabelLocs=np.array([(-0.95,0.64),(-0.58,0.76)])
    # TLabelLocs=np.array([(-1.1,0.37),(-0.74,0.54),(-0.46,0.61),(0,0.67)])
    # #For WHAM
    # DLabelLocs=np.array([(-0.75,0.17),(-0.63,0.23),(-0.50,0.23),(-0.28,0.27)])
    # TLabelLocs=np.array([(-0.79,0.09),(-0.68,0.12),(-0.59,0.13),(-0.5,0.15)])
    #For WHAM D-He3
    DLabelLocs=np.array([(-0.72,0.103),(-0.56,0.115),(-0.44,0.136),(-0.01,0.147)])
    TLabelLocs=np.array([(-0.69,0.15),(-0.50,0.14),(-0.11,0.15),(-0.5,0.15)])
        
    ax.clabel(dCont,fmt=DArrDict,manual=DLabelLocs)
    ax.clabel(tCont,fmt=TArrDict,manual=TLabelLocs)
    
    #Plot the ray paths
    #Number of rays
    numRays=np.shape(wx)[0]
    #Evenly spaced array for chaning colors
    colorArr=np.linspace(0,1,numRays)
    #Plot each ray
    for i in range(numRays):
        ax.plot(wz[i,:],wx[i,:],linewidth=2,color=(colorArr[i],0,0))
        
    #Plot the start point of the rays
    startP=ax.scatter(startPos[3,:],startPos[0,:],s=100,color='black',zorder=10)
    
    #Legend
    d1,_=dCont.legend_elements()
    t1,_=tCont.legend_elements()
    ax.legend([d1[0],t1[0],startP],[label1,label2,'Start Point'],loc=[1.01,0.775])
    
    #Axes labels and plot title
    ax.set_xlabel('Z [m]')
    ax.set_ylabel('X [m]')
    #Convert frequency to MHz
    freqMhz=np.round(freq/1e6,2)
    #Strength of the field at the center of the device
    B0=np.round(Bmag[math.floor(len(Bmag)/2),0],2)
    ax.set_title(r'Ray Paths vs Magnetic Flux Surfaces [f='+str(freqMhz)+'MHz] [$B_0$='+str(B0)+'T]')
    
    #Add a grid
    ax.grid(True)
    
    #Save the plot
    if saveplot==True:
        plt.savefig(plotDest+savename,bbox_inches='tight')
    
    plt.show()
    
    return

def plot_ray_path_with_e_dens_XZ(filename,saveplot=False):
    """
    This function plots the ray wavepaths and electron density contours on the 
    XZ Plane.

    Parameters
    ----------
    filename : string
        Location of the Genray output file.
    saveplot : boolean
        Save the plot.
        
    Returns
    -------
    None.
    """
    
    # =========================================================================
    # Get the raw data
    # =========================================================================
    
    #Antenna frequency
    freq=int(ds['freqcy'][:])
    
    #Get the electron density
    Xmesh,Zmesh,eDens=electron_density_XZ(filename)
    
    #Get the ray path data
    wr,wx,wy,wz,ws=ray_paths(filename)
    
    #Get the start position data
    startPos=start_point(filename)
    
    # =========================================================================
    # Create the plot
    # =========================================================================
    
    fig=plt.figure(figsize=(21,8))
    ax=fig.add_subplot(111)
    
    #Plot the electron density
    pltobj=ax.contour(Zmesh,Xmesh,eDens,levels=50)
    cbar=fig.colorbar(pltobj)
    cbar.set_label(r'$n_e$ [$m^{-3}$]')
    
    #Plot the ray paths
    #Number of rays
    numRays=np.shape(wx)[0]
    #Evenly spaced array for changing colors
    colorArr=np.linspace(0,1,numRays)
    #Plot each ray
    for i in range(numRays):
        ax.plot(wz[i,:],wx[i,:],linewidth=2,color=(colorArr[i],0,0))
        
    #Plot the start point of the rays
    startP=ax.scatter(startPos[3],startPos[0],s=100,color='black',zorder=10)
    
    #Axes labels and plot title
    ax.set_xlabel('Z [m]')
    ax.set_ylabel('X [m]')
    #Convert frequency to MHz
    freqMhz=np.round(freq/1e6,2)
    ax.set_title(r'Ray Paths vs Electron Density [f='+str(freqMhz)+'MHz]')
    
    #Add a grid
    ax.grid(True)
    
    #Save the plot
    if saveplot==True:
        
        #Generate the savename of the plot
        #Get the name of the .nc file
        ncName=filename.split('/')[-1]
        #Remove the .nc part
        ncName=ncName[0:-3]
        savename=ncName+'_ray_path_with_e_dens_XZ.png'
        
        plt.savefig(plotDest+savename,bbox_inches='tight')
    
    plt.show()
    
    return

def plot_ray_path_XY(filename,saveplot=False):
    """
    This function plots the ray wavepaths on the XY Plane.

    Parameters
    ----------
    filename : string
        Location of the Genray output file.
    saveplot : boolean
        Save the plot.
        
    Returns
    -------
    None.
    """
    
    # =========================================================================
    # Get the raw data
    # =========================================================================
    
    #Antenna frequency
    freq=int(ds['freqcy'][:])
    
    #Get the ray path data
    wr,wx,wy,wz,ws=ray_paths(filename)
    
    #Get the start position data
    startPos=start_point(filename)
    
    # =========================================================================
    # Create the plot
    # =========================================================================
    
    fig=plt.figure(figsize=(10,10))
    ax=fig.add_subplot(111)
    
    #Plot the ray paths
    #Number of rays
    numRays=np.shape(wx)[0]
    #Evenly spaced array for changing colors
    colorArr=np.linspace(0,1,numRays)
    #Plot each ray
    for i in range(numRays):
        ax.plot(wx[i,:],wy[i,:],linewidth=2,color=(colorArr[i],0,0))
        
    #Plot the start point of the rays
    startP=ax.scatter(startPos[1],startPos[2],s=100,color='black',zorder=10)
    
    #Axes labels and plot title
    ax.set_xlabel('X [m]')
    ax.set_ylabel('Y [m]')
    #Convert frequency to MHz
    freqMhz=np.round(freq/1e6,2)
    ax.set_title(r'Ray Paths [f='+str(freqMhz)+'MHz]')
    
    #Add a grid
    ax.grid(True)
    
    #Save the plot
    if saveplot==True:
        
        #Generate the savename of the plot
        #Get the name of the .nc file
        ncName=filename.split('/')[-1]
        #Remove the .nc part
        ncName=ncName[0:-3]
        savename=ncName+'_ray_path_XY.png'
        
        plt.savefig(plotDest+savename,bbox_inches='tight')
    
    plt.show()
    
    return

def plot_refractive_index_with_prop_distance(filename,saveplot=False):
    """
    This function plots the various refractive indices as a function of the
    distance each ray has travelled through the plasma.

    Parameters
    ----------
    filename : string
        Location of the Genray output file.
    saveplot : boolean
        Save the plot.
        
    Returns
    -------
    None.
    """
    
    # =========================================================================
    # Get the raw data
    # =========================================================================
    
    #Ray paths
    wr,wx,wy,wz,ws=ray_paths(filename)
    
    #Refractive indices
    wn_x,wn_z,wn_phi,wnper,wnpar,wntot=refractive_indices(filename)

    #Mask to remove (0,0) points due to Genray errors
    wsMask=[]
    
    for i in range(np.shape(ws)[0]):
        currMask=np.where(ws[i]!=0)[0]
        wsMask.append(currMask)
    
    # =========================================================================
    # Create the plot
    # =========================================================================
    
    #Generate the savename of the plot
    #Get the name of the .nc file
    ncName=filename.split('/')[-1]
    #Remove the .nc part
    ncName=ncName[0:-3]
    savename=ncName+'_ref_ind_with_prop_dist.png'
    
    #Initialize the plot
    fig,axs=plt.subplots(2,3,figsize=(35,15))
    
    #Title for all plots
    fig.suptitle('Refractive Indices vs. Propagation Distance')
    
    #Axis limit
    xMax=np.max(ws)
    
    #Number of rays
    numRays=np.shape(ws)[0]
    #Evenly spaced array for changing colors
    colorArr=np.linspace(0,1,numRays)
    
    # =========================================================================
    # Plot n_x vs propagation distance
    # =========================================================================
    
    #Pick the correct subplot
    ax=axs[0,0]
    
    #Plot each ray
    for i in range(numRays):
        ax.plot(ws[i,wsMask[i]],wn_x[i,wsMask[i]],linewidth=2,color=(colorArr[i],0,0))
        
    #Axes limits and labels
    ax.set_xlim(0,xMax)
    ax.set_xlabel('Propagation Distance [m]')
    ax.set_ylabel(r'$n_x$ [arb. u.]')
    
    ax.grid(True)
    
    # =========================================================================
    # Plot n_z vs propagation distance
    # =========================================================================
    
    #Pick the correct subplot
    ax=axs[1,0]
    
    #Plot each ray
    for i in range(numRays):
        ax.plot(ws[i,wsMask[i]],wn_z[i,wsMask[i]],linewidth=2,color=(colorArr[i],0,0))
        
    #Axes limits and labels
    ax.set_xlim(0,xMax)
    ax.set_xlabel('Propagation Distance [m]')
    ax.set_ylabel(r'$n_z$ [arb. u.]')
    
    ax.grid(True)
    
    # =========================================================================
    # Plot n_|| vs propagation distance
    # =========================================================================
    
    #Pick the correct subplot
    ax=axs[0,1]
    
    #Plot each ray
    for i in range(numRays):
        ax.plot(ws[i,wsMask[i]],wnpar[i,wsMask[i]],linewidth=2,color=(colorArr[i],0,0))
        
    #Axes limits and labels
    ax.set_xlim(0,xMax)
    ax.set_xlabel('Propagation Distance [m]')
    ax.set_ylabel(r'$n_{||}$ [arb. u.]')
    
    ax.grid(True)
    
    # =========================================================================
    # Plot n_perp vs propagation distance
    # =========================================================================
    
    #Pick the correct subplot
    ax=axs[1,1]
    
    #Plot each ray
    for i in range(numRays):
        ax.plot(ws[i,wsMask[i]],wnper[i,wsMask[i]],linewidth=2,color=(colorArr[i],0,0))
        
    #Axes limits and labels
    ax.set_xlim(0,xMax)
    ax.set_xlabel('Propagation Distance [m]')
    ax.set_ylabel(r'$n_{\perp}$ [arb. u.]')
    
    ax.grid(True)
    
    # =========================================================================
    # Plot n_phi vs propagation distance
    # =========================================================================
    
    #Pick the correct subplot
    ax=axs[0,2]
    
    #Plot each ray
    for i in range(numRays):
        ax.plot(ws[i,wsMask[i]],wn_phi[i,wsMask[i]],linewidth=2,color=(colorArr[i],0,0))
        
    #Axes limits and labels
    ax.set_xlim(0,xMax)
    ax.set_xlabel('Propagation Distance [m]')
    ax.set_ylabel(r'$n_{\phi}$ [arb. u.]')
    
    ax.grid(True)
    
    # =========================================================================
    # Plot n_tot vs propagation distance
    # =========================================================================
    
    #Pick the correct subplot
    ax=axs[1,2]
    
    #Plot each ray
    for i in range(numRays):
        ax.plot(ws[i,wsMask[i]],wntot[i,wsMask[i]],linewidth=2,color=(colorArr[i],0,0))
        
    #Axes limits and labels
    ax.set_xlim(0,xMax)
    ax.set_xlabel('Propagation Distance [m]')
    ax.set_ylabel(r'$n$ [arb. u.]')
    
    ax.grid(True)
    
    #Save the plot
    if saveplot==True:
        plt.savefig(plotDest+savename,bbox_inches='tight')
    
    plt.show()
    
    return

def plot_field_along_flux_surfaces(filename,fluxValues,saveplot=False):
    """
    This function plots the magnetic field strength and its 1st derivative wrt
    z for a given set of flux values.

    Parameters
    ----------
    filename : string
        Location of the Genray output file.
    fluxValues : np.array
        Array with the desired flux values.
    ssaveplot : boolean
        Save the plot.

    Returns
    -------
    None.
    """
    
    # =========================================================================
    # Initialize the plot
    # =========================================================================
    
    fig=plt.figure(figsize=(21,12))
    
    # =========================================================================
    # Plot the field strength
    # =========================================================================
    
    ax=fig.add_subplot(211)
    
    #Number of flux values
    numRays=len(fluxValues)
    #Evenly spaced array for changing colors
    colorArr=np.linspace(0,1,numRays)
    #Go over each flux value
    for i in range(len(fluxValues)):
        
        #Get the data
        zPosArr,rPosArr,Bflux=field_along_flux_surface(filename,fluxValues[i])
        
        #Plot the data
        ax.plot(zPosArr,Bflux,color=(colorArr[i],0,0),label=r'$\Psi_B$='+str(round(fluxValues[i],3)))
    
    ax.set_ylabel('|B| [T]')
    ax.set_title('Field strength |B|')
    
    ax.set_xlim(min(zPosArr),max(zPosArr))
    
    ax.grid(True)
    
    plt.setp(ax.get_xticklabels(),visible=False)
    
    ax.legend(bbox_to_anchor=(1.01,1.03))
    
    # =========================================================================
    # Plot the 1st derivative wrt z
    # =========================================================================
    
    ax=fig.add_subplot(212)
    
    #Go over each flux value
    for i in range(len(fluxValues)):
        
        #Get the data
        zPosArr,rPosArr,Bflux=field_along_flux_surface(filename,fluxValues[i])
        
        #Take the 1st derivative
        dBdz=np.gradient(Bflux,zPosArr)
        
        #Plot the data
        ax.plot(zPosArr,dBdz,color=(colorArr[i],0,0),label=r'$\Psi_B$='+str(round(fluxValues[i],3)))
        
    ax.set_xlabel('Z [m]')
    ax.set_ylabel('dB/dz [T/m]')
    ax.set_title('dB/dz')
    
    ax.set_xlim(min(zPosArr),max(zPosArr))
    
    ax.grid(True)
    
    if saveplot==True:
        
        #Generate the savename of the plot
        #Get the name of the .nc file
        ncName=filename.split('/')[-1]
        #Remove the .nc part
        ncName=ncName[0:-3]
        savename=ncName+'_field_along_flux_surfaces.png'
        
        fig.savefig(plotDest+savename,bbox_inches='tight')
    
    fig.show()
    
    return

#%% Analysis

#Location of Genray output
filename=genrayDest+'220601_onlyD_morePoints.nc'

#Get the netCDF4 data
ds=nc.Dataset(filename)

#Get the electron density
# Xmesh,Zmesh,eDens=electron_density_XZ(filename,makeplot=False)

#Get the magnetic field data
# Rmesh,Zmesh,Br,Bz,Bmag=magnetic_field_RZ(filename,makeplot=True)

#Get the flux surfaces
# Rmesh,Zmesh,eqdsk_psi=flux_surfaces(filename,makeplot=True)

#Get the field along a given flux surface
# zPosArr,rPosArr,Bflux=field_along_flux_surface(filename,0.3,makeplot=True)

#Get the resonance field values
# BDArr,BTArr=resonant_field_strengths(filename)

#Get the refective indices
# wn_x,wn_z,wn_phi,wnper,wnpar,wntot=refractive_indices(filename)

#Get the ray path data
# wr,wx,wy,wz,ws=ray_paths(filename)

#Get the ray power deposition
# ws,delpwr=ray_power_deposition(filename)

#Get the start position data
# startPos=start_point(filename)

#Plot the power in each ray channel
# plot_power_in_ray_channel(filename,saveplot=True)

#Plot the ray path with Psi_B on the XZ Plane
plot_ray_path_with_B_flux_surfaces_XZ(filename,species1='D',species2='T',saveplot=True)

#Plot the ray path with n_e on the XZ Plane
# plot_ray_path_with_e_dens_XZ(filename,saveplot=True)

#Plot the ray path on the XY Plane
# plot_ray_path_XY(filename,saveplot=True)

#Plot the refractive indices vs propagation distance
# plot_refractive_index_with_prop_distance(filename,saveplot=True)

#Plot the field along multiple flux surfaces
# plot_field_along_flux_surfaces(filename,np.linspace(0.001,0.5,10),saveplot=True)

#%% Plot ray path vs poloidal flux

# #Generate the savename of the plot
# #Get the name of the .nc file
# ncName=filename.split('/')[-1]
# #Remove the .nc part
# ncName=ncName[0:-3]
# savename=ncName+'_ray_path_with_poloidal_flux_XZ.png'

# fig=plt.figure(figsize=(21,8))
# ax=fig.add_subplot(111)

# #Plot the poloidal flux
# #Plot limits
# psimag=ds.variables['psimag'].getValue()  #getValue() for scalar
# psimag=psimag.item()
# psilim=ds.variables['psilim'].getValue()  #getValue() for scalar
# psilim=psilim.item()
# PSImin=psimag
# PSImax=psilim+0.30*(psilim-psimag) #plot a few more surfaces than the LCFS
# #Plot levels
# levels=np.arange(PSImin,PSImax,(PSImax-PSImin)/(50))

# pltobj=ax.contour(Zmesh,Rmesh,eqdsk_psi,levels=levels)
# pltobj=ax.contour(Zmesh,-Rmesh,eqdsk_psi,levels=levels)
# cbar=fig.colorbar(pltobj)
# cbar.set_label(r'$\Psi_B$ [$T \cdot m^2$]')

# #Plot the D resonances
# dCont=ax.contour(Zmesh,Rmesh,Bmag,levels=BDArr,colors='black',linewidths=3)
# ax.contour(Zmesh,-Rmesh,Bmag,levels=BDArr,colors='black',linewidths=3)
# #Plot the T resonances
# tCont=ax.contour(Zmesh,Rmesh,Bmag,levels=BTArr,colors='black',linestyles='dashed',linewidths=3)
# ax.contour(Zmesh,-Rmesh,Bmag,levels=BTArr,colors='black',linestyles='dashed',linewidths=3)

# #Add a legend for the resonance contours
# d1,_=dCont.legend_elements()
# t1,_=tCont.legend_elements()
# ax.legend([d1[0],t1[0]],['Deuterium','Tritium'],loc=[1.2,0.85])

# #Plot the ray paths
# #Number of rays
# numRays=np.shape(wx)[0]
# #Evenly spaced array for chaning colors
# colorArr=np.linspace(0,1,numRays)
# #Plot each ray
# for i in range(numRays):
#     ax.plot(wz[i,:],wx[i,:],linewidth=2,color=(colorArr[i],0,0))
    
# #Plot the start point of the rays
# ax.scatter(startPos[3],startPos[0],s=100,color='black',zorder=10)

# ax.set_xlabel('Z [m]')
# ax.set_ylabel('X [m]')
# ax.set_title('Ray Paths vs Magnetic Flux Surfaces')

# ax.grid(True)

# plt.savefig(plotDest+savename,bbox_inches='tight')

# plt.show()

#%% Plot field on multiple flux surfaces

# fluxValues=np.linspace(0.001,0.3,10)

# # =========================================================================
# # Initialize the plot
# # =========================================================================

# fig=plt.figure(figsize=(21,8))
# ax=fig.add_subplot(111)

# # =========================================================================
# # Plot each flux value
# # =========================================================================

# #Number of flux values
# numRays=len(fluxValues)
# #Evenly spaced array for changing colors
# colorArr=np.linspace(0,1,numRays)
# #Go over each flux value
# for i in range(len(fluxValues)):
    
#     #Get the data
#     zPosArr,rPosArr,Bflux=field_along_flux_surface(filename,fluxValues[i])
    
#     #Plot the data
#     ax.plot(zPosArr,Bflux,color=(colorArr[i],0,0),label=str(fluxValues[i]))
    
# ax.set_xlabel('Z [m]')
# ax.set_ylabel('|B| [T]')
    
# ax.grid(True)
# ax.legend(loc='best')

# fig.show()