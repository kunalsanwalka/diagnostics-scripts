# -*- coding: utf-8 -*-
"""
Created on Wed Jul  7 14:10:36 2021

@author: Michael

This .py file calculates fast ion orbit given a magnetic field and starting 
position. 

In the future, this will be updated to get a starting position from
a neutral beam injection. 


"""

import numpy as np
from scipy.integrate import solve_ivp
from scipy.interpolate import RectBivariateSpline
from scipy import interpolate
#this isn't imported onto dave yet. it is a better integrator than what is 
#standard in scipy.
#from extensisq import Pr8
import matplotlib.pyplot as plt
import math

#this was copied and pasted from Jon Pizzo and was used in past versions for debugging.
#This is is cd ../wham/pleiades_GUI/plieades_gui_main.py if you want the commented version.
def plot_B_contours(b,r,z):
        try:
            log_lvl_max = math.ceil(np.log2(np.max(b)))
            log_lvl_min = math.floor(np.log2(np.min(b)))
        except:
            print("Need at least one coil to plot fields")
            return
        n_lvls = log_lvl_max-log_lvl_min+1
        levels = np.logspace(log_lvl_min,log_lvl_max,n_lvls,base=2)
        label_dict = dict()
        for i in range(len(levels)):
            if levels[i] >= 1:
                label_dict[levels[i]] = "{} T".format(levels[i])
            elif levels[i] >= 0.1:
                label_dict[levels[i]] = "{} kG".format(levels[i]*10)
            else:
                label_dict[levels[i]] = "{} G".format(levels[i]*10000)

        fig1 = plt.figure(1)
        ax1 = fig1.add_subplot(111)
        ax1.set_xlabel("Z (m)")
        ax1.set_ylabel("R (m)")

        cs = plt.contour(z,r,b,levels=levels,colors='black') #Plot B Contours
        plt.contour(z,-r,b,levels=levels,colors='black')
        plt.clabel(cs,inline_spacing=10,fmt=label_dict,colors='black')  #Contour Labels
        temp_B = np.copy(b)
        Bmin = np.min(temp_B[np.nonzero(temp_B)])
        Bmax = np.max(temp_B)
        plt.show()

def interpolate_field(inputDict):
    global bz_global
    global br_global
    global z_grid
    global r_grid

    #for debugging purposes 
    global brField_used
    global bzField_used
    global zPos_used
    global rPos_used
    global yActual
    brField_used= []
    zPos_used = []
    bzField_used = []
    yActual = []
    rPos_used = []
    #grabbing components we need from the input dictionary
    z = inputDict["z"]
    r= inputDict["r"]
    bz_in = inputDict["b_z"].reshape(z.shape)
    br_in=inputDict["b_r"].reshape(r.shape)
    npoints=inputDict["npoints"]
    z=z[:,0]
    r=r[0,:]

#these lines were used for debugging. Commented in case we need to use it later.
#    fig1,ax7 = plt.subplots()
#    ax7.plot(z,bz_in)
#    ax7.set_title("Z field from pleiades, not interpolated")
#    fig1,ax8 =plt.subplots()
#    ax8.plot(z,br_in)
#    ax8.set_title("R field from pleiades, not interpolated")
    
    #here we redifine the position grids as 1D arrays, get mins and maxes for us to make 
    #bigger ones later.
    rmin=0
    rmax=np.amax(r)
    zmin=np.amin(z)
    zmax=np.amax(z)


    #initializing the interp2d function. When we call this with a larger number of points, 
    #it interpolates to fit.
    
    br_global = interpolate.interp2d(r,z,br_in,kind='linear')
    bz_global = interpolate.interp2d(r,z,bz_in,kind='linear')
    
#more debugging lines..
#    #initializing bigger r and z axes for us to interpolate on
#    r_axis = np.linspace(rmin,rmax,npoints)
#    z_axis = np.linspace(zmin,zmax,npoints)
#    r_grid,z_grid = np.meshgrid(r_axis,z_axis)
#    #interpolating to make sure our interpolation is correct
#    bz_interped=bz_global(r_axis,z_axis)
#    br_interped=br_global(r_axis,z_axis)
#    b = np.hypot(bz_interped,br_interped)
#    plot_B_contours(b,r_grid,z_grid)
#    #for debugging...
#    fig1,ax1 = plt.subplots()
#    fig1,ax2 = plt.subplots()
#    ax1.plot(z_grid,bz_interped)    
#    ax2.plot(z_grid,br_interped)
#    ax1.set_title("Interpolated z Field (not used by calc)")
#    ax2.set_title("Interpolated r Field (not used by calc) ")

#unused due to errors at r=0
def lorentz_cylindrical(t,y,inputs):

    qm=inputs[0]
    bfield=get_bfield_estimate(inputs[1],inputs[2],y)
    br=bfield[0]
    bphi=bfield[1]
    bz=bfield[2]
    ddt = [
            y[3], #r comp velocity
            y[4], #phi comp velocity
            y[5], #z comp velocity
            qm*(y[0]*y[4]*bz-y[5]*bphi)+(y[4]**2)*y[0], #r comp acceleration
            qm*(y[5]*br-y[3]*bz)-2*y[4]*y[3]/y[0], #phi comp acceleration
            qm*(y[3]*bphi-y[0]*y[4]*br), #z comp acceleration
            ] 
    return ddt


#there are going to be three of these so we don't need to run an if statement every
#time the integrator loops.
def lorentz_cartesian_uniform(t,y,inputs):
#lorentz with a uniform bfield in z direction
    bx=0
    by=0
    bz=0.3
    ddt=[ 
        y[3], #x comp velocity
        y[4], #y comp velocity
        y[5], #z comp velocity
        qm*(y[4]*bz-y[5]*by),#x comp acceleration 
        qm*(y[5]*bx-y[3]*bz),#y comp acceleration
        qm*(y[3]*by-y[4]*bx) #z comp acceleration 
        ]
    return ddt

def lorentz_cartesian_toroidal(t,y,inputs):
#lorentz with a toriodal bfield
    bx=0
    by=0
    bz=2/np.hypot(y[0],y[1]) 
    ddt=[ 
        y[3], #x comp velocity
        y[4], #y comp velocity
        y[5], #z comp velocity
        qm*(y[4]*bz-y[5]*by),#x comp acceleration 
        qm*(y[5]*bx-y[3]*bz),#y comp acceleration
        qm*(y[3]*by-y[4]*bx) #z comp acceleration 
        ]
    return ddt

def lorentz_cartesian_customfield(t,y,inputs):
#lorentz with a bfield that has been input.
    
    #if the interpolation works the way i think it does, this will give the indices of the field.
    r = np.hypot(y[0],y[1])
    z= y[2]
    
    #for converting to cartesian
    phi=np.arctan2(y[1],y[0])
    
    #getting from the field interpolator
    br = br_global(r,z)
    bx=br*np.cos(phi)
    by=br*np.sin(phi)
    bz=bz_global(r,z)
    
#Debugging block......
#    bzField_used.append(bz)
#    brField_used.append(br)
#    zPos_used.append(z)
#    yActual.append(y[2])
#    rPos_used.append(r)


    #return derivative of path to the integrator
    ddt=[ 
        y[3], #x comp velocity
        y[4], #y comp velocity
        y[5], #z comp velocity
        qm*(y[4]*bz-y[5]*by),#x comp acceleration 
        qm*(y[5]*bx-y[3]*bz),#y comp acceleration
        qm*(y[3]*by-y[4]*bx) #z comp acceleration
        ]
    return ddt


#these are estimates of mirror field- will be obsolete soon. 
def initialize_mirror_estimate(zax):
    bzg=17*np.exp(-abs(zax-0.96)/2/0.33**2)
    bzgn=np.flip(bzg)
    bz=bzg+bzgn
    
    return bz

def get_bfield_estimate(zax,y,bz):
    index=int(closest(zax,y[2]))
    #recalculate for phi=0,r=f(b)
    rComp=-np.hypot(y[0],y[1])*np.gradient(bz)/2
    rComp=rComp[index]
    phi=np.arctan(y[1]/y[0])

    return[np.cos(phi)*rComp,np.sin(phi)*rComp,bz[index]]

def orbitEnergy(path,qm):
    '''
    
    Parameters
    ----------
    path : sol (from integrator)
        Path output from integrator.
    qm : float
        Charge/mass ratio, used in energy calculations.

    Returns
    -------
    tuple
        Energy at each point.

    '''
    sol=path.y
    return (sol[3]**2+sol[4]**2+sol[5]**2)/(qm*2)

def run_orbit(inputDict):

    #making this global so we don't have to deal with it all that much..
    global qm
    
    qm = inputDict["qm"]
    y0=inputDict["y0"]
    fieldType=inputDict["fieldType"]
    tArray=inputDict["time_array"]
    inputs = inputDict

    #This commented line is run with a better integrator that is not yet installed on dave. 
    #return solve_ivp(lambda t,y: lorentz_cartesian(t, y, inputs),[tArray[0],tArray[-1]],y0,t_eval=tArray,method=Pr8)
    if fieldType==0:
        interpolate_field(inputDict)
        sol= solve_ivp(lambda t,y: lorentz_cartesian_customfield(t, y, inputs),[tArray[0],tArray[-1]],y0,t_eval=tArray)
#these are also lines used for debugging. Commmented out in case we need to use them later.
#        fig2,ax3=plt.subplots()
#        fig2,ax4=plt.subplots()
#        ax3.scatter(zPos_used,bzField_used)
#        ax3.set_title("Z field used in Fast ion orbit calc")
#        ax4.scatter(zPos_used,brField_used)
#        ax4.set_title("R field used in Fast ion orbit calc")
        return sol
    if fieldType==1:
        return solve_ivp(lambda t,y: lorentz_cartesian_uniform(t, y, inputs),[tArray[0],tArray[-1]],y0,t_eval=tArray)
    if fieldType==2:
        return solve_ivp(lambda t,y: lorentz_cartesian_toroidal(t, y, inputs),[tArray[0],tArray[-1]],y0,t_eval=tArray)


def closest(lst, K):
      
     lst = np.asarray(lst)
     idx = (np.abs(lst - K)).argmin()
     return int(lst[idx])
