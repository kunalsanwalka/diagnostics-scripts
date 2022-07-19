# -*- coding: utf-8 -*-
"""
Created on Tue Jul 20 16:07:29 2021

Calculates beam profile for fast ion orbit integrator. This will likely be 
replaced by cql3d beam profiles, and comments are sparse because it is unlikely
this will ever be used. This was interpreted from IDL code found in [directory]

@author: micha
"""
import numpy as np


#Doesn't quite work yet, but got the bulk coded. 
#this will be moved to its own .py file when updated. 
def beam_path(y_0,zslope):
    
    #beam entrance point (change: was set up for MST Experiment. This was a torus)
    #look into IDL code for mirror development: beam codes in there.
    z0=0.52*np.sin(np.radians(19))
    r0_projection=1.5+0.52*np.cos(np.radians(19))
    # x0= -1*r0_projection*np.sin(np.radians(55))
    # y0= r0_projection*np.cos(np.radians(55))
    
    #point source of beam
    xps0=-2.55517
    yps0=0.97950
    zps0=0.26842
    
    dl_ps_ep = .943106
    
    dxdl2 = (y_0[0] - xps0)/dl_ps_ep
    dydl2 = (y_0[1] - yps0)/dl_ps_ep
    dzdl2 = (y_0[2] - zps0)/dl_ps_ep
    
    if zslope != 0:
        if zslope>0:
            zslope=-1*zslope
    else:
        zslope=-6
    
    
    path_len = np.sqrt(2*r0_projection**2+4*z0**2)
    
    l = ((np.linspace(0,1000,1000)+0.5)/1000)*path_len
    
    xt=xps0+dxdl2*l
    yt=yps0+dydl2*l
    zt=zps0+dzdl2*l
    
    i_bdry = np.hypot((np.hypot(xt,yt)-1.5),zt)<0.518
    
    x=xt[i_bdry]
    y=yt[i_bdry]
    z=zt[i_bdry]
    
    return [x,y,z]
