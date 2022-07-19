# -*- coding: utf-8 -*-
"""
Created on Mon Feb 15 16:18:03 2021

@author: kunal
"""

import netCDF4 as nc
import numpy as np
import matplotlib.pyplot as plt

#File location
filename='C:/Users/kunal/OneDrive - UW-Madison/WHAM/Data/CQL3D/220207_withNtor.nc'

#Open the file
ds=nc.Dataset(filename)

# #Go over each dimension
# for dim in ds.dimensions.values():
#     print('****************************************************************')
#     print(dim)

#How to access a specific dimension
# dim=ds.dimensions['xdim']

# Go over each variable
for var in ds.variables.values():
    print('****************************************************************')
    print(var)
    
#%% Sandbox

# #Energy per particle
# energym=np.array(ds['energym'][:])*1000 #Convert to eV

# #Normalized radius
# rya=np.array(ds['rya'][:])

# #Take the last timestep
# energyLastT=energym[-1]