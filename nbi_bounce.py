# -*- coding: utf-8 -*-
"""
Created on Thu Aug 12 16:56:45 2021

@author: kunal
"""

import numpy as np
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 22})

Bfinal=np.load('Bfinal_01beta.npy')

fig,ax=plt.subplots(figsize=(12,8))
ax.hist(Bfinal,bins=30)
ax.set_title('Bounce Field Strength of NBI Particles')
ax.set_ylabel('Number of Particles')
ax.set_xlabel('Magnetic Field Strength [T]')
plt.show()