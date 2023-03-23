# 
# Replot kunal's results 
#

###########################################################################

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy.interpolate import griddata

###########################################################################

dataDest='C:/Users/kunal/OneDrive - UW-Madison/WHAM/Processed Data/'

fname1 = dataDest+'density.npz'
fname2 = dataDest+'fus_flux_axial_dependance.npz'

with np.load(fname1) as data:
#    print(data.files)
    ne = data['densArr']/1.e19
    R = data['R']
    Z = data['Z']; #print('Z.shape = ',Z.shape)

with np.load(fname2) as data:
#    print(data.files)
    nu = data['fusArr']
    z = data['Z']

###########################################################################

# Create regular grid for interpolation 
zGrid,rGrid = np.mgrid[0:1.00:101j, 0:0.1:51j]
# Symmetrize about r=0 for better interpolation
Zs = np.ravel(Z)
Rs = np.ravel(R)
nes = np.ravel(ne)
Zs = np.concatenate((Zs,Zs)); #print('Zs.shape = ',Zs.shape)
Rs = np.concatenate((Rs,-Rs))
nes = np.concatenate((nes,nes))
# Use griddata to interpolate onto regular grid
neGrid = griddata((Zs,Rs),nes,(zGrid,rGrid),method='cubic',fill_value=0)

###########################################################################

# Begin plotting routines
fig = plt.figure(figsize=(4,3))
plt.subplots_adjust(left=.1,right=.92,top=.95,bottom=.1)

gs = GridSpec(100,100)

axNe = fig.add_subplot(gs[:65,:85])
axCB = fig.add_subplot(gs[:65,90:95])
axNu = fig.add_subplot(gs[70:,:85])

#im0 = axNe.contourf(Z,R,ne,cmap='inferno')
im0 = axNe.contourf(zGrid,rGrid,neGrid,20,cmap='inferno')
axNe.scatter(Z,R,s=.1,c='k')
plt.colorbar(im0,cax=axCB,ticks=np.arange(6))
axNe.set_xlim((zGrid[0,0],zGrid[-1,0]))

axNu.plot(z,nu,'k',label='$\\nu$')
#axNu.plot(zGrid[:,0],neGrid[:,0],'r',label='$n_e$ [10$^{19}$/m$^3$]')
axNu.set_xlim((zGrid[0,0],zGrid[-1,0]))
axNu.set_ylim((0,1.1*np.amax(nu)))
axNu.legend(loc='lower left')

axNu1 = axNu.twinx()
axNu1.plot(zGrid[:,0],neGrid[:,0],'r',label='$n_e$')
axNu1.tick_params(labelcolor='r',color='r')
axNu1.legend(loc='upper right')
axNu1.set_ylim((0,1.1*np.amax(neGrid[:,0])))


axNu.text(0.5,-.1,'Z [m]',transform=axNu.transAxes,ha='center',va='top')
axNu.text(-.07,.5,'[W/m$^2$/sr]',transform=axNu.transAxes,ha='right',va='center',rotation=90)
axNu1.text(1.1,.5,'[10$^{19}$/m$^3$]',transform=axNu1.transAxes,ha='left',va='center',rotation=90,color='r')

#axNe.set_title('CQL3D Result, WHAM B=0.86 T, 1 MW NBI')
axNe.text(-0.07,.5,('R [m]'),transform=axNe.transAxes,ha='right',va='center',rotation=90)
axCB.set_ylabel('$n_e$ [10$^{19}$/m$^3$]')

axNe.set_yticks([0.0,0.02,0.04,0.06,0.08,0.1])
axNe.set_yticklabels(['0.0','','','','','0.1'])


axNe.set_xticklabels([])
axNu.set_xticklabels(['0.0','','','','','1.0'])

figname = 'C:/Users/kunal/OneDrive - UW-Madison/WHAM/Plots/CQL3D_WHAM_B0.86_NBIonly.png'
plt.savefig(figname,dpi=300); print('Fig saved as ',figname)

plt.show()
# plt.close()

    


