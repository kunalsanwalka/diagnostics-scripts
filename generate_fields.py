import numpy as np
from pleiades import RectMesh, compute_equilibrium, write_eqdsk, RectangularCoil
from pleiades.configurations import WHAM
from pleiades.configurations.wham_vns import WHAM_VNS
from pleiades.analysis import get_gpsi, locs_to_vals, get_fieldlines
import matplotlib.pyplot as plt

def generate_fields(config_name,equil=False):

    mu0 = 4*np.pi*10**-7
# create the WHAM device
    print("Buiding WHAM Field Profile")

    if (config_name == "WHAM Phase 1"):
        wham = WHAM() 
        #Set grid size
        rmax = 0.3
        zmax = 1.2
        nr = 501
        nz = 501
        wham.w7a1.current = 2850
        wham.w7a2.current = 2850
    elif (config_name == "WHAM Phase 2"):
        wham = WHAM() 
        #Set grid size
        rmax = 0.3
        zmax = 1.2
        nr = 501
        nz = 501
        wham.w7a1.current = 14550
        wham.w7a2.current = 14550
    elif (config_name == "WHAM VNS"):
        wham = WHAM_VNS() 
        #Set grid size
        rmax = 0.8
        zmax = 1.3
        nr = 201
        nz = 801

    elif (config_name == "WHAM VNS Helm"):
        wham = WHAM_VNS()

        #Helmholtz Coils for WHAM-VNS
        dr = 0.05
        dz = 0.04
        nr_turns = 24
        nz_turns = 4
        r_min = 2.5-(dr*nr_turns/2)
        z_cen = 1.25
        #Calc current with formula for helmholtz coil of given number of turns to make 2T
        r0 = r_min + dr*nr_turns/2
        cur = 1.717*(5/4)**(3/2)*r0/(mu0*nr_turns*nz_turns)
        wham.helm1 = RectangularCoil(r0, z_cen, nr=nr_turns, nz=nz_turns, dr=dr, dz=dz)
        wham.helm2 = RectangularCoil(r0, -z_cen, nr=nr_turns, nz=nz_turns, dr=dr, dz=dz)
        wham.helm1.current = cur
        wham.helm2.current = cur
        #Set grid size
        rmax = 1.0
        zmax = 2.5
        nr = 201
        nz = 1001

    e = 1.602E-19     #Elementary Charge
    mi = 3.3435E-27   #Deuteron Mass

    mesh = RectMesh(rmin=0, rmax=rmax, zmin=-zmax, zmax=zmax, nr=nr, nz=nz)
    R, Z = mesh.R, mesh.Z

    # set the grid (does all greens functions calculations right here)
    wham.mesh = mesh

    B = np.sqrt(wham.BR()**2 + wham.BZ()**2).reshape(R.shape)
    Bz = wham.BZ().reshape(R.shape)
    Br = wham.BR().reshape(R.shape)
    B_max = np.amax(B[:,0])   #On axis maximum magnetic field
    B_min = np.amin(B[:,0])   #On axis minimum magnetic field
    R_m = B_max/B_min   #Mirror Ratio

    psi = wham.psi().reshape(R.shape)

    #For doing equilibrium calculations
    if equil:
    # Setup pressure profile and compute P0 for a given desired initial beta
        a = 0.1
        alpha = 2.0
        beta0 = .4
        B0 = locs_to_vals(R,Z,B,[(0,0)])[0]
        P0 = beta0*B0**2/(2*4*np.pi*1E-7)
        #print("pressure ", P0)
    # Build pressure function of cylindrical radius
        Pfunc = lambda x: P0*(1-(x/a)**2)**alpha if x < a else 0
    # Get greens function for plasma currents
        gplas = get_gpsi(R,Z)
    # Compute equilibrium
        psieq,plas_currents,pfit = compute_equilibrium(R,Z,Pfunc,psi,gplas,maxiter=400,plotit=False)
        psi = psieq
    # Ouput as eqdsk
    else:
        plas_currents = np.zeros(R.shape)
    
    if ((config_name == "WHAM Phase 1") or (config_name == "WHAM Phase 2")):
        psi_lim = locs_to_vals(R,Z,psi, [(0.022,0.98)])[0]
    elif ((config_name == "WHAM VNS") or (config_name == "WHAM VNS Helm")):
        psi_lim = locs_to_vals(R,Z,psi, [(0.5,0)])[0]
        #psi_lim = locs_to_vals(R,Z,psi, [(0.22,z_cen_vns)])[0]
    else:
        psi_lim = locs_to_vals(R,Z,psi, [(0.1,0)])[0]

    write_eqdsk(R,Z,psi,plas_currents,"eqdsk",config_name+" Equilibrium",psi_lim=psi_lim)

    print("Profile Generated")
    return Z,R,B,psi


if __name__ == "__main__":
    Z,R,B,psi = generate_fields("WHAM VNS")

    print("Min B Field on axis:",np.amin(B[:,0]))

    fig = plt.figure(1)
    #Plot VNS Magnet Coils with Cryostats
    z0_cfs = 1.25
    r0_cfs = 0.453
    cfs_thick = 0.306
    cfs_width = 0.24
    z_cfs = np.array([z0_cfs-cfs_width/2,z0_cfs-cfs_width/2,z0_cfs+cfs_width/2,z0_cfs+cfs_width/2,z0_cfs-cfs_width/2])
    r_cfs = np.array([r0_cfs-cfs_thick/2,r0_cfs+cfs_thick/2,r0_cfs+cfs_thick/2,r0_cfs-cfs_thick/2,r0_cfs-cfs_thick/2])
    z_cfs_cryo = np.array([z_cfs[0]-0.04,z_cfs[1]-0.04,z_cfs[2]+0.04,z_cfs[3]+0.04,z_cfs[4]-0.04])
    r_cfs_cryo = np.array([r_cfs[0]-0.04,r_cfs[1]+0.04,r_cfs[2]+0.04,r_cfs[3]-0.04,r_cfs[4]-0.04])
    
    plt.fill(z_cfs,r_cfs,lw=3,color='r',zorder=3)
    plt.fill(z_cfs,-r_cfs,lw=3,color='r',zorder=3)
    plt.fill(-z_cfs,r_cfs,lw=3,color='r',zorder=3)
    plt.fill(-z_cfs,-r_cfs,lw=3,color='r',zorder=3)

    plt.plot(z_cfs_cryo,r_cfs_cryo,lw=3,color='b')
    plt.plot(z_cfs_cryo,-r_cfs_cryo,lw=3,color='b')
    plt.plot(-z_cfs_cryo,r_cfs_cryo,lw=3,color='b')
    plt.plot(-z_cfs_cryo,-r_cfs_cryo,lw=3,color='b')

    #Plot ModB Contours
    levels = [1,2,4,8,16,32]
    labeled_lvls = [1,2,4,8,16]
    contour_dict = dict()
    contour_dict[16] = "16 T"
    contour_dict[8] = "8 T"
    contour_dict[4] = "4 T"
    contour_dict[2] = "2 T"
    contour_dict[1] = "1 T"
    label_locs = np.array([(0,-0.7),(0.5,-0.6),(1,-0.7),(1,-0.5),(1,-0.1)])
   # label_locs = np.array([(1,-1.1),(1,-0.9),(1,-0.7),(1,-0.5),(1,-0.1)])
  #  colors = ['lightseagreen','teal','darkslategrey']

    plt.contour(Z,R,B,levels=levels,zorder=1)
    cs = plt.contour(Z,-R,B,levels=levels,zorder=1)
    plt.clabel(cs,labeled_lvls,fontsize=12,fmt=contour_dict,manual=label_locs)

    #Plot Plasma LCFS
    psi_lim = locs_to_vals(R,Z,psi, [(0.5,0)])[0]
    #psi_lim = locs_to_vals(R, Z, psi, [(0.22, z0_cfs)])[0]
#    psi_cen = locs_to_vals(R, Z, psi, [(0.05, z0_cfs)])[0]
    psis = [psi_lim]
    cs = plt.contour(Z,R,psi,levels=psis,colors=["orange"])
    plt.contour(Z,-R,psi,levels=psis,colors=["orange"])

    plt.title("WHAM VNS",fontsize=24)
    plt.xlim(-1.3,1.3)
    plt.ylim(-0.8,0.8)
    plt.xlabel("Z(m)",fontsize=16)
    plt.ylabel("R(m)",fontsize=16)
    plt.show()
