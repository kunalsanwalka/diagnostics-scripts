# -*- coding: utf-8 -*-
"""
Created on Wed Mar  3 13:13:40 2021

@author: Jon Pizzo
"""

import matplotlib.pylab as plt
from matplotlib import cm
import matplotlib.patches as mpatches
import numpy as np
import tkinter as tk
import csv
import netCDF4 as nc
from make_genrayin_ech import make_genray_input
from generate_fields import generate_fields
from tkinter import filedialog
import os
from matplotlib.backends.backend_tkagg import (FigureCanvasTkAgg,  
NavigationToolbar2Tk) 
from pleiades.analysis import locs_to_vals
from skimage import measure
import scipy.odr as odr

class Genray_GUI(tk.Frame):
    def __init__(self):
        print("Initializing")
        super().__init__()
        self.initUI()
    
    
    def initUI(self):

        def update_wgd_zloc_entry(v):
            #Check if waveguide is inside port
            if (wgd_launch.get()):
                wgd_R2 = (wgd_zloc.get()-0.46503)**2+wgd_rloc.get()**2 
                port_R2 = ((13.5-1.25)*2.54/200)**2
                if (wgd_R2 > port_R2):
                    wgd_zloc_new = np.floor((np.sqrt(port_R2-(wgd_rloc.get())**2)+0.46503)*1000)/1000
                    wgd_zloc.set(wgd_zloc_new)
                    v = wgd_zloc_new
                    wgd_zloc_slider.set(wgd_zloc_new)
            wgd_zloc_entry.delete(0,"end")
            wgd_zloc_entry.insert(0,v)
        
        def update_wgd_rloc_entry(v):
            #Check if waveguide is inside port
            if (wgd_launch.get()):
                wgd_R2 = (wgd_zloc.get()-0.46503)**2+wgd_rloc.get()**2 
                port_R2 = ((13.5-1.25)*2.54/200)**2
                if (wgd_R2 > port_R2):
                    wgd_rloc_new = np.floor(np.sqrt(port_R2-(wgd_zloc.get()-0.46503)**2)*1000)/1000
                    wgd_rloc.set(wgd_rloc_new)
                    v = wgd_rloc_new
                    wgd_rloc_slider.set(wgd_rloc_new)
            wgd_rloc_entry.delete(0,"end")
            wgd_rloc_entry.insert(0,v)
                
        def update_ech_w_entry(v):
            ech_w_entry.delete(0,"end")
            ech_w_entry.insert(0,v)
        
        def update_cone_ang_entry(v):
            cone_ang_entry.delete(0,"end")
            cone_ang_entry.insert(0,v)
            
        def update_ne_core_entry(v):
            ne_core_entry.delete(0,"end")
            ne_core_entry.insert(0,v)
            
        def update_ne_bound_entry(v):
            ne_bound_entry.delete(0,"end")
            ne_bound_entry.insert(0,v)
            
        def update_te_core_entry(v):
            te_core_entry.delete(0,"end")
            te_core_entry.insert(0,v)
            
        def update_te_bound_entry(v):
            te_bound_entry.delete(0,"end")
            te_bound_entry.insert(0,v)

        def generate_preset(v):
            print("Generating Preset: "+preset_name.get())
            if (preset_name.get() == "X1 HFS"):
                wgd_zloc.set(0.533)
                wgd_rloc.set(0.139)
                wgd_ang.set(13.0)
                wgd_len.set(0.24)
                mir_ang.set(15.0)
                mir_len.set(0.05)
                mir2_ang.set(45.0)
                mir2_len.set(0.00)
                beam_len.set(0.13)
                if (wgd_launch.get()):
                    plot_start_point()
                else:
                    wgd_launch_but.invoke()
            if (preset_name.get() == "O to X1 HFS"):
                wgd_zloc.set(0.533)
                wgd_rloc.set(0.139)
                wgd_ang.set(15.0)
                wgd_len.set(0.24)
                mir_ang.set(33.6)
                mir_len.set(0.07)
                mir2_ang.set(72.0)
                mir2_len.set(0.07)
                beam_len.set(0.25)
                if (wgd_launch.get()):
                    plot_start_point()
                else:
                    wgd_launch_but.invoke()

        def swap_launch_method():
            if (wgd_launch.get()):   #If launching using waveguide
                start_pos_frame.grid_forget()
                wgd_launch_frame.grid(row=0,column=0,columnspan=2)
            else:
                wgd_launch_frame.grid_forget()
                start_pos_frame.grid(row=0,column=0,columnspan=2)
            plot_start_point()

        def set_xy_loc(event):
            x,y = event.xdata,event.ydata
            x = int(x*1000)/1000
            y = int(y*1000)/1000
            if ((set_launch_var.get() == 1) and (x != None) and (y != None)):
                if (not self.pt1_set):
                    zloc.set(x)
                    rloc.set(y)
                    self.pt1_set = True
                    print("Select Ending Position")
                else:
                    ang = np.rad2deg(np.arctan((x-zloc.get())/(rloc.get()-y)))
                    ang = int(ang*10)/10
                    theta.set(ang)
                    length = np.sqrt((x-zloc.get())**2 + (rloc.get()-y)**2)
                    beam_len.set(length)
                    self.pt1_set = False
                    set_launch_but.deselect()
                    plot_start_point()

        def set_launch_pos():
            #If the set launch button is off, do nothing
            if (set_launch_var.get() == 0):
                return
            self.pt1_set = False
            print("Select Starting Position")    

        def gauss_func(param,x):
            """
            A gaussian curve with parameters for height,variance,and location for ODR
            param[0] = height
            param[1] = mean
            param[2] = variance
            """
            return param[0]*np.exp(-(x-param[1])**2/(2*param[2]**2))


        def plot_ray_traj(z0,r0,phi,len_beam):
            #phi must be given in radians

            #First Check if Beam hits either mirror
            #Calculate a few values to see if beam intersects with mirror
            len_mir1 = mir_len.get() #length of mirror
            w_mir1 = 0.01 #width of mirror
            ang_mir1 = np.deg2rad(mir_ang.get()) #angle of mirror
            z4_mir1 = 0.8214-w_mir1*np.cos(ang_mir1)
            r4_mir1 = 0.05-w_mir1*np.sin(ang_mir1)
            hit1 = False

            len_mir2 = mir2_len.get() #length of mirror
            w_mir2 = 0.01 #width of mirror
            ang_mir2 = -np.deg2rad(mir2_ang.get()) #angle of mirror
            z4_mir2 = 0.8214+len_mir2*np.sin(ang_mir2)-w_mir2*np.cos(ang_mir2)
            r4_mir2 = -0.05-len_mir2*np.cos(ang_mir2)-w_mir2*np.sin(ang_mir2)

            hit2 = False

            #Mirror 1

            #Z0 transformed to mirror coords
            z0_trans1 = (z0 - z4_mir1)*np.cos(ang_mir1) + \
                (r0 - r4_mir1)*np.sin(ang_mir1)
            #Z location of end of beam transformed to mirror coords
            z_beam_end_trans1 = (z0 + len_beam*np.sin(phi) - z4_mir1)*np.cos(ang_mir1) + \
                (r0 - len_beam*np.cos(phi) - r4_mir1)*np.sin(ang_mir1)
            #Distance when beam crosses mirror plane
            len_cross1 = ((z0-z4_mir1)+(r0-r4_mir1)*np.tan(ang_mir1)) / (np.cos(phi)*np.tan(ang_mir1)-np.sin(phi))
            #R location where beam crossing mirror plane transformed to mirror coords (z_trans = 0)
            r_cross_trans1 = -(z0 + len_cross1*np.sin(phi) - z4_mir1)*np.sin(ang_mir1) + \
                (r0 - len_cross1*np.cos(phi) - r4_mir1)*np.cos(ang_mir1)
            if (z0_trans1 < 0 and z_beam_end_trans1 > 0 and
                r_cross_trans1 > 0 and r_cross_trans1 < len_mir1):
                #print("Beam Hits Mirror 1")
                hit1 = True
            #Mirror 2
            #Z0 transformed to mirror coords
            z0_trans2 = (z0 - z4_mir2)*np.cos(ang_mir2) + \
                (r0 - r4_mir2)*np.sin(ang_mir2)
            #Z location of end of beam transformed to mirror coords
            z_beam_end_trans2 = (z0 + len_beam*np.sin(phi) - z4_mir2)*np.cos(ang_mir2) + \
                (r0 - len_beam*np.cos(phi) - r4_mir2)*np.sin(ang_mir2)
            #Distance when beam crosses mirror plane
            len_cross2 = ((z0-z4_mir2)+(r0-r4_mir2)*np.tan(ang_mir2)) / (np.cos(phi)*np.tan(ang_mir2)-np.sin(phi))
            #R location where beam crossing mirror plane transformed to mirror coords (z_trans = 0)
            r_cross_trans2 = -(z0 + len_cross2*np.sin(phi) - z4_mir2)*np.sin(ang_mir2) + \
                (r0 - len_cross2*np.cos(phi) - r4_mir2)*np.cos(ang_mir2)

            if (z0_trans2 < 0 and z_beam_end_trans2 > 0 and
                r_cross_trans2 > 0 and r_cross_trans2 < len_mir2):
                #print("Beam Hits Mirror 2")
                hit2 = True

            if (hit1 and (not hit2)) or (hit1 and (len_cross1 <= len_cross2)):
                #Beam Hits 1st mirror before 2nd mirror
                z_bounce = z0+len_cross1*np.sin(phi)  #Z location of bounce
                r_bounce = r0-len_cross1*np.cos(phi)  #R location of bounce
                ang_refl = -phi + 2*ang_mir1 #Angle of reflected beam
                plot1.plot([z0,z_bounce],[r0,r_bounce],color='red')
                plot_ray_traj(z_bounce,r_bounce,ang_refl,len_beam-len_cross1)
            elif (hit2 and (not hit1)) or (hit2 and (len_cross1 > len_cross2)):
                #Beam Hits 2nd mirror before 1st mirror
                z_bounce = z0+len_cross2*np.sin(phi)  #Z location of bounce
                r_bounce = r0-len_cross2*np.cos(phi)  #R location of bounce
                ang_refl = -phi + 2*ang_mir2 #Angle of reflected beam
                plot1.plot([z0,z_bounce],[r0,r_bounce],color='red')
                plot_ray_traj(z_bounce,r_bounce,ang_refl,len_beam-len_cross2)
            else:
                #Beam hits neither mirror (should get here eventually)
                z_end = z0+len_beam*np.sin(phi)  #Z location of end
                r_end = r0-len_beam*np.cos(phi)  #R location of end
                plot1.plot([z0,z_end],[r0,r_end],color='red')
                tot_beam_len = beam_len.get()
                zloc.set(z_end - tot_beam_len*np.sin(phi))
                rloc.set(r_end + tot_beam_len*np.cos(phi))
                theta.set(np.rad2deg(phi))

        def plot_start_point():
            plot1.clear()
            plot_mirror_profile()
            if (wgd_launch.get()):
                z0 = wgd_zloc.get() #z loc of waveguide start
                r0 = wgd_rloc.get() #r loc of waveguide start
                phi = np.deg2rad(90-wgd_ang.get()) #angle of waveguide
                len_beam = beam_len.get()+wgd_len.get() #length of beam including waveguide section to plot
                plot_ray_traj(z0,r0,phi,len_beam)
            else:
                z0 = zloc.get()
                r0 = rloc.get()
                ang = np.deg2rad(theta.get())
                len_beam = beam_len.get()
                plot1.plot([z0,z0+len_beam*np.sin(ang)],[r0,r0-len_beam*np.cos(ang)],color='red')
            plot1.scatter(z0,r0,color='red')
            canvas.draw()
            
        def generate_field_profile(config_name):
            self.Z,self.R,self.modB,self.psi = generate_fields(config_name,equil=False)
            plot_start_point()

        def plot_mirror_profile():
            #Plot vacuum vessel
            z_vac = np.array([-1.2,-0.875,-0.8214,-0.8214,-0.6731,-0.6731])
            z_vac = np.append(z_vac,np.flip(z_vac)*-1)
            z_vac = np.append(z_vac,np.flip(z_vac))
            z_vac = np.append(z_vac,z_vac[0])
            x_vac = np.array([0.0275,0.0275,0.035,0.1250,0.1250,0.3683])
            x_vac = np.append(x_vac,np.flip(x_vac))
            x_vac = np.append(x_vac,x_vac*-1)
            x_vac = np.append(x_vac,x_vac[0])
            plot1.plot(z_vac,x_vac,color='b',lw=3)

            #Plot CFS Magnet Coils with Cryostats
            z0_cfs = 0.98
            cfs_thick = 0.11
            z_cfs = np.array([z0_cfs-cfs_thick/2,z0_cfs-cfs_thick/2,z0_cfs+cfs_thick/2,z0_cfs+cfs_thick/2,z0_cfs-cfs_thick/2])
            r_cfs = np.array([0.06,0.44,0.44,0.06,0.06])
            z_cfs_cryo = np.array([z0_cfs-0.3/2,z0_cfs-0.15/2,z0_cfs+0.15/2,z0_cfs+0.3/2,z0_cfs+0.3/2,z0_cfs-0.3/2,z0_cfs-0.3/2])
            r_cfs_cryo = np.array([0.035,0.0275,0.0275,0.035,0.5,0.5,0.035])
            
            plot1.fill(z_cfs,r_cfs,lw=3,color='r',zorder=3)
            plot1.fill(z_cfs,-r_cfs,lw=3,color='r',zorder=3)
            plot1.fill(-z_cfs,r_cfs,lw=3,color='r',zorder=3)
            plot1.fill(-z_cfs,-r_cfs,lw=3,color='r',zorder=3)
            
            plot1.plot(z_cfs_cryo,r_cfs_cryo,lw=3,color='b')
            plot1.plot(z_cfs_cryo,-r_cfs_cryo,lw=3,color='b')
            plot1.plot(-z_cfs_cryo,r_cfs_cryo,lw=3,color='b')
            plot1.plot(-z_cfs_cryo,-r_cfs_cryo,lw=3,color='b')
            
            if (wgd_launch.get()):
                #Plot 1cm Mirror
                l_mir = mir_len.get() #length of mirror
                w_mir = 0.01 #width of mirror
                ang_mir = np.deg2rad(mir_ang.get()) #angle of mirror
                z1_mir = 0.8214 
                r1_mir = 0.05
                z2_mir = z1_mir-l_mir*np.sin(ang_mir)
                r2_mir = r1_mir+l_mir*np.cos(ang_mir)
                z3_mir = z1_mir-l_mir*np.sin(ang_mir)-w_mir*np.cos(ang_mir)
                r3_mir = r1_mir+l_mir*np.cos(ang_mir)-w_mir*np.sin(ang_mir)
                z4_mir = z1_mir-w_mir*np.cos(ang_mir)
                r4_mir = r1_mir-w_mir*np.sin(ang_mir)
                z_mir = [z1_mir,z2_mir,z3_mir,z4_mir,z1_mir]
                r_mir = [r1_mir,r2_mir,r3_mir,r4_mir,r1_mir]
                plot1.fill(z_mir,r_mir,lw=0,color="silver")

                #Plot mirror #2
                l_mir2 = mir2_len.get() #length of mirror
                w_mir2 = 0.01 #width of mirror
                ang_mir2 = np.deg2rad(mir2_ang.get())#np.deg2rad(mir_ang.get()) #angle of mirror
                z1_mir2 = 0.8214 
                r1_mir2 = -0.05
                z2_mir2 = z1_mir2-l_mir2*np.sin(ang_mir2)
                r2_mir2 = r1_mir2-l_mir2*np.cos(ang_mir2)
                z3_mir2 = z1_mir2-l_mir2*np.sin(ang_mir2)-w_mir2*np.cos(ang_mir2)
                r3_mir2 = r1_mir2-l_mir2*np.cos(ang_mir2)+w_mir2*np.sin(ang_mir2)
                z4_mir2 = z1_mir2-w_mir2*np.cos(ang_mir2)
                r4_mir2 = r1_mir2+w_mir2*np.sin(ang_mir2)
                z_mir2 = [z1_mir2,z2_mir2,z3_mir2,z4_mir2,z1_mir2]
                r_mir2 = [r1_mir2,r2_mir2,r3_mir2,r4_mir2,r1_mir2]
                plot1.fill(z_mir2,r_mir2,lw=0,color="silver")

                #Plot Waveguide
                r0_wgd = wgd_rloc.get()   #R location of waveguide into vessel
                z0_wgd = wgd_zloc.get()  #Z location of waveguide into vessel
                ang_wgd = np.deg2rad(wgd_ang.get())
                l_wgd = wgd_len.get()  #Length of waveguide
                r_wgd = 1.25*2.54/2/100  #Radius of waveguide
                z_splitwg = np.array([z0_wgd+r_wgd*np.sin(ang_wgd),z0_wgd+r_wgd*np.sin(ang_wgd)+l_wgd*np.cos(ang_wgd)])
                r_splitwg = np.array([r0_wgd+r_wgd*np.cos(ang_wgd),r0_wgd+r_wgd*np.cos(ang_wgd)-l_wgd*np.sin(ang_wgd)])
                plot1.plot(z_splitwg,r_splitwg,lw=2,color='black')
                z_splitwg -= 2*r_wgd*np.sin(ang_wgd)
                r_splitwg -= 2*r_wgd*np.cos(ang_wgd)
                plot1.plot(z_splitwg,r_splitwg,lw=2,color='black')
                ech_circ = plt.Circle((z0_wgd,r0_wgd),r_wgd,fill=False,color="black",lw=2)
                plot1.add_artist(ech_circ)
                #Plot top port
                port_circ = plt.Circle((0.46503,0),13.5*2.54/200,fill=False,color="black",lw=2,ls='dashed')
                plot1.add_artist(port_circ)
                

            #Plot ModB Contours
            levels = [1.33,2,4]
            labeled_lvls = [1.33,2,4]
            contour_dict = dict()
            contour_dict[4] = " 4 T "
            contour_dict[2] = " 2 T "
            contour_dict[1.33] = "1.33 T"
            label_locs = np.array([(0.58,-0.15),(0.68,-0.15),(0.78,-0.15)])
            colors = ['lightseagreen','teal','darkslategrey']

            plot1.contour(self.Z,self.R,self.modB,levels=levels,zorder=1,colors=colors)
            cs = plot1.contour(self.Z,-self.R,self.modB,levels=levels,zorder=1,colors=colors)
            plot1.clabel(cs,labeled_lvls,fontsize=12,fmt=contour_dict,manual=label_locs)
            
            #Plot Plasma LCFS
            psi_lim = locs_to_vals(self.R, self.Z, self.psi, [(0.018, z0_cfs)])[0]
            psis = [psi_lim]
            cs = plot1.contour(self.Z,self.R,self.psi,levels=psis,colors=["g"])
            plot1.contour(self.Z,-self.R,self.psi,levels=psis,colors=["g"])
            
            #Plot Limiters
            psi_lim = locs_to_vals(self.R, self.Z, self.psi, [(0.022,0.98)])[0] #Map location to flux surface
            z0_lim = 0.6
            temp = cs.collections[-1].get_paths()[0]
            temp = temp.vertices
            self.z_lcfs = temp[:,0]
            self.x_lcfs = temp[:,1]
            r0_lim = self.x_lcfs[np.argmin(np.abs(self.z_lcfs-z0_lim))]
            
            z_lim = np.array([z0_lim-0.005,z0_lim-0.005,z0_lim+0.005,z0_lim+0.005])
            r_lim = np.array([r0_lim+0.005,0.3683,0.3683,r0_lim+0.005])
            lim_curve = mpatches.Wedge((z0_lim,r0_lim+0.005),0.005,180,360,color="grey",lw=2)
            plot1.add_artist(lim_curve)
            plot1.plot(z_lim,r_lim,color='grey',lw=2)
            lim_curve = mpatches.Wedge((z0_lim,-r0_lim-0.005),0.005,0,180,color="grey",lw=2)
            plot1.add_artist(lim_curve)
            plot1.plot(z_lim,-r_lim,color='grey',lw=2)
            
            plot1.set_xlim(xmin.get(),xmax.get())
            plot1.set_ylim(ymin.get(),ymax.get())
            plot1.set_xlabel("Z(m)")
            plot1.set_ylabel("R(m)")
        
        def run_genray():
            plot_start_point()
            #Make the genray input file
            make_genray_input(n_e0=ne_core.get(),n_eb=ne_bound.get(),Te0=te_core.get(),Teb=te_bound.get(),
                              rst=rloc.get(),zst=zloc.get(),betast=theta.get(),alpha1=ech_w.get(),alpha2=cone_ang.get()
                              ,freq=wave_freq.get(),mode=mode.get(),harmonic=harm.get(),nrays=nrays.get(),
                              ncones=ncones.get(),mnemonic=name.get())
            print("genray.in File Written")
            #Run genray with the input file
#            eqdsk_file = filedialog.askopenfilename(initialdir="/home/pizzo/genray_gui",title="Select eqdsk file")
#            os.system("cp "+eqdsk_file+" .")
            print("Running Genray")
            os.system("/home/mstfit/cql3d/genray-c_201001.2/xgenray-c")            

            #Show results
            print("Plotting Genray Results")
            plot_genray_results()

        def plot_genray_results():
            ncfilename = filedialog.askopenfilename(initialdir="/home/pizzo/genray_gui",title="Select Genray Output File")
            ds = nc.Dataset(ncfilename)

            #Get mnemonic from nc file
            mnemon = ds['mnemonic'][:]
            mnemon = mnemon[~mnemon.mask].tostring().decode('UTF-8')
            name.set(mnemon)

            wx = ds['wx'][:]/100
            wy = ds['wy'][:]/100
            wz = ds['wz'][:]/100
            wr = ds['wr'][:]/100
            ws = ds['ws'][:]/100
            delpwr = ds['delpwr'][:]            
            nrayelt = ds['nrayelt'][:]  

            plot1.clear()

            #Mirror the plot of beams when it hits mirror
            if (wgd_launch.get()):
                #Plot beam inside of waveguide
                wgd_z0 = wgd_zloc.get()
                wgd_r0 = wgd_rloc.get()
                ang_wgd = np.deg2rad(90-wgd_ang.get())
                beam_z = [wgd_z0,wgd_z0+(wgd_len.get())*np.sin(ang_wgd)]
                beam_r = [wgd_r0,wgd_r0-(wgd_len.get())*np.cos(ang_wgd)]
                plot1.plot(beam_z,beam_r,color='r')

                len_mir1 = mir_len.get() #length of mirror
                w_mir1 = 0.01 #width of mirror
                ang_mir1 = np.deg2rad(mir_ang.get()) #angle of mirror
                z4_mir1 = 0.8214-w_mir1*np.cos(ang_mir1) #bottom left corner of mirror z position
                r4_mir1 = 0.05-w_mir1*np.sin(ang_mir1) #bottom left corner of mirror r position

                len_mir2 = mir2_len.get() #length of mirror
                w_mir2 = 0.01 #width of mirror
                ang_mir2 = -np.deg2rad(mir2_ang.get()) #angle of mirror
                z4_mir2 = 0.8214+len_mir2*np.sin(ang_mir2)-w_mir2*np.cos(ang_mir2)
                r4_mir2 = -0.05-len_mir2*np.cos(ang_mir2)-w_mir2*np.sin(ang_mir2)
                
                #Starting from end, track backwards and reflect every time it hits a mirror
                for i in range(len(wx)):
                    istop = nrayelt[i]
                    for j in range(istop-1,0,-1):
                        x1 = wx[i,j]          
                        x2 = wx[i,j-1]
                        z1 = wz[i,j]
                        z2 = wz[i,j-1]
                        
                        z1_trans1 = (z1 - z4_mir1)*np.cos(ang_mir1) + (x1 - r4_mir1)*np.sin(ang_mir1)
                        z2_trans1 = (z2 - z4_mir1)*np.cos(ang_mir1) + (x2 - r4_mir1)*np.sin(ang_mir1)
                        x1_trans1 = -(z1 - z4_mir1)*np.sin(ang_mir1) + (x1 - r4_mir1)*np.cos(ang_mir1)
                        x2_trans1 = -(z2 - z4_mir1)*np.sin(ang_mir1) + (x2 - r4_mir1)*np.cos(ang_mir1)
                        
                        z1_trans2 = (z1 - z4_mir2)*np.cos(ang_mir2) + (x1 - r4_mir2)*np.sin(ang_mir2)
                        z2_trans2 = (z2 - z4_mir2)*np.cos(ang_mir2) + (x2 - r4_mir2)*np.sin(ang_mir2)
                        x1_trans2 = -(z1 - z4_mir2)*np.sin(ang_mir2) + (x1 - r4_mir2)*np.cos(ang_mir2)
                        x2_trans2 = -(z2 - z4_mir2)*np.sin(ang_mir2) + (x2 - r4_mir2)*np.cos(ang_mir2)

                        if ((z1_trans1>=0 and z2_trans1<0) or (z1_trans1<0 and z2_trans1>=0)):
                            #Next point crosses mirror plane
                            if ((x2_trans1 > 0) and (x2_trans1 < len_mir1)):
                                #Next point is between mirror bounce points
                                #print("Crosses Mirror 1")
                                dz = (z2-z1)
                                dx = (x2-x1)
                                ang = np.arctan(dz/dx)
                                if dz > 0 and dx < 0:
                                    ang += np.pi
                                elif dz < 0 and dx < 0:
                                    ang -= np.pi   

                                #Distance traveled by beam before bounce
                                temp = np.sqrt((wx[i,:j]-wx[i,j])**2 + (wz[i,:j]-wz[i,j])**2)
                                wx[i,:j] += (temp)*(np.cos(2*ang_mir1+ang)-np.cos(ang))
                                wz[i,:j] -= (temp)*(np.sin(2*ang_mir1+ang)+np.sin(ang))
                        elif ((z1_trans2>=0 and z2_trans2<0) or (z1_trans2<0 and z2_trans2>=0)):
                            #Next point crosses mirror plane
                            if ((x2_trans2 > 0) and (x2_trans2 < len_mir2)):
                                #Next point is between mirror bounce points
                                #print("Crosses Mirror 1")
                                dz = (z2-z1)
                                dx = (x2-x1)
                                ang = np.arctan(dz/dx)
                                if dz > 0 and dx < 0:
                                    ang += np.pi
                                elif dz < 0 and dx < 0:
                                    ang -= np.pi
                                
                                #Distance traveled by beam before bounce (in x-z plane)
                                temp = np.sqrt((wx[i,:j]-wx[i,j])**2 + (wz[i,:j]-wz[i,j])**2)
                                wx[i,:j] += (temp)*(np.cos(2*ang_mir2+ang)-np.cos(ang))
                                wz[i,:j] -= (temp)*(np.sin(2*ang_mir2+ang)+np.sin(ang))

            #Plot Rays
            for i in range(len(wx)):
                istop = nrayelt[i]
                plot1.scatter(wz[i,:istop],wx[i,:istop],c=cm.Reds(delpwr[i,:istop]/np.max(delpwr[i,:istop])),edgecolor='none',marker='.')
            plot_mirror_profile()

            #Plot percent energy absorbed
            pow_abs_frac = ds['power_total'][:]/ds['power_inj_total'][:]
            plot1.text(0,1.1,"{:.2f}% Power Absorbed".format(pow_abs_frac*100),transform=plot1.transAxes)

            #Plot Polarization at WG End
            #Complex Electric Field Polarization in Stix Frame (at starting point)
            Ex_s = ds['cwexde'][:][:,0,0]
            Ey_s = ds['cweyde'][:][:,0,0]
            Ez_s = ds['cwezde'][:][:,0,0]
            #Background Magnetic Field components (at starting point)
            Bx  = ds['sb_x'][:][0,0]
            By  = ds['sb_y'][:][0,0]
            Bz  = ds['sb_z'][:][0,0]
            #N refractive index component (at starting point)
            wn_x = ds['wn_x'][:][0,0]
            wn_y = ds['wn_y'][:][0,0]
            wn_z = ds['wn_z'][:][0,0]
            #Transform Electric field components to machine frame
            alpha = np.arctan(Bx/Bz)
            Ex_m = -np.cos(alpha)*Ex_s + np.sin(alpha)*Ez_s
            Ey_m = -Ey_s
            Ez_m = np.sin(alpha)*Ex_s + np.cos(alpha)*Ez_s
            #Transform Electric Field components to z=k, y=y_m frame
            beta = np.arctan(wn_x/wn_z)
            Ex_k = -np.cos(beta)*Ex_m + np.sin(beta)*Ez_m
            Ey_k = Ey_m
            Ez_k = -np.sin(beta)*Ex_m - np.cos(beta)*Ez_m

            #Jones Vector Parameters
            E_0x = np.sqrt(Ex_k[0]**2+Ex_k[1]**2)
            E_0y = np.sqrt(Ey_k[0]**2+Ey_k[1]**2)
            phi_x = np.arctan(Ex_k[1]/Ex_k[0])
            phi_y = np.arctan(Ey_k[1]/Ey_k[0])

            plot1.text(0.5,1.1,"Jones Vector Parameters at waveguide end",transform=plot1.transAxes)
            plot1.text(0.5,1.06,"$E_x$ = {:.2f}\t$E_y$ = {:.2f}".format(E_0x,E_0y),transform=plot1.transAxes)
            plot1.text(0.5,1.02,"$\phi_x$ = {:.2f}\t$\phi_y$ = {:.2f}".format(phi_x,phi_y),transform=plot1.transAxes)
            canvas.draw()

        def plot_Pofr(scaled=False):
            ncfilename = filedialog.askopenfilename(initialdir="/home/pizzo/genray_gui",title="Select Genray Output File")
            ds = nc.Dataset(ncfilename)
            
            #Get mnemonic from nc file
            mnemon = ds['mnemonic'][:]
            mnemon = mnemon[~mnemon.mask].tostring().decode('UTF-8')
            name.set(mnemon)

            psi_lim = ds['psilim'][:]
            wz = ds['wz'][:]/100
            wr = ds['wr'][:]/100
            delpwr = ds['delpwr'][:]            
            nrayelt = ds['nrayelt'][:]
            #Power absorbed on grid
            pow_dep_rz = ds['spwr_rz_e'][:]
            r_grid = ds['Rgrid'][:]
            z_grid = ds['Zgrid'][:]
            
            #Loop through array of power absorbed only for non-zero grid points
            pow_dep_rz_pts = np.argwhere(pow_dep_rz > 0)
            rpts = np.zeros(len(pow_dep_rz_pts))
            pow_pts = np.zeros(len(rpts))

            if len(pow_dep_rz_pts) > 0:
                z_max = np.amax(z_grid[pow_dep_rz_pts[:,0]]) #Largest Z value with deposition
            else:
                z_max = 0
            #Find r value of LCFS at the largest z value
          #  contours = measure.find_contours(self.psi, psi_lim)
          #  iz = np.argmin(np.abs(self.Z[:,0] - z_max))
          #  izz = np.argmin(np.abs(contours[0][:,0]-iz))
          #  rlim = self.R[0,int(contours[0][izz,1])]
            rlim = self.x_lcfs[np.argmin(np.abs(self.z_lcfs-z_max))]

            r_spacing = np.max([r_grid[1] - r_grid[0],self.R[0,1] - self.R[0,0]])
            num_bins = int(1+rlim//r_spacing) #Define the number of bins based on the grid spacing of r
            rs = np.linspace(0,1,num_bins)
            Pofr = np.zeros(len(rs))

            for i in range(len(rpts)):
                pow_dep = pow_dep_rz[pow_dep_rz_pts[i][0],pow_dep_rz_pts[i][1]]
                rpt = r_grid[pow_dep_rz_pts[i][1]]
                zpt = z_grid[pow_dep_rz_pts[i][0]]
                #Scale r such that LCFS at that z value = 1
                rlim = self.x_lcfs[np.argmin(np.abs(self.z_lcfs-zpt))]
   
                pow_pts[i] = pow_dep
                rpts[i] = rpt/rlim
                r0 = rpt/rlim
                idx = np.argmin(np.abs(rs-r0))

                Pofr[idx] += pow_dep
                #Deposit the power into bins according to how close r0 is to the bin
          #      if r0 < rs[idx]:
          #          Pofr[idx] += pow_dep * abs(rs[idx]-r0)*(num_bins-1)
          #          Pofr[idx-1] += pow_dep * abs(rs[idx-1]-r0)*(num_bins-1)
          #      elif (r0 > rs[idx] and idx < (num_bins-1)):
          #          Pofr[idx] += pow_dep * abs(rs[idx]-r0)*(num_bins-1)
          #          Pofr[idx+1] += pow_dep * abs(rs[idx+1]-r0)*(num_bins-1)
          #      else:
          #          Pofr[idx] += pow_dep       
            
            plot1.clear()

            #Check if there is an outlier and remove it if it exists
            Pofr_sorted = np.sort(Pofr)
            if (Pofr_sorted[-1] > 2*Pofr_sorted[-2]):
                Pofr = np.delete(Pofr,np.argmax(Pofr))
                rs = np.delete(rs,np.argmax(Pofr))
            
            if scaled:
                r0lim = self.x_lcfs[np.argmin(np.abs(self.z_lcfs-0))]
                idx0 = np.argmax(self.modB[:int(len(self.modB[:,0])/2),0])  #Index of left mirror point
                idx1 = np.argmax(self.modB[int(len(self.modB[:,0])/2):,0])+int(len(self.modB[:,0])/2) #Index of right mirror point
                
                last_vol = 0
                for i in range(len(rs)):
                    if i < len(rs)-1:
                        rho_val = (rs[i+1]+rs[i]) / 2
                    else:
                        rho_val = rs[i]
                    psi_val = locs_to_vals(self.R,self.Z,self.psi,[(rho_val*r0lim,0)])[0]
                    flux_surf = np.zeros(idx1-idx0+1)
                    for j in range(idx1-idx0+1):
                        flux_surf[j] = np.interp(psi_val,self.psi[idx0+j,:],self.R[idx0+j,:])
                    volume = np.pi*np.trapz(flux_surf**2,self.Z[idx0:idx1+1,0]) - last_vol
                    last_vol = volume
                    #print("Rmax = {:.4f}\tVolume = {:.4f}".format(rho_val*r0lim,volume))
                    Pofr[i] /= volume

            plot1.scatter(rs,Pofr)

            #Do Gaussian Fit to Data
            gauss = odr.Model(gauss_func)
            odr_data = odr.RealData(rs,Pofr/np.amax(Pofr),sx=1/(num_bins-1))
            regressed_model = odr.ODR(odr_data,gauss,beta0=[1,rs[np.argmax(Pofr)],0.05])
            output = regressed_model.run()
            output.pprint()
            params = output.beta
            gauss_rs = np.linspace(0,1,200)
            gauss_fit = gauss_func(params,gauss_rs)

            plot1.plot(gauss_rs,gauss_fit*np.amax(Pofr))

            plot1.set_xlabel("Radius")
            if scaled:
                plot1.set_ylabel("Scaled P(R) [erg/m^3]")
            else:
                plot1.set_ylabel("P(R) [erg]")
            canvas.draw()

        self.master.title("Genray GUI")

        self.rowconfigure(0, pad=30)   #Title Row
        self.rowconfigure(1, pad=10)   #Top Options Row
        self.rowconfigure(2, pad=20)   #Sliders Row
        self.rowconfigure(3, pad=20)   #Modes/Options Row
        self.rowconfigure(4, pad=20)   #Start/End Buttons Row

        self.columnconfigure(0, pad=20)
        self.columnconfigure(1, pad=20)
        self.columnconfigure(2, pad=20)
    
        title_frame = tk.Frame(self)
        title_frame.grid(row=0,column=0)
        
        text_font = ("Calibri", 20, "bold")
        slider_font = ("Calibri", 12,"normal")
        title = tk.Label(master=title_frame,text="Genray GUI",font=text_font,width=20,height=3)
        title.grid(row=0,columnspan=2)

        #Options at the top
        top_options_frame = tk.Frame(self)
        top_options_frame.grid(row=1,column=0)
        top_options_frame.columnconfigure(0, pad=30)
        top_options_frame.columnconfigure(1, pad=30)
        top_options_frame.columnconfigure(2, pad=30)

        preset_name = tk.StringVar()
        preset_name.set("")
        preset_options = tk.OptionMenu(top_options_frame,preset_name,*["X1 HFS","O to X1 HFS"],command=generate_preset)
        preset_options_label = tk.Label(master=top_options_frame,text="Launch Preset",font=slider_font) 
    
        wgd_launch = tk.IntVar()
        wgd_launch.set(0)
        wgd_launch_but = tk.Checkbutton(top_options_frame,variable=wgd_launch,height=5,width=5,command=swap_launch_method)
        wgd_launch_label = tk.Label(master=top_options_frame,text="Waveguide Mode",font=slider_font)

        name = tk.StringVar()
        name.set("default")
        name_entry = tk.Entry(top_options_frame,textvariable=name,width=20)
        name_entry_label = tk.Label(master=top_options_frame,text="Mnemonic",font=slider_font,width=15,height=1)

        preset_options.grid(row=0,column=0)
        preset_options_label.grid(row=1,column=0)
        wgd_launch_but.grid(row=0,column=1)
        wgd_launch_label.grid(row=1,column=1)
        name_entry.grid(row=0,column=2)
        name_entry_label.grid(row=1,column=2)
        
        #Frame for defining launch parameters
        slider_frame = tk.Frame(self)
        
        slider_frame.rowconfigure(0, pad=10)
        slider_frame.rowconfigure(1, pad=10)
        slider_frame.rowconfigure(2, pad=10)
        slider_frame.rowconfigure(3, pad=10)
        slider_frame.rowconfigure(4, pad=10)
        
        slider_frame.columnconfigure(0, pad=10)
        slider_frame.columnconfigure(1, pad=10)
        
        #Start Parameters Frame
        start_pos_frame = tk.Frame(master=slider_frame)

        #Button to choose start parameters with mouse click
        set_launch_var = tk.IntVar()
        set_launch_var.set(0)
        set_launch_but = tk.Checkbutton(master=start_pos_frame,command=set_launch_pos,
                                        variable=set_launch_var,width=5,height=5)
        set_launch_label = tk.Label(master=start_pos_frame,text="Set Launch\nPosition",font=slider_font,width=15)

        #Z location
        zloc = tk.DoubleVar()
        zloc.set(0.820)
        zloc_entry = tk.Entry(master=start_pos_frame,textvariable=zloc,width=10)

        zloc_label = tk.Label(master=start_pos_frame,text="Z Start (m)",font=slider_font,width=15)
        
        #R location        
        rloc = tk.DoubleVar()
        rloc.set(0.07)
        rloc_entry = tk.Entry(master=start_pos_frame,textvariable=rloc,width=10)
        rloc_label = tk.Label(master=start_pos_frame,text="R Start (m)",font=slider_font,width=15)
        
        #Launch Angle
        theta = tk.DoubleVar()
        theta.set(-60.0)
        theta_entry = tk.Entry(master=start_pos_frame,textvariable=theta,width=10)
        theta_label = tk.Label(master=start_pos_frame,text="Launch Angle",font=slider_font,width=15)

        set_launch_but.grid(row=0,column=0)
        set_launch_label.grid(row=1,column=0)
        zloc_entry.grid(row=0,column=1)
        zloc_label.grid(row=1,column=1)
        rloc_entry.grid(row=0,column=2)
        rloc_label.grid(row=1,column=2)
        theta_entry.grid(row=0,column=3)
        theta_label.grid(row=1,column=3)
        
        start_pos_frame.grid(row=0,column=0,columnspan=2)
        
        #Frames for using waveguide launch method (not shown unless option checked)
        #Waveguide Zloc Frame
        wgd_launch_frame = tk.Frame(master=slider_frame)
        
        wgd_zloc = tk.DoubleVar()
        wgd_zloc.set(0.533)
        wgd_rloc = tk.DoubleVar()
        wgd_rloc.set(0.139)

        #Waveguide Z location
        wgd_zloc_entry = tk.Entry(master=wgd_launch_frame,textvariable=wgd_zloc,width=8)
        wgd_zloc_label = tk.Label(master=wgd_launch_frame,text="Waveguide Z Loc (m)",font=slider_font,width=25,height=1)
        
        wgd_zloc_entry.grid(row=0,column=0)
        wgd_zloc_label.grid(row=1,column=0)
        
        #Waveguide R location
        wgd_rloc_entry = tk.Entry(master=wgd_launch_frame,textvariable=wgd_rloc,width=8)
        wgd_rloc_label = tk.Label(wgd_launch_frame,text="Waveguide R Loc (m)",font=slider_font,width=25,height=1)
        
        wgd_rloc_entry.grid(row=0,column=1)
        wgd_rloc_label.grid(row=1,column=1)

        #Wavguide angle/length controls
        wgd_ang = tk.DoubleVar()
        wgd_ang.set(13.0)
        wgd_ang_entry = tk.Entry(wgd_launch_frame,textvariable=wgd_ang,width=8)
        wgd_ang_label = tk.Label(wgd_launch_frame,text="Waveguide Angle",font=slider_font,width=20,height=1)
        
        wgd_ang_entry.grid(row=0,column=2)
        wgd_ang_label.grid(row=1,column=2)

        wgd_len = tk.DoubleVar()
        wgd_len.set(0.24)
        wgd_len_entry = tk.Entry(wgd_launch_frame,textvariable=wgd_len,width=8)
        wgd_len_label = tk.Label(wgd_launch_frame,text="Waveguide Length",font=slider_font,width=20,height=1)
        
        wgd_len_entry.grid(row=0,column=3)
        wgd_len_label.grid(row=1,column=3)

        #Mirror 1 angle/length controls
        mir_ang = tk.DoubleVar()
        mir_ang.set(15.0)
        mir_ang_entry = tk.Entry(wgd_launch_frame,textvariable=mir_ang,width=8)
        mir_ang_label = tk.Label(wgd_launch_frame,text="Mirror 1 Angle",font=slider_font,width=20,height=1)
        
        mir_ang_entry.grid(row=2,column=0)
        mir_ang_label.grid(row=3,column=0)

        mir_len = tk.DoubleVar()
        mir_len.set(0.05)
        mir_len_entry = tk.Entry(wgd_launch_frame,textvariable=mir_len,width=8)
        mir_len_label = tk.Label(master=wgd_launch_frame,text="Mirror 1 Length",font=slider_font,width=20,height=1)
        
        mir_len_entry.grid(row=2,column=1)
        mir_len_label.grid(row=3,column=1)

        #Mirror 2 angle/length controls
        mir2_ang = tk.DoubleVar()
        mir2_ang.set(15.0)
        mir2_ang_entry = tk.Entry(wgd_launch_frame,textvariable=mir2_ang,width=8)
        mir2_ang_label = tk.Label(wgd_launch_frame,text="Mirror 2 Angle",font=slider_font,width=20,height=1)
        
        mir2_ang_entry.grid(row=2,column=2)
        mir2_ang_label.grid(row=3,column=2)

        mir2_len = tk.DoubleVar()
        mir2_len.set(0.05)
        mir2_len_entry = tk.Entry(wgd_launch_frame,textvariable=mir2_len,width=8)
        mir2_len_label = tk.Label(master=wgd_launch_frame,text="Mirror 2 Length",font=slider_font,width=20,height=1)
        
        mir2_len_entry.grid(row=2,column=3)
        mir2_len_label.grid(row=3,column=3)

        #ECH cone half width Frame
        ech_w_frame = tk.Frame(master=slider_frame)
        
        ech_w = tk.DoubleVar()
        ech_w.set(2.45)
        ech_w_entry = tk.Entry(master=ech_w_frame,textvariable=ech_w,width=8)
        ech_w_slider = tk.Scale(master=ech_w_frame,from_=0.0, to=20.0,resolution=0.01,
                                orient="horizontal",width=20,length=200,command=update_ech_w_entry,showvalue=0)
        ech_w_slider.set(ech_w.get())
        ech_w_entry.bind("<Return>", lambda event: ech_w_slider.set(ech_w.get()))
        
        label = tk.Label(master=ech_w_frame,text="ECH Cone Half Width",font=slider_font,width=20,height=1)
        
        ech_w_entry.pack(side=tk.TOP)
        ech_w_slider.pack(side=tk.TOP)
        label.pack(side=tk.TOP)
        ech_w_frame.grid(row=2,column=0)
        
        #ECH cone angle Frame
        cone_ang_frame = tk.Frame(master=slider_frame)
        
        cone_ang = tk.DoubleVar()
        cone_ang.set(0.0)
        cone_ang_entry = tk.Entry(master=cone_ang_frame,textvariable=cone_ang,width=8)
        cone_ang_slider = tk.Scale(master=cone_ang_frame,from_=0.0, to=30.0,resolution=0.1,
                                   orient="horizontal",width=20,length=200,command=update_cone_ang_entry,showvalue=0)
        cone_ang_slider.set(cone_ang.get())
        cone_ang_entry.bind("<Return>", lambda event: cone_ang_slider.set(cone_ang.get()))
        
        label = tk.Label(master=cone_ang_frame,text="ECH Cone Start Angle",font=slider_font,width=20,height=1)
        
        cone_ang_entry.pack(side=tk.TOP)
        cone_ang_slider.pack(side=tk.TOP)
        label.pack(side=tk.TOP)
        cone_ang_frame.grid(row=2,column=1)
        
        #Core Density Frame
        ne_core_frame = tk.Frame(master=slider_frame)
        
        ne_core = tk.DoubleVar()
        ne_core.set(1.0)
        ne_core_entry = tk.Entry(master=ne_core_frame,textvariable=ne_core,width=8)
        ne_core_slider = tk.Scale(master=ne_core_frame,from_=0.01, to=10.0,resolution=0.01,
                                  orient="horizontal",width=20,length=200,command=update_ne_core_entry,showvalue=0)
        ne_core_slider.set(ne_core.get())
        ne_core_entry.bind("<Return>", lambda event: ne_core_slider.set(ne_core.get()))
        
        label = tk.Label(master=ne_core_frame,text="Core n (10^19 m^-3)",font=slider_font,width=25,height=1)
        
        ne_core_entry.pack(side=tk.TOP)
        ne_core_slider.pack(side=tk.TOP)
        label.pack(side=tk.TOP)
        ne_core_frame.grid(row=3,column=0)
        
        #Boundary Density Frame
        ne_bound_frame = tk.Frame(master=slider_frame)
        
        ne_bound = tk.DoubleVar()
        ne_bound.set(0.1)
        ne_bound_entry = tk.Entry(master=ne_bound_frame,textvariable=ne_bound,width=8)
        ne_bound_slider = tk.Scale(master=ne_bound_frame,from_=0.01, to=5.0,resolution=0.01,
                                   orient="horizontal",width=20,length=200,command=update_ne_bound_entry,showvalue=0)
        ne_bound_slider.set(ne_bound.get())
        ne_bound_entry.bind("<Return>", lambda event: ne_bound_slider.set(ne_bound.get()))
        
        label = tk.Label(master=ne_bound_frame,text="Boundary n (10^19 m^-3)",font=slider_font,width=25,height=1)
        
        ne_bound_entry.pack(side=tk.TOP)
        ne_bound_slider.pack(side=tk.TOP)
        label.pack(side=tk.TOP)
        ne_bound_frame.grid(row=3,column=1)
        
        #Core Electron Temp Frame
        te_core_frame = tk.Frame(master=slider_frame)
        
        te_core = tk.DoubleVar()
        te_core.set(500.0)
        te_core_entry = tk.Entry(master=te_core_frame,textvariable=te_core,width=8)
        te_core_slider = tk.Scale(master=te_core_frame,from_=0.1, to=5000,resolution=0.1,
                                  orient="horizontal",width=20,length=200,command=update_te_core_entry,showvalue=0)
        te_core_slider.set(te_core.get())
        te_core_entry.bind("<Return>", lambda event: te_core_slider.set(te_core.get()))
        
        label = tk.Label(master=te_core_frame,text="Core Te (eV)",font=slider_font,width=20,height=1)
        
        te_core_entry.pack(side=tk.TOP)
        te_core_slider.pack(side=tk.TOP)
        label.pack(side=tk.TOP)
        te_core_frame.grid(row=4,column=0)
        
        #Boundary Electron Temp Frame
        te_bound_frame = tk.Frame(master=slider_frame)
        
        te_bound = tk.DoubleVar()
        te_bound.set(100.0)
        te_bound_entry = tk.Entry(master=te_bound_frame,textvariable=te_bound,width=8)
        te_bound_slider = tk.Scale(master=te_bound_frame,from_=0.1, to=2500,resolution=0.1,
                                   orient="horizontal",width=20,length=200,command=update_te_bound_entry,showvalue=0)
        te_bound_slider.set(te_bound.get())
        te_bound_entry.bind("<Return>", lambda event: te_bound_slider.set(te_bound.get()))
        
        label = tk.Label(master=te_bound_frame,text="Boundary Te (eV)",font=slider_font,width=25,height=1)
        
        te_bound_entry.pack(side=tk.TOP)
        te_bound_slider.pack(side=tk.TOP)
        label.pack(side=tk.TOP)
        te_bound_frame.grid(row=4,column=1)
        
        slider_frame.grid(row=2,column=0)
        
        #---------- Options Frame -----------
        options_frame = tk.Frame(self)
        options_font = ("Calibri", 16,"bold")
        
        options_frame.columnconfigure(0, pad=40)
        options_frame.columnconfigure(1, pad=40)
        options_frame.columnconfigure(2, pad=40)
        options_frame.columnconfigure(3, pad=40)
        
        #Wham version to use
        config_name = tk.StringVar()
        config_name.set("WHAM Phase 1")
        config_name_menu = tk.OptionMenu(options_frame,config_name,*["WHAM Phase 1","WHAM Phase 2"],command=generate_field_profile)
        config_name_menu.config(font=options_font)
        config_name_menu.grid(row=0,column=0)
        label = tk.Label(master=options_frame,text="Coil Configuration",font=slider_font,width=20,height=1)
        label.grid(row=1,column=0)
        #Mode of Launch Frame
        mode = tk.StringVar()
        mode.set("X")
        modes = tk.OptionMenu(options_frame,mode,*["X","O"])
        modes.config(font=options_font)
        modes.grid(row=0,column=1)
        label = tk.Label(master=options_frame,text="ECH Mode",font=slider_font,width=10,height=1)
        label.grid(row=1,column=1)
        #Harmonic For Damping Frame
        harm = tk.IntVar()
        harm_box = tk.Spinbox(options_frame,from_=1,to=3,font=options_font,textvariable=harm,width=4)
        harm_box.grid(row=0,column=2)
        label = tk.Label(master=options_frame,text="Harmonic",font=slider_font,width=10,height=1)
        label.grid(row=1,column=2)
        #Wave Frequency Frame
        wave_freq = tk.DoubleVar()
        wave_freq.set(110.0)
        wave_freq_entry = tk.Entry(master=options_frame,textvariable=wave_freq,width=8)
        wave_freq_entry.grid(row=2,column=0)
        label = tk.Label(master=options_frame,text="Wave Frequency",font=slider_font,width=15,height=1)
        label.grid(row=3,column=0)
        
        nrays = tk.IntVar()
        nrays.set(10)
        nrays_box = tk.Spinbox(options_frame,from_=1,to=20,font=options_font,textvariable=nrays,width=4)
        nrays_box.grid(row=2,column=1)
        label = tk.Label(master=options_frame,text="Num Rays",font=slider_font,width=10,height=1)
        label.grid(row=3,column=1)

        ncones = tk.IntVar()
        ncones.set(1)
        ncones_box = tk.Spinbox(options_frame,from_=0,to=10,font=options_font,textvariable=ncones,width=4)
        ncones_box.grid(row=2,column=2)
        label = tk.Label(master=options_frame,text="Num Cones",font=slider_font,width=10,height=1)
        label.grid(row=3,column=2)
        
        options_frame.grid(row=3,column=0)
        
        #Buttons at the bottom
        button_frame = tk.Frame(self)
        button_frame.grid(row=4,column=0)
        
        start = tk.Button(master=button_frame,text="Run Genray",command=run_genray,width=20,height=3)
        start.grid(row=1,column=0)
        view_start_pt = tk.Button(master=button_frame,text="View Start Point",width=20,height=3,command=plot_start_point)
        view_start_pt.grid(row=1,column=1)
        plot_nc_but = tk.Button(master=button_frame,text="Plot Genray Output",width=20,height=3,command=plot_genray_results)
        plot_nc_but.grid(row=0,column=0)
        plot_Pofr_but = tk.Button(master=button_frame,text="Plot P(r)",width=20,height=3,command=plot_Pofr)
        plot_Pofr_but.grid(row=0,column=1)
        plot_Pofr_scaled_but = tk.Button(master=button_frame,text="Plot Scaled P(r)",width=20,height=3,command=lambda: plot_Pofr(scaled=True))
        plot_Pofr_scaled_but.grid(row=0,column=2)
        quit_but = tk.Button(master=button_frame,text="Quit",width=20,height=3,command=self.master.destroy)
        quit_but.grid(row=1,column=2)
        
        self.pack()
        
        #Plot Area
        canvas_frame = tk.Frame(self)
        canvas_frame.grid(column=1,row=0,rowspan=5)
        fig = plt.figure(figsize=(8,8))
        plot1 = fig.add_subplot(111)
        canvas = FigureCanvasTkAgg(fig,master=canvas_frame)
        self.pt1_set = False
        canvas.callbacks.connect('button_press_event', set_xy_loc)
        canvas.get_tk_widget().pack()
        #canvas.draw()
        
        #Plot options frame
        plot_options_frame = tk.Frame(self)
        plot_options_frame.grid(row=0,column=2,rowspan=5)
        plot_options_frame.rowconfigure(0, pad=5)
        plot_options_frame.rowconfigure(1, pad=5)
        plot_options_frame.rowconfigure(2, pad=5)
        plot_options_frame.rowconfigure(3, pad=5)
        plot_options_frame.rowconfigure(4, pad=5)
        plot_options_frame.rowconfigure(5, pad=5)

        plot_options_frame.columnconfigure(0, pad=5)
        plot_options_frame.columnconfigure(1, pad=5)
        plot_options_frame.columnconfigure(2, pad=5)
        plot_options_frame.columnconfigure(3, pad=5)


        #Plot axis ranges
        xmin = tk.DoubleVar()
        xmin.set(0.5)
        xmin_entry = tk.Entry(master=plot_options_frame,textvariable=xmin,width=8)
        xmin_entry.grid(row=0,column=0)
        label = tk.Label(master=plot_options_frame,text="x min",font=slider_font,width=10,height=1)
        label.grid(row=1,column=0)
        xmax = tk.DoubleVar()
        xmax.set(1.1)
        xmax_entry = tk.Entry(master=plot_options_frame,textvariable=xmax,width=8)
        xmax_entry.grid(row=0,column=1)
        label = tk.Label(master=plot_options_frame,text="x max",font=slider_font,width=10,height=1)
        label.grid(row=1,column=1)
        ymin = tk.DoubleVar()
        ymin.set(-0.3)
        ymin_entry = tk.Entry(master=plot_options_frame,textvariable=ymin,width=8)
        ymin_entry.grid(row=0,column=2)
        label = tk.Label(master=plot_options_frame,text="y min",font=slider_font,width=10,height=1)
        label.grid(row=1,column=2)
        ymax = tk.DoubleVar()
        ymax.set(0.3)
        ymax_entry = tk.Entry(master=plot_options_frame,textvariable=ymax,width=8)
        ymax_entry.grid(row=0,column=3)
        label = tk.Label(master=plot_options_frame,text="y max",font=slider_font,width=10,height=1)
        label.grid(row=1,column=3)


        #Length of plotted beam path
        beam_len = tk.DoubleVar()
        beam_len.set(0.15)
        beam_len_entry = tk.Entry(master=plot_options_frame,textvariable=beam_len,width=8)
        beam_len_entry.grid(row=2,column=0)
        label = tk.Label(master=plot_options_frame,text="Beam Length",font=slider_font,width=15,height=1)
        label.grid(row=3,column=0)
    
        #Initialize the plot
        generate_field_profile(config_name.get())
    
    
    
def main():
    root = tk.Tk()
    root.geometry("1800x900")
    
    app = Genray_GUI()
    root.mainloop()
    
    
    
if __name__ == "__main__":
    main()
