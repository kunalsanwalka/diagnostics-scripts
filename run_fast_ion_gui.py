# -*- coding: utf-8 -*-
"""
Created on Thu Jul 22 10:06:26 2021

@author: micha
"""

import FastIonOrbit
import numpy as np
from PyQt5 import QtCore,QtGui,QtWidgets
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
from PyQt5.QtWidgets import QMainWindow,QApplication,QFileDialog,qApp,QMessageBox
import timeit
import mpl_toolkits.mplot3d
from mpl_toolkits.mplot3d import Axes3D
from fast_ion_gui import Ui_MainWindow


#getting the field from pleiades..
def pass_field_and_run_gui(r_in,z_in,br_in,bz_in):
    
    r=r_in
    z=z_in
    br=br_in
    bz=bz_in
    
    #running gui
    import sys
    app = QtWidgets.QApplication(sys.argv)
    app.setQuitOnLastWindowClosed(True)
    window= fast_ion_gui()
    window.show()
    sys.exit(app.exec_())



class fast_ion_gui(QtWidgets.QMainWindow):
    #Gui window.
    def __init__(self,parent=None):
        super(fast_ion_gui,self).__init__(parent)
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        
        #"run orbit integrator" button logic
        self.ui.runButton.clicked.connect(self.run_fast_ion)
        
        self.ui.energyButton.clicked.connect(self.check_energy)

        self.r = np.empty(0)
        self.z = np.empty(0)
        self.br = np.empty(0)
        self.bz = np.empty(0)
        
    def run_fast_ion(self):
        #getting graph mins and maxes from text box
        xmin=int(self.ui.xmin.text())
        xmax=int(self.ui.xmax.text())
        ymin=int(self.ui.ymin.text())
        ymax=int(self.ui.ymax.text())
        zmin=int(self.ui.zmin.text())
        zmax=int(self.ui.zmax.text())
        
        #initializing ion properties.
        npoints=int(self.ui.npoints.text())
        charge=float(self.ui.chargeValue.text())*1.602e-19
        mass=float(self.ui.massValue.text())*1.660503E-27
        time=float(self.ui.time_passed.text())
        time_array=np.linspace(0,time,npoints)
        fieldType=str(self.ui.fieldOptions.currentText())
        global qm
        qm=charge/mass
        
        #getting field type from gui.
        if fieldType=="From Coilset":
            fieldType=0
        if fieldType=="Uniform in Z":
            fieldType=1
        if fieldType=="Toroidal":
            fieldType=2
        
        #initial conditions from the velocity input box
        if self.ui.velocityBox.isChecked():
            #initial position and velocity
            ion_r_v=float(self.ui.r_2.text()) #r
            ion_phi_v=float(self.ui.phi_2.text()) #phi
            ion_z_v=float(self.ui.z_2.text()) #z
            ion_vr_v=float(self.ui.vr_2.text()) #vr
            ion_vphi_v=float(self.ui.vphi_2.text()) #vphi
            ion_vz_v=float(self.ui.vz_2.text())  #vz
            
            
            
            #converting to cartesian. orbit calcualtion works better in cartesian 
            #due to divide by 0 errors happening when r=0 in cylindrical.
            yinit_v = [
                     ion_r_v*np.cos(np.radians(ion_phi_v)), #x
                     ion_r_v*np.sin(np.radians(ion_phi_v)), #y
                     ion_z_v,                             #z
                     ion_vr_v*np.cos(np.radians(ion_phi_v)),#vx
                     ion_vr_v*np.sin(np.radians(ion_phi_v)),#vy
                     ion_vz_v                             #vz
                     ]
            #creating an input dictionary for the fast ion integrator.
            inputs_v = {
                "fieldType":fieldType,
                "qm":qm,
                "npoints":npoints,
                "time_array":time_array,
                "y0":yinit_v,
                "b_r":self.br,
                "b_z":self.bz,
                "z":self.z,
                "r":self.r
                }
            
            #gathering timing data and running orbit integrator 
            time0_v = timeit.default_timer()
            print("Computing fast ion orbit...")
            global ionPath_v
            ionPath_v= FastIonOrbit.run_orbit(inputs_v)
            print("Computation finished.")
            print("Computation time: " +str(timeit.default_timer()-time0_v))
            #making the path global so we don't have to recalculate
            global path_v
            path_v=ionPath_v.y
            
            #getting ready to plot
            x_plot_v=path_v[0]
            y_plot_v=path_v[1]
            z_plot_v=path_v[2]
            
            fig_v=plt.figure()
            
            ax_v.plot3D(z_plot_v,y_plot_v,x_plot_v)
            ax_v.view_init(0,90)
            
            ax_v.set_xlabel("X")
            ax_v.set_ylabel("Y")
            ax_v.set_zlabel("Z")
            
            if self.ui.graphSizing.isChecked():
                ax_v.set_xlim(xmin,xmax)
                ax_v.set_ylim(ymin,ymax)
                ax_v.set_zlim(zmin,zmax)
                
            ax_v.set_title("Fast Ion Path")
            ax_v.plot(x_plot_v,y_plot_v,z_plot_v)
            ax_v.show()
            
            
        #inital conditions from energy input box
        if self.ui.energyBox.isChecked():
            #initial position and velocity
            ion_r_e=float(self.ui.r.text()) #r
            ion_phi_e=float(self.ui.phi.text()) #phi
            ion_z_e=float(self.ui.z.text()) #z
            energy=float(self.ui.energy.text())
            ion_energy=float(self.ui.energy.text())*1.6022e-19
            speed = np.sqrt(2*ion_energy/mass)
            
            
            #converting to cartesian. orbit calcualtion works better in cartesian 
            #due to divide by 0 errors happening when r=0 in cylindrical.
            yinit_e =[
                    ion_r_e*np.cos(np.radians(ion_phi_e)),#x pos
                    ion_r_e*np.cos(np.radians(ion_phi_e)),#y pos
                    ion_z_e,                              #z pos
                    speed*np.sin(np.radians(ion_phi_e)),#x comp velocity
                    0,                                  #y comp velocity. in this coordinate system, this is vertical.
                    speed*np.cos(np.radians(ion_phi_e)),#z comp velocity
                ]
            distance_followed = float(self.ui.distanceFollowed.text())
            if distance_followed != 0:
                time=distance_followed/speed
                time_array=np.linspace(0,time,npoints)
            #creating an input dictionary for the fast ion integrator.
            inputs_e = {
                "fieldType":fieldType,
                "qm":qm,
                "npoints":npoints,
                "time_array":time_array,
                "y0":yinit_e,
                "b_r":self.br,
                "b_z":self.bz,
                "z":self.z,
                "r":self.r
                }
            global ionPath_e
            #running fast ion orbit integrator and gathering run time data
            time0_e = timeit.default_timer()
            print("Running orbit integrator..")
            ionPath_e = FastIonOrbit.run_orbit(inputs_e)
            print("Computation finished")
            print("Computation time: " +str(timeit.default_timer()-time0_e))
            #making the path global so we don't have to recalculate
            global path_e
            
            path_e=ionPath_e.y
            
            #Defining x,y,z values that we will plot
            x_plot_e=path_e[0]
            y_plot_e=path_e[1]
            z_plot_e=path_e[2]
            
            #initializing plot, setting view angle
            fig_e=plt.figure()
            ax_e=plt.axes(projection='3d')
            #dividing by 4 is arbitrary.
            axmin_e=min(z_plot_e)/10
            axmax_e=max(z_plot_e)/10
            
            ax_e.set_xlabel("Z")
            ax_e.set_ylabel("Y")
            ax_e.set_zlabel("X")
            
            if self.ui.graphSizing.isChecked():
                ax_e.set_xlim(xmin,xmax)
                ax_e.set_ylim(ymin,ymax)
                ax_e.set_zlim(zmin,zmax)
            else: 
#                ax_e.set_xlim(axmin_e,axmax_e)
                ax_e.set_ylim(axmin_e,axmax_e)
                ax_e.set_zlim(axmin_e,axmax_e)  
            ax_e.set_title("Fast Ion Path, energy :" +str(energy) + " eV")
            ax_e.plot(z_plot_e,y_plot_e,x_plot_e,lw=1)
            plt.show()
    def check_energy(self):
        try:
            energyPlot_e=FastIonOrbit.orbitEnergy(ionPath_e,qm)
            energyPlot_e=energyPlot_e/np.average(energyPlot_e)
            fig3,ax3 = plt.subplots()
            ax3.plot(energyPlot_e)
            ax3.set_xlabel("Point number in array.")
            ax3.set_ylabel("Normalized Energy: E/E_avg")
            ax3.set_title("Energy conservation curve, energy initial conditions")
            plt.show()
        except:
            print("Orbit not calculated with initial energy conditions")
        try:
            energyPlot_v = FastIonOrbit.orbitEnergy(ionPath_v,qm)
            energyPlot_v = energyPlot_v/np.average(energyPlot_v)
            fig4,ax4=plt.subplots()
            ax4.plot(energyPlot_v)
            ax4.set_xlabel("Point number in array.")
            ax4.set_ylabel("Normalized Energy: E/E_avg")
            ax4.set_title("Energy conservation curve,initial velocity conditions")
            plt.show()  
        except:
            print("Orbit not calculated with initial velocity conditions")
            
