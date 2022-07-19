# -*- coding: utf-8 -*-
"""
Created on Tue Jun  1 16:34:00 2021

@author: kunal

This programs runs the main GUI developed in QtDesigner.

This code may be sparse on comments since I did a lot of copy-paste because I 
do not fully understand how PyQt5 works. If someone does, please feel free to 
add comments explaining some of the behaviour.
"""

import sys
import rf_gui
import numpy as np
from PyQt5 import uic
from datetime import date
from PyQt5.QtWidgets import QMainWindow,QApplication,qApp

#ui file made in QtDesigner
qtCreatorFile="CQL3D_GUI_ui.py"

#Load the .ui file into python
Ui_MainWindow,QtBaseClass=uic.loadUiType(qtCreatorFile)

class MyApp(QMainWindow, Ui_MainWindow):
    
    def __init__(self):
        
        QMainWindow.__init__(self)
        Ui_MainWindow.__init__(self)
        self.setupUi(self)
        
        #Create attributes for the NBI and RF windows
        self.window_rf=None
        self.window_nbi=None
        
        #'RF Setup' button logic
        self.pushButton_rf_setup.clicked.connect(self.run_rf_gui)
        
        #'NBI Setup' button logic
        
        #'Start Run' button logic
        self.pushButton_start_run.clicked.connect(self.get_simulation_params)
        self.pushButton_start_run.clicked.connect(self.create_cqlinput)
        
        #'Start Run and Quit' button logic
        self.pushButton_start_run_quit.clicked.connect(self.get_simulation_params)
        self.pushButton_start_run_quit.clicked.connect(self.create_cqlinput)
        self.pushButton_start_run_quit.clicked.connect(self.quit_program)
        
        #'Quit' button logic
        self.pushButton_quit.clicked.connect(self.quit_program)
        
    def get_simulation_params(self):
        """
        This function reads all the input parameters from the GUI and stores
        them in a dictionary.
        
        There are 3 dictionaries-
        1. cqlDict - All variables from the main GUI window
        2. rfDict - Data from the RF sub-GUI
        3. nbiDict - Data from the NBI sub-GUI

        Returns
        -------
        None.
        """
        
        #Initialize dictionary
        self.cqlDict={}
        
        # =====================================================================
        # Basic Plasma Parameters
        # =====================================================================
        
        #Electrons
        
        #Temperture
        self.cqlDict['eCoreTemp']=self.get_float(self.lineEdit_electron_core_temperature)
        self.cqlDict['eEdgeTemp']=self.get_float(self.lineEdit_electron_edge_temperature)    
        #Density
        self.cqlDict['eCoreDens']=self.get_float(self.lineEdit_electron_core_density)
        self.cqlDict['eEdgeDens']=self.get_float(self.lineEdit_electron_edge_density)
        
        #Ion Species 1
        
        #Species
        self.cqlDict['ion1Species']=str(self.comboBox_ion_1_species.currentText())
        #Distribution Function Tracking
        self.cqlDict['ion1Track']=self.checkBox_ion_1_dist_func.isChecked()
        #Temperture
        self.cqlDict['ion1CoreTemp']=self.get_float(self.lineEdit_ion_1_core_temperature)
        self.cqlDict['ion1EdgeTemp']=self.get_float(self.lineEdit_ion_1_edge_temperature)    
        #Density
        self.cqlDict['ion1CoreDens']=self.get_float(self.lineEdit_ion_1_core_density)
        self.cqlDict['ion1EdgeDens']=self.get_float(self.lineEdit_ion_1_edge_density)
        
        #Ion Species 2
        
        #Enable
        self.cqlDict['ion2Enable']=self.checkBox_ion_2_enable.isChecked()
        #Species
        self.cqlDict['ion2Species']=str(self.comboBox_ion_2_species.currentText())
        #Distribution Function Tracking
        self.cqlDict['ion2Track']=self.checkBox_ion_2_dist_func.isChecked()
        #Temperture
        self.cqlDict['ion2CoreTemp']=self.get_float(self.lineEdit_ion_2_core_temperature)
        self.cqlDict['ion2EdgeTemp']=self.get_float(self.lineEdit_ion_2_edge_temperature)    
        #Density
        self.cqlDict['ion2CoreDens']=self.get_float(self.lineEdit_ion_2_core_density)
        self.cqlDict['ion2EdgeDens']=self.get_float(self.lineEdit_ion_2_edge_density)
        
        #Ion Species 3
        
        #Enable
        self.cqlDict['ion3Enable']=self.checkBox_ion_3_enable.isChecked()
        #Species
        self.cqlDict['ion3Species']=str(self.comboBox_ion_3_species.currentText())
        #Distribution Function Tracking
        self.cqlDict['ion3Track']=self.checkBox_ion_3_dist_func.isChecked()
        #Temperture
        self.cqlDict['ion3CoreTemp']=self.get_float(self.lineEdit_ion_3_core_temperature)
        self.cqlDict['ion3EdgeTemp']=self.get_float(self.lineEdit_ion_3_edge_temperature)    
        #Density
        self.cqlDict['ion3CoreDens']=self.get_float(self.lineEdit_ion_3_core_density)
        self.cqlDict['ion3EdgeDens']=self.get_float(self.lineEdit_ion_3_edge_density)
        
        # =====================================================================
        # Simulation Parameters
        # =====================================================================
        
        #cqlinput mnemonic
        self.cqlDict['input_mnemonic']=str(self.lineEdit_input_mnemonic.text())
        #Output mnemonic
        self.cqlDict['output_mnemonic']=str(self.lineEdit_output_mnemonic.text())
        #Radial bins
        self.cqlDict['lrz']=self.get_int(self.lineEdit_Radial_Bins)        
        #Normalization energy
        self.cqlDict['enorm']=self.get_float(self.lineEdit_norm_energy)
        #Speed mesh points
        self.cqlDict['jx']=self.get_int(self.lineEdit_speed_mesh_points)
        #Theta mesh points
        self.cqlDict['iy']=self.get_int(self.lineEdit_theta_mesh_points)
        #Timestep 
        self.cqlDict['dtr']=self.get_float(self.lineEdit_timestep_size)
        #Number of timesteps
        self.cqlDict['nstop']=self.get_int(self.lineEdit_timestep_number)
        #Max energy of spectra
        self.cqlDict['enmax']=self.get_float(self.lineEdit_max_spectra)
        #Min energy of spectra
        self.cqlDict['enmin']=self.get_float(self.lineEdit_min_spectra)
        #Potential jump in the throat
        self.cqlDict['ephicc']=self.get_float(self.lineEdit_ephicc)
        
        # =====================================================================
        # Fusion Diagnostics
        # =====================================================================
        
        #Radial positions
        self.cqlDict['x_fus']=self.get_array(self.lineEdit_rad_pos)
        #Axial positions
        self.cqlDict['z_fus']=self.get_array(self.lineEdit_ax_pos)
        #Polar angle
        self.cqlDict['thet1_fus']=self.get_array(self.lineEdit_thet1_fus)
        #Toroidal angle
        self.cqlDict['thet2_fus']=self.get_array(self.lineEdit_thet2_fus)
        #Step size
        self.cqlDict['fds_fus']=self.get_float(self.lineEdit_fds_fus)
        
        # =====================================================================
        # RF Setup
        # =====================================================================
        
        #Data from sub-GUI
        rf_data=open('rf_data.json','r')
        self.rfDict=rf_data.read()
        rf_data.close()
        #Enable
        self.cqlDict['rfEnable']=self.checkBox_RF.isChecked()
        
        # =====================================================================
        # NBI Setup
        # =====================================================================
        
        #Data from sub-GUI
        
        #Enable
        self.cqlDict['nbiEnable']=self.checkBox_NBI.isChecked()
        
        return
    
    def create_cqlinput(self):
        """
        This function creates the cqlinput file based on input parameters from
        the GUI.
        
        It also contains most of the logic based on what checkboxes are
        enabled.

        Returns
        -------
        None.
        """
        
        # =====================================================================
        # Construct the filename
        # =====================================================================
        
        #Date
        today=self.get_date()
        #Filename
        self.filename=today+'_cqlinput_'+self.cqlDict['input_mnemonic']
        
        #Open the file
        with open(self.filename,'w') as cqlinput:
            
            # =================================================================
            # setup0
            # =================================================================
            cqlinput.write('$setup0\n')
            
            #Stop cql3d from making its own plots
            cqlinput.write('noplots=\'enabled\',\n')
            cqlinput.write('ioutput=6,\n')
            #Number of radial bins
            cqlinput.write('lrz='+str(self.cqlDict['lrz'])+',\n')
            #Output mnemonic
            cqlinput.write('mnemonic=\''+today+'_'+self.cqlDict['output_mnemonic']+'\',\n')
            #Allow cql3d to run commands like pwd 
            cqlinput.write('special_calls=\'enabled\'\n')
            
            #End setup0
            cqlinput.write('&end\n\n')
            
            # =================================================================
            # setup
            # =================================================================
            cqlinput.write('&setup\n')
            
            #Number of species
            #General and maxwellian parts of the same ion are treated as
            #seperate species in cql3d
            k=2 #e- and ion 1 maxwellian are active by default
            #Ion 1 general
            if self.cqlDict['ion1Track']==True:
                k+=1
            #Ion 2 general
            if self.cqlDict['ion2Enable']==True and self.cqlDict['ion2Track']==True:
                k+=2
            #Ion 2 maxwellian
            elif self.cqlDict['ion2Enable']==True and self.cqlDict['ion2Track']==False:
                k+=1
            #Ion 3 general
            if self.cqlDict['ion3Enable']==True and self.cqlDict['ion3Track']==True:
                k+=2
            #Ion 3 maxwellian
            elif self.cqlDict['ion3Enable']==True and self.cqlDict['ion3Track']==False:
                k+=1
                
            #Number of 'general' species
            ngen=0
            #Ion 1 general
            if self.cqlDict['ion1Track']==True:
                ngen+=1
            #Ion 2 general
            if self.cqlDict['ion2Enable']==True and self.cqlDict['ion2Track']==True:
                ngen+=1
            #Ion 3 general
            if self.cqlDict['ion3Enable']==True and self.cqlDict['ion3Track']==True:
                ngen+=1
            
            #Name and type of the species
            #kspeci(1,i) is the letter designation of the species
            #kspeci(2,i) determines if it is general or maxwellian
            speciesTracker=3 #Since electrons and ion 1 maxwellian are always active
            #Electrons
            cqlinput.write('kspeci(1,1)=\'electron\',\n')
            cqlinput.write('kspeci(2,1)=\'maxwell\',\n')
            #Ion species 1
            ion1Name=self.cqlDict['ion1Species']
            cqlinput.write('kspeci(1,2)=\''+ion1Name+'\',\n')
            cqlinput.write('kspeci(2,2)=\'maxwell\',\n')
            if self.cqlDict['ion1Track']==True:
                cqlinput.write('kspeci(1,'+str(speciesTracker)+')=\''+ion1Name+'\',\n')
                cqlinput.write('kspeci(2,'+str(speciesTracker)+')=\'general\',\n')
                speciesTracker+=1
            #Ion species 2
            ion2Name=self.cqlDict['ion2Species']
            if self.cqlDict['ion2Enable']==True:
                cqlinput.write('kspeci(1,'+str(speciesTracker)+')=\''+ion2Name+'\',\n')
                cqlinput.write('kspeci(2,'+str(speciesTracker)+')=\'maxwell\',\n')
                speciesTracker+=1
            if self.cqlDict['ion2Enable']==True and self.cqlDict['ion2Track']==True:
                cqlinput.write('kspeci(1,'+str(speciesTracker)+')=\''+ion2Name+'\',\n')
                cqlinput.write('kspeci(2,'+str(speciesTracker)+')=\'general\',\n')
                speciesTracker+=1
            #Ion species 3
            ion3Name=self.cqlDict['ion3Species']
            if self.cqlDict['ion3Enable']==True:
                cqlinput.write('kspeci(1,'+str(speciesTracker)+')=\''+ion3Name+'\',\n')
                cqlinput.write('kspeci(2,'+str(speciesTracker)+')=\'maxwell\',\n')
                speciesTracker+=1
            if self.cqlDict['ion3Enable']==True and self.cqlDict['ion3Track']==True:
                cqlinput.write('kspeci(1,'+str(speciesTracker)+')=\''+ion3Name+'\',\n')
                cqlinput.write('kspeci(2,'+str(speciesTracker)+')=\'general\',\n')
                speciesTracker+=1
            
            #Atomic number of the species
            speciesTracker=3 #Since electrons and ion 1 maxwellian are always active
            #Electrons
            cqlinput.write('bnumb(1)=-1.0,\n')
            #Ion species 1
            ion1Charge=self.get_charge(self.cqlDict['ion1Species'])
            cqlinput.write('bnumb(2)='+str(ion1Charge)+',\n')
            if self.cqlDict['ion1Track']==True:
                cqlinput.write('bnumb('+str(speciesTracker)+')='+str(ion1Charge)+',\n')
                speciesTracker+=1
            #Ion species 2
            ion2Charge=self.get_charge(self.cqlDict['ion2Species'])
            if self.cqlDict['ion2Enable']==True:
                cqlinput.write('bnumb('+str(speciesTracker)+')='+str(ion2Charge)+',\n')
                speciesTracker+=1
            if self.cqlDict['ion2Enable']==True and self.cqlDict['ion2Track']==True:
                cqlinput.write('bnumb('+str(speciesTracker)+')='+str(ion2Charge)+',\n')
                speciesTracker+=1
            #Ion species 3
            ion3Charge=self.get_charge(self.cqlDict['ion3Species'])
            if self.cqlDict['ion3Enable']==True:
                cqlinput.write('bnumb('+str(speciesTracker)+')='+str(ion3Charge)+',\n')
                speciesTracker+=1
            if self.cqlDict['ion3Enable']==True and self.cqlDict['ion3Track']==True:
                cqlinput.write('bnumb('+str(speciesTracker)+')='+str(ion3Charge)+',\n')
                speciesTracker+=1
            
            #Mass of the species
            speciesTracker=3 #Since electrons and ion 1 maxwellian are always active
            #Electrons
            cqlinput.write('fmass(1)=9.109e-28,\n')
            #Ion species 1
            ion1Mass=self.get_mass(self.cqlDict['ion1Species'])
            cqlinput.write('fmass(2)='+str(ion1Mass)+',\n')
            if self.cqlDict['ion1Track']==True:
                cqlinput.write('fmass('+str(speciesTracker)+')='+str(ion1Mass)+',\n')
                speciesTracker+=1
            #Ion species 2
            ion2Mass=self.get_mass(self.cqlDict['ion2Species'])
            if self.cqlDict['ion2Enable']==True:
                cqlinput.write('fmass('+str(speciesTracker)+')='+str(ion2Mass)+',\n')
                speciesTracker+=1
            if self.cqlDict['ion2Enable']==True and self.cqlDict['ion2Track']==True:
                cqlinput.write('fmass('+str(speciesTracker)+')='+str(ion2Mass)+',\n')
                speciesTracker+=1
            #Ion species 3
            ion3Mass=self.get_mass(self.cqlDict['ion3Species'])
            if self.cqlDict['ion3Enable']==True:
                cqlinput.write('fmass('+str(speciesTracker)+')='+str(ion3Mass)+',\n')
                speciesTracker+=1
            if self.cqlDict['ion3Enable']==True and self.cqlDict['ion3Track']==True:
                cqlinput.write('fmass('+str(speciesTracker)+')='+str(ion3Mass)+',\n')
                speciesTracker+=1
            
            #Density of the species
            speciesTracker=3 #Since electrons and ion 1 maxwellian are always active
            #Electrons
            cqlinput.write('reden(1,0)='+str(self.cqlDict['eCoreDens'])+',\n')
            cqlinput.write('reden(1,1)='+str(self.cqlDict['eEdgeDens'])+',\n')
            #Ion species 1
            cqlinput.write('reden(2,0)='+str(self.cqlDict['ion1CoreDens'])+',\n')
            cqlinput.write('reden(2,1)='+str(self.cqlDict['ion1EdgeDens'])+',\n')
            if self.cqlDict['ion1Track']==True:
                cqlinput.write('reden('+str(speciesTracker)+',0)='+str(self.cqlDict['ion1CoreDens'])+',\n')
                cqlinput.write('reden('+str(speciesTracker)+',1)='+str(self.cqlDict['ion1EdgeDens'])+',\n')
                speciesTracker+=1
            #Ion species 2
            if self.cqlDict['ion2Enable']==True:
                cqlinput.write('reden('+str(speciesTracker)+',0)='+str(self.cqlDict['ion2CoreDens'])+',\n')
                cqlinput.write('reden('+str(speciesTracker)+',1)='+str(self.cqlDict['ion2EdgeDens'])+',\n')
                speciesTracker+=1
            if self.cqlDict['ion2Enable']==True and self.cqlDict['ion2Track']==True:
                cqlinput.write('reden('+str(speciesTracker)+',0)='+str(self.cqlDict['ion2CoreDens'])+',\n')
                cqlinput.write('reden('+str(speciesTracker)+',1)='+str(self.cqlDict['ion2EdgeDens'])+',\n')
                speciesTracker+=1
            #Ion species 3
            if self.cqlDict['ion3Enable']==True:
                cqlinput.write('reden('+str(speciesTracker)+',0)='+str(self.cqlDict['ion3CoreDens'])+',\n')
                cqlinput.write('reden('+str(speciesTracker)+',1)='+str(self.cqlDict['ion3EdgeDens'])+',\n')
                speciesTracker+=1
            if self.cqlDict['ion3Enable']==True and self.cqlDict['ion3Track']==True:
                cqlinput.write('reden('+str(speciesTracker)+',0)='+str(self.cqlDict['ion3CoreDens'])+',\n')
                cqlinput.write('reden('+str(speciesTracker)+',1)='+str(self.cqlDict['ion3EdgeDens'])+',\n')
                speciesTracker+=1
            
            ##Fusion Diagnostics##
            
            #Enable fusion diagnostics
            cqlinput.write('fus_diag=\'enabled\',\n')
            #Radial positions
            x_fus=self.cqlDict['x_fus']
            #Axial positions
            z_fus=self.cqlDict['z_fus']
            #Polar angle
            thet1_fus=self.cqlDict['thet1_fus']
            #Toroidal angle
            thet2_fus=self.cqlDict['thet2_fus']
            #Fusion sightlines
            nv_fus=len(x_fus)
            #Convert numpy arrays to strings
            x_fusStr=self.nparray_to_string(x_fus)
            z_fusStr=self.nparray_to_string(z_fus)
            thet1_fusStr=self.nparray_to_string(thet1_fus)
            thet2_fusStr=self.nparray_to_string(thet2_fus)
            #Write to cqlinput
            cqlinput.write('nv_fus='+str(nv_fus)+',\n')
            cqlinput.write('x_fus='+x_fusStr+',\n')
            cqlinput.write('z_fus='+z_fusStr+',\n')
            cqlinput.write('thet1_fus='+thet1_fusStr+',\n')
            cqlinput.write('thet2_fus='+thet2_fusStr+',\n')
            cqlinput.write('fds_fus='+str(self.cqlDict['fds_fus'])+',\n')
            
            #Step size along viewing chord
            cqlinput.write('fds=0.2,\n')
            
            ##Plotting##
            
            #Normalized minimum for contour plots
            cqlinput.write('contrmin=1.0e-12,\n')
            
            #Radial bins to plot in the .ps
            cqlinput.write('irzplt(1)=1,\n')
            cqlinput.write('irzplt(2)=2,\n')
            cqlinput.write('irzplt(3)=3,\n')
            cqlinput.write('irzplt(4)=5,\n')
            cqlinput.write('irzplt(5)=7,\n')
            cqlinput.write('irzplt(6)=10,\n')
            
            ##Simulation size ##
            
            #Timestep size
            cqlinput.write('dtr='+str(self.cqlDict['dtr'])+',\n')
            
            #Number of timesteps
            cqlinput.write('nstop='+str(self.cqlDist['nstop'])+',\n')
            
            #Theta mesh points
            cqlinput.write('iy='+str(self.cqlDict['iy'])+',\n')
            
            #Speed mesh points
            cqlinput.write('jx='+str(self.cqlDict['jx'])+',\n')
            
            ##Other parameters##
            
            #Bootstrap current (disabled by default since mirrors don't have currents)
            cqlinput.write('bootst=\'disabled\',\n')
            
            #Chang Cooper differencing (helps minimize negative values in the dist. func.)
            cqlinput.write('chang=\'noneg\',\n')
            
            #Collision model (3 - fully non-linear collisions with a background maxwellian)
            cqlinput.write('colmodl=3,\n')
            
            #Relativistic calculations ('enabled'=quasi-relativistic)
            cqlinput.write('relativ=\'enabled\',\n')
            
            #Method of specifying zeff profiles (izeff='ion' because colmodl=3)
            cqlinput.write('izeff=\'ion\',\n')
            
            #Electron quasi-neutrality condition ('disabled'=quasi-neutrality is not enforced)
            cqlinput.write('qsineut=\'disabled\',\n')
            
            #Electron distribution for collisionality ('enabled'=e are locally maxwellian)
            cqlinput.write('locquas=\'enabled\',\n')
            
            #Profile shapes of the density and temperature
            #Becuase ipro.. is parabola, we use npwr and mpwr to specify density profiles along with core/edge density values
            cqlinput.write('iprone=\'parabola\',\n')
            cqlinput.write('iprote=\'parabola\',\n')
            cqlinput.write('iproti=\'parabola\',\n')
            cqlinput.write('iprozeff=\'parabola\',\n')
            
            #acoefne,acoefte are only used if ipro.. are set to 'asdex'
            cqlinput.write('acoefne=0,0,0,0,\n')
            cqlinput.write('acoefte=0,0,0,0,\n')
            
            #enein is only used if ipro.. are set to 'spline'
            cqlinput.write('enein=0,0,0,0,0,0,\n')
            
            #Toroidal electric field (absent in a mirror)
            cqlinput.write('elecfld(0)=1.0e-20,\n')
            cqlinput.write('elecfld(1)=1.0e-20,\n')
            
            #Min/max spectra energy
            cqlinput.write('enmax='+str(self.cqlDict['enmax'])+',\n')
            cqlinput.write('enmin='+str(self.cqlDict['enmin'])+',\n')
            
            #Normalization energy
            cqlinput.write('enorm='+str(self.cqlDict['enorm'])+',\n')
            
            #Electric field calculation parameter
            cqlinput.write('eoved=0.0,\n')
            
            #Potential jump in the throat
            cqlinput.write('ephicc='+str(self.cqlDict['ephicc'])+',\n')
            
            #Coulomb logarithm
            cqlinput.write('gamaset=16.0,\n')
            
            #Implicit differencing and gaussian elimination of the Fokker-Plank equation
            cqlinput.write('implct=\'enabled\',\n')
            
            #Particle loss method
            if ngen==1:
                cqlinput.write('torloss=\'energy\',\n')
            elif ngen==2:
                cqlinput.write('torloss=\'energy\',\'energy\',\n')
            elif ngen==3:
                cqlinput.write('torloss=\'energy\',\'energy\',\'energy\',\n')
            
            #Enable different fusion reactions
            cqlinput.write('isigmas(1)=1,\n') # D + T --> n + 4He
            cqlinput.write('isigmas(3)=1,\n') # D + D --> n + 3He
            cqlinput.write('isigmas(4)=1,\n') # D + D --> p + T
            
            #Enable fusion reactions to depend on the distribution function
            cqlinput.write('isigsgv1=0,\n')
            #Prevent background maxwellian from being included in the reactions
            cqlinput.write('isigsgv2=0,\n')
            
            ##Disabled/unused parameters##
            
            #kpress is enabled by default
            
            #torloss is skipped since we are not currently simulating loss of high energy particles
            #tauloss and enloss are part of torloss if it's enabled
            
            #eegy is skipped since it is used to generate plots in the .ps file
            
            #End setup
            cqlinput.write('&end\n\n')
            
            # =================================================================
            # Close the file
            # =================================================================
            cqlinput.close()
        
        return
    
    def run_cqlinput(self):
        
        return
    
    def run_rf_gui(self):
        """
        This function runs the rf_gui program to setup the RF parameters of the
        simulation.

        Returns
        -------
        None.

        """
        #Check if an instance of rf_gui is already running
        if self.window_rf is None:
            self.window_rf=rf_gui.MyApp()
            self.window_rf.show()
        else:
            self.window_rf.show()
        
        return
    
    def get_mass(self,species):
        """
        This function returns the mass of the ion species in the plasma.
        Units- Grams (g)

        Parameters
        ----------
        species : string
            Name of the ion species.

        Returns
        -------
        float
            Mass of the ion species.
        """
        
        if species=='Deuterium':
            return 3.344e-24
        elif species=='Tritium' or species=='Helium-3':
            return 5.008e-24
        elif species=='Helium-4':
            return 6.647e-24
        elif species=='Carbon':
            return 1.994e-23
        elif species=='Tungsten':
            return 3.053e-22
    
    def get_charge(self,species):
        """
        This function returns the charge of the ion species in the plasma.
        Units- Coulombs (C)

        Parameters
        ----------
        species : string
            Name of the ion species.

        Returns
        -------
        float
            Charge of the ion species.
        """
        
        if species=='Deuterium' or species=='Tritium':
            return 1.0
        elif species=='Helium-3' or species=='Helium-4':
            return 2.0
        elif species=='Carbon':
            return 6.0
        elif species=='Tungsten':
            return 74.0
    
    def get_date(self):
        """
        This function returns the current date in the format YYMMDD.

        Returns
        -------
        dateVal : string
            Current date in the format YYMMDD.
        """
        
        #Get the date in the format YYYY-MM-DD
        today=str(date.today())
        
        #Remove 1st 2 digits of the year
        dateVal=today[2:]
        
        #Remove hyphens
        dateVal=dateVal.replace('-','')
        
        return dateVal
    
    def get_int(self,inputBox):
        """
        This function returns the integer value in a given QLineEdit input
        window.
        
        Default value is 0
        
        If a char or string is entered on accident, 0 is returned.

        Parameters
        ----------
        inputBox : QLineEdit
            Input window where a float value is expected.

        Returns
        -------
        float
            float value in the QLineEdit input window.
        """
        try:
            return int(inputBox.text())
        except ValueError:
            return 0
    
    def get_float(self,inputBox):
        """
        This function returns the float value in a given QLineEdit input
        window.
        
        Default value is 0.0
        
        If a char or string is entered on accident, 0.0 is returned.

        Parameters
        ----------
        inputBox : QLineEdit
            Input window where a float value is expected.

        Returns
        -------
        float
            float value in the QLineEdit input window.
        """
        try:
            return float(inputBox.text())
        except ValueError:
            return 0.0
        
    def get_array(self,inputBox):
        """
        This function returns the array in a given QLineEdit input window.
        
        Default value is np.array([0])
        
        If the data type of the entry is not numpy.ndarray or list, 
        np.array([0]) is returned.

        Parameters
        ----------
        inputBox : QLineEdit
            Input window where a float value is expected.

        Returns
        -------
        arrayIn: np.array
            Array in the QLineEdit input window.
        """
        
        #Evaluate the string in the QLineEdit window
        try:
            arrayIn=eval(inputBox.text())
            
            #Check data type
            if type(arrayIn)==np.ndarray:
                return arrayIn
            
            elif type(arrayIn)==list:
                arrayIn=np.array(arrayIn)
                return arrayIn
            
            else:
                return np.array([0])
            
        except:
            return np.array([0])
        
    def nparray_to_string(self,dataArray):
        """
        This function converts a numpy array into a csv string.

        Parameters
        ----------
        dataArray : np.array
            numpy array to be converted to a string

        Returns
        -------
        joinedStr : str
            string with all the data of the array
        """
        
        #Convert array to list
        dataList=dataArray.tolist()
        #Convert each element in the array to a string
        strList=[]
        for ele in dataList:
            strList.append(str(ele))
        #Join the list
        joinedStr=','.join(strList)
        
        return joinedStr
        
    def quit_program(self):
        """
        This function ends the GUI.

        Returns
        -------
        None.
        """
        
        qApp.quit()
        self.hide()
        
        return

#The part that executes automatically when the program is run
if __name__=="__main__":
    
    app=QApplication(sys.argv)
    window=MyApp()
    window.show()
    sys.exit(app.exec_())