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
import json
import rf_gui
import nbisetup_gui
import numpy as np
from PyQt5 import uic
from datetime import date
from PyQt5.QtWidgets import QMainWindow,QApplication,QFileDialog,qApp

#ui file made in QtDesigner
qtCreatorFile="CQL3D_GUI_michael.ui"

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
        
        self.pushButton_rf_setup.clicked.connect(self.showrf)
        
        #'NBI Setup' button logic
        self.pushButton_nbi_setup.clicked.connect(self.shownbi)
        
        #'Populate GUI' button logic
        self.pushButton_populate_gui.clicked.connect(self.populate_GUI)
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
    
    def populate_GUI(self):
        """
        This function pre populates fields from a CQL input file. 
        Currently only for nbi setup, but same methods are used to populate 
        fields in the whole GUI.

        Returns
        -------
        None.

        """
        
        fileName = QFileDialog.getOpenFileName(self,'Choose CQL input file','c://')[0]
        # fileName ="C:/Users/micha/Downloads/210524_cqlinput_DT_90deg_flat"
        cqlData = self.cql_to_dict(fileName)
        
        
        if self.window_nbi is None:
            self.run_nbi_gui()
        if self.window_rf is None:
            self.run_rf_gui()

        
        #Filling in the Main GUI
        
        #These lines find out which species is the electron. Set to -1 initially so we can know if we have an electron species. 
        electronIndex=0
        lenCondition=False
        #This is so we know how many species there are. 
        speciesCounter=1
        speciesNames = []
        while lenCondition==False:
            try:
                speciesName=cqlData["kspeci(1,"+str(speciesCounter)+")"][0]
                speciesNames.append(speciesName)
                speciesCounter+=1
            except KeyError:
                print(speciesCounter)
                lenCondition=True
        ionIndices = list(range(1,speciesCounter))
        print(ionIndices)
        print(speciesNames)
        
        for name in speciesNames:
            if name[1] =='e':
                electronIndex=speciesNames.index(name)

        #filling electron species first, since we have the index
        self.lineEdit_electron_core_temperature.setText(cqlData['temp('+str(electronIndex+1)+',0)'][0])
        #print("Core temp is: " + cqlData['temp('+str(electronIndex+1)+',0)'][0])
        self.lineEdit_electron_edge_temperature.setText(cqlData['temp('+str(electronIndex+1)+',1)'][0])
        #print("Edge temp is: " +cqlData['temp('+str(electronIndex+1)+',1)'][0])
        self.lineEdit_electron_core_density.setText(cqlData['reden('+str(electronIndex+1)+',0)'][0])
        #print("Core Density is: " +cqlData['reden('+str(electronIndex+1)+',0)'][0] )
        self.lineEdit_electron_edge_density.setText(cqlData['reden('+str(electronIndex+1)+',1)'][0])
        #print("Edge density is: " +cqlData['reden('+str(electronIndex+1)+',1)'][0])
        
        #remove so we 
        del ionIndices[electronIndex]
        currentIndex = ionIndices[0]
        
        #ion 1 core temp
        self.lineEdit_ion_1_core_temperature.setText(cqlData['temp('+str(currentIndex)+',0)'][0])
        #ion 1 edge temp
        self.lineEdit_ion_1_edge_temperature.setText( cqlData['temp('+str(currentIndex)+',1)'][0])
        #ion 1 core density
        self.lineEdit_ion_1_core_density.setText(cqlData['reden('+str(currentIndex)+',0)'][0])
        #ion 1 edge density
        self.lineEdit_ion_1_edge_density.setText(cqlData['reden('+str(currentIndex)+',1)'][0])
            
        #move to ion 2
        currentIndex= ionIndices[1]
        #ion 2 core temp
        self.lineEdit_ion_2_core_temperature.setText(cqlData['temp('+str(currentIndex)+',0)'][0])
        #ion 2 edge temp
        self.lineEdit_ion_2_edge_temperature.setText(cqlData['temp('+str(currentIndex)+',1)'][0])
        #ion 1 core density
        self.lineEdit_ion_2_core_density.setText(cqlData['reden('+str(currentIndex)+',0)'][0])
        #ion 1 edge density
        self.lineEdit_ion_2_edge_density.setText(cqlData['reden('+str(currentIndex)+',1)'][0])
        
        #move to ion 3
        currentIndex=ionIndices[2]
        
        #ion 3 core temp
        self.lineEdit_ion_3_core_temperature.setText(cqlData['temp('+str(currentIndex)+',0)'][0])
        #ion 3 edge temp
        self.lineEdit_ion_3_edge_temperature.setText(cqlData['temp('+str(currentIndex)+',1)'][0])
        #ion 3 core density
        self.lineEdit_ion_3_core_density.setText(cqlData['reden('+str(currentIndex)+',0)'][0])
        #ion 3 edge density
        self.lineEdit_ion_3_edge_density.setText(cqlData['reden('+str(currentIndex)+',1)'][0])

        #input and output mnemonic
        self.lineEdit_input_mnemonic.setText(fileName)
        self.lineEdit_output_mnemonic.setText(cqlData['mnemonic'][0])
        
        #will comment
        self.lineEdit_Radial_Bins.setText(cqlData['lrz'][0])
        self.lineEdit_norm_energy.setText(cqlData['enorm'][0])
        self.lineEdit_speed_mesh_points.setText(cqlData['jx'][0])
        self.lineEdit_theta_mesh_points.setText(cqlData['iy'][0])
        self.lineEdit_timestep_size.setText(cqlData['dtr'][0])
        self.lineEdit_timestep_number.setText(cqlData['nstop'][0])
        self.lineEdit_max_spectra.setText(cqlData['enmax'][0])
        self.lineEdit_min_spectra.setText(cqlData['enmin'][0])
        self.lineEdit_ephicc.setText(cqlData['ephicc'][0])
        self.lineEdit_rad_pos.setText(cqlData['x_fus'][0])
        self.lineEdit_ax_pos.setText(cqlData['z_fus'][0])
        self.lineEdit_thet1_fus.setText(cqlData['thet1_fus'][0])
        self.lineEdit_thet2_fus.setText(cqlData['thet2_fus'][0])
        self.lineEdit_fds_fus.setText(cqlData['fds_fus'][0])




        #I don't know a good way to input these combo boxes. Maybe we should change the 
        #GUI. 
        """
        #self.comboBox_ion_1_species.setText(cqlData['ion1Species'][0])
        #self.checkBox_ion_1_dist_func.setText(cqlData['ion1Track'][0])
       
        #self.checkBox_ion_2_enable.setText(cqlData['ion2Enable'][0])
        #self.comboBox_ion_2_species.setText(cqlData['ion2Species'][0])
        #self.checkBox_ion_2_dist_func.setText(cqlData['ion2Track'][0])
        
        #self.checkBox_ion_3_enable.setText(cqlData['ion3Enable'][0])
        #self.comboBox_ion_3_species.setText(cqlData['ion3Species'][0])
        #self.checkBox_ion_3_dist_func.setText(cqlData['ion3Track'][0])
        """        
        
        
        
        #Filling in the NBI setup sub GUI
        self.window_nbi.bheigh_W.setText(cqlData['bheigh(1)'][0])
        self.window_nbi.bheigh_W_4.setText(cqlData['bheigh(1)'][1])
        self.window_nbi.bwidth_W.setText(cqlData['bwidth(1)'][0])
        self.window_nbi.bwidth_W_4.setText(cqlData['bwidth(1)'][1])
        self.window_nbi.bvfoc_W.setText(cqlData['bvfoc(1)'][0])
        self.window_nbi.bvfoc_W_4.setText(cqlData['bvfoc(1)'][1])
        self.window_nbi.bhfoc_W.setText(cqlData['bhfoc(1)'][0])
        self.window_nbi.bhfoc_W_4.setText(cqlData['bhfoc(1)'][1])
        self.window_nbi.bvdiv_W.setText(cqlData['bvdiv(1)'][0])
        self.window_nbi.bvdiv_W_4.setText(cqlData['bvdiv(1)'][1])
        self.window_nbi.bhdiv_W.setText(cqlData['bhdiv(1)'][0])
        self.window_nbi.bhdiv_W_4.setText(cqlData['bhdiv(1)'][1])
        self.window_nbi.angleh_W.setText(cqlData['angleh(1)'][0])
        self.window_nbi.angleh_W_4.setText(cqlData['angleh(1)'][1])
        self.window_nbi.anglev_W.setText(cqlData['anglev(1)'][0])
        self.window_nbi.anglev_W_4.setText(cqlData['anglev(1)'][1])
        self.window_nbi.bvofset_W.setText(cqlData['bvofset(1)'][0])
        self.window_nbi.bvofset_W_4.setText(cqlData['bvofset(1)'][1])
        self.window_nbi.bhofset_W.setText(cqlData['bhofset(1)'][0])
        self.window_nbi.bhofset_W_4.setText(cqlData['bhofset(1)'][1])
        self.window_nbi.bleni_W.setText(cqlData['bleni(1)'][0])
        self.window_nbi.bleni_W_4.setText(cqlData['bleni(1)'][1])
        self.window_nbi.blenp_W.setText(cqlData['blenp(1)'][0])
        self.window_nbi.blenp_W_4.setText(cqlData['blenp(1)'][1])
        self.window_nbi.bptor_W.setText(cqlData['bptor(1)'][0])
        self.window_nbi.bptor_W_4.setText(cqlData['bptor(1)'][1])   
        self.window_nbi.rpivot_W.setText(cqlData['rpivot(1)'][0])
        self.window_nbi.zpivot_W.setText(cqlData['zpivot(1)'][0])
        self.window_nbi.bcur_W.setText(cqlData['bcur(1)'][0])
        self.window_nbi.ebkev_W.setText(cqlData['ebkev(1)'][0])
        self.window_nbi.fbcur1_W.setText(cqlData['fbcur(1,1)'][0])
        self.window_nbi.fbcur2_W.setText(cqlData['fbcur(2,1)'][0])
        self.window_nbi.fbcur3_W.setText(cqlData['fbcur(3,1)'][0])
        self.window_nbi.naptr_W.setText(cqlData['naptr'][0])
        self.window_nbi.aheigh_W.setText(cqlData['aheigh(1,1)'][0])
        self.window_nbi.awidth_W.setText(cqlData['awidth(1,1)'][0])
        self.window_nbi.alen_W.setText(cqlData['alen(1,1)'][0])      
        self.window_nbi.rpivot_W_4.setText(cqlData['rpivot(1)'][1])
        self.window_nbi.zpivot_W_4.setText(cqlData['zpivot(1)'][1])
        self.window_nbi.bcur_W_4.setText(cqlData['bcur(1)'][1])
        self.window_nbi.ebkev_W_4.setText(cqlData['ebkev(1)'][1])
        self.window_nbi.fbcur1_W_4.setText(cqlData['fbcur(1,1)'][1])
        self.window_nbi.fbcur2_W_4.setText(cqlData['fbcur(2,1)'][1])
        self.window_nbi.fbcur3_W_4.setText(cqlData['fbcur(3,1)'][1])
        self.window_nbi.naptr_W_4.setText(cqlData['naptr'][0])
        self.window_nbi.aheigh_W_4.setText(cqlData['aheigh(1,1)'][1])
        self.window_nbi.awidth_W_4.setText(cqlData['awidth(1,1)'][1])
        self.window_nbi.alen_W_4.setText(cqlData['alen(1,1)'][1])
        self.window_nbi.nfrplt_W.setText(cqlData['nfrplt'][0])
        self.window_nbi.nimp_W.setText(cqlData['nimp'][0])
        
        #filling in the rf sub gui
        
        
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
       #     self.window_rf.show()
        #else:
        #    self.window_rf.show()
        
        return
    
    def showrf(self):
        '''
        This function makes the RF GUI visible if shown in the background, and
        opens it if not already open in the background. These functions are 
        seperated so we can populate all 3 GUI windows at once without them
        all being visiable

        Returns
        -------
        None.

        '''
        if self.window_rf is None:
            self.run_rf_gui()
            self.window_rf.show()
        else:
            self.window_rf.show
    def run_nbi_gui(self):
        if self.window_nbi is None:
            self.window_nbi=nbisetup_gui.MyApp()
    def shownbi(self):
        if self.window_nbi is None:
            self.run_nbi_gui()
            self.window_nbi.show()
        else:
            self.window_nbi.show()
            
    
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
    def cql_to_dict(self,string):
        #puts each line from file into an array
        with open(string) as f:
            inputLines=f.readlines()
        
        #initialzing intermediate arrays
        intermediateLines1 = []
        intermediateLines2 = []
        intermediateLines3=[]
        finalLines = []
        #This is the first pass through. Removes comments and the \n after each line.
        for line in inputLines:
            #Remove Comments with a "!"
            line = line[:line.find("!")]
            #Splits the line into a part before and after the \n
            line=line.split('\n')
            #Grabs only the part of the line before \n
            line = line[0].strip()
            #Moves the partially formatted lines into the first intermediate array
            intermediateLines1.append(line)
        
        
        #This for loop seperates lines with two values into two lines with one value each
        for line in intermediateLines1:
            #changes only lines with two equals signs
            if line.count("=")==2:
                #Lines with two strings are dealt with in this if statement
                if line[(line.find("=")+1)]=="'":
                    #this makes a new line with everything after the first value
                    line2=line[(line.find("'",(line.find("'")+1))+1):]
                    #This is what remains after we do the first part
                    line=line[:line.find("'",(line.find("'")+1))+1]
                    #adds these lines to a second intermediate array.
                    intermediateLines2.append(line)
                    intermediateLines2.append(line2)
                #Lines with two floats are dealth with in this statement
                else:
                    #splits line by comma after equals sign
                    line1=line[:line.find(",",line.find("="))]
                    line2=line.replace(line1+",","")
                    if " " in line1:
                        line11=line1[line1.find(" "):].strip()
                        line12=line1[:line1.find(" ")].strip()
                        intermediateLines2.append(line11)
                        intermediateLines2.append(line12)
                    else:
                        intermediateLines2.append(line1)
                    if " " in line2:
                        line21=line2[line2.find(" "):].strip()
                        line22=line2[:line2.find(" ")].strip()
                        intermediateLines2.append(line21)
                        intermediateLines2.append(line22)
                    else:
                        intermediateLines2.append(line2)
            else:
                #moves all other lines to the second array
                intermediateLines2.append(line)
                #end of loop
        
        #removes all blank lines accumulated
        while "" in intermediateLines2:
            intermediateLines2.remove("")
        
        
        #this for loop deals with the parameters whose values span multiple lines
        
        #The idea: we look through the loop and "hold on" to each line with an equals
        #sign with the lastLine string. if there is an equals sign then we pass the 
        #line to the next array. if not, we add the line to the last line that had an 
        #equals sign. This will keep happening until we run into an equals sign.
        
        lastLine=""
        for line in intermediateLines2:
            if "=" not in line:
                lastLine=lastLine+ line
            else:
                intermediateLines3.append(lastLine)
                lastLine=line
              
        #the previous for loop doesn't grab the last value. while loop just gets the
        #last line with a value.
        endBool = False
        i = -1
        while endBool == False:
            if "=" not in intermediateLines2[i]:
                i=i-1
            else:
                line=intermediateLines2[i]
                line= line.strip()
                line=line[:line.find("!")]
                intermediateLines3.append(line)
                endBool=True
        
        #this line removes all remaining comments in the file.
        for line in intermediateLines3:
            line = line.strip()
            if "&" in line:
                line = line[:line.find("&")]
            if "$" in line:
                line = line[:line.find("$")]
            finalLines.append(line)
        
        #removes blank lines again
        while "" in finalLines:
            finalLines.remove("")
            
        
        cqlData = {} #initialize dictionary
        
        #this loop converts values to dictionary easily read by JSON
        for line in finalLines:
            line = line.split("=")
            line[1]=line[1].split(",")
            value = []
            for l in line[1]:
                value.append(l.strip())
            valueName=line[0].strip()
            cqlData[valueName]=value
        return cqlData
    # def find_species(self,string):
    #     if string[0]=="d" or string[0]=="D" or string[1]=="d" or string[1]=="D":
    #         return 'Deuterium'
    #     elif string[0]=="t" or string[0]=="T" or string[1]=="t" or string[1]=="T":
    #         return "Tritium"
    #     elif string[0]=="d" or string[0]=="D" or string[1]=="d" or string[1]=="D":
            
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