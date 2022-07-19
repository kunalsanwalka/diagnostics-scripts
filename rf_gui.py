# -*- coding: utf-8 -*-
"""
Created on Thu Jun  3 15:38:35 2021

@author: kunal

This programs runs the RF sub-GUI developed in QtDesigner.

It is intended to be used in conjuction with cql3d_gui.py and not as a
standalone function.

This code may be sparse on comments since I did a lot of copy-paste because I 
do not fully understand how PyQt5 works. If someone does, please feel free to 
add comments explaining some of the behaviour.
"""

import sys
import json
from PyQt5 import uic
from PyQt5.QtWidgets import QMainWindow,QApplication,qApp

#ui file made in QtDesigner
qtCreatorFile="RF_GUI.ui"

#Load the .ui file into python
Ui_MainWindow,QtBaseClass=uic.loadUiType(qtCreatorFile)

class MyApp(QMainWindow, Ui_MainWindow):
    
    def __init__(self):
        
        QMainWindow.__init__(self)
        Ui_MainWindow.__init__(self)
        self.setupUi(self)
        
        #'Save Data' button logic
        self.pushButton_save_data.clicked.connect(self.get_rf_params)
        
        #'Save Data and Quit' button logic
        self.pushButton_save_data_quit.clicked.connect(self.get_rf_params)
        self.pushButton_save_data_quit.clicked.connect(self.quit_program)
        
        #'Quit' button logic
        self.pushButton_quit.clicked.connect(self.quit_program)
    
    def get_rf_params(self):
        """
        This function reads all the input parameters from the GUI and saves
        them as a dictionary in a text file.

        Returns
        -------
        None.
        """
        
        #Initialize dictionary
        rfDict={}
        
        #Ion species 1
        
        #Genray filepath
        rfDict['rffile1']=str(self.lineEdit_eqdsk_filepath.text())
        #Powerscale
        rfDict['pwrscale1']=self.get_float(self.lineEdit_powerscale)
        #1st damping harmonic
        rfDict['nharm11']=self.get_int(self.lineEdit_1st_harm)
        #Number of harmonics
        rfDict['nharms1']=self.get_int(self.lineEdit_num_harms)
        
        #Ion species 2
        
        #Genray filepath
        rfDict['rffile2']=str(self.lineEdit_eqdsk_filepath_2.text())
        #Powerscale
        rfDict['pwrscale2']=self.get_float(self.lineEdit_powerscale_2)
        #1st damping harmonic
        rfDict['nharm12']=self.get_int(self.lineEdit_1st_harm_2)
        #Number of harmonics
        rfDict['nharms2']=self.get_int(self.lineEdit_num_harms_2)
        
        #Ion species 3
        
        #Genray filepath
        rfDict['rffile3']=str(self.lineEdit_eqdsk_filepath_3.text())
        #Powerscale
        rfDict['pwrscale3']=self.get_float(self.lineEdit_powerscale_3)
        #1st damping harmonic
        rfDict['nharm13']=self.get_int(self.lineEdit_1st_harm_3)
        #Number of harmonics
        rfDict['nharms3']=self.get_int(self.lineEdit_num_harms_3)
        
        #Write to a json file
        rf_data=open('rf_data.json','w')
        json.dump(rfDict,rf_data)
        rf_data.close()
        
        return
    
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
        
    def quit_program(self):
        """
        This function ends the GUI.

        Returns
        -------
        None.
        """
        
        #Only enable if testing gui independently
        # qApp.quit()
        
        self.hide()
        
        return
        
#The part that executes automatically when the program is run
if __name__=="__main__":
    app=QApplication(sys.argv)
    window=MyApp()
    window.show()
    sys.exit(app.exec_())