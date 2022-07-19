# -*- coding: utf-8 -*-
"""
Created on Thu Jun  3 13:48:30 2021

@author: michael

This program runs the nbi setup sub-GUI developed in QtDesigner

( below comments are from rf_gui.py by kunal, same information applies here)
It is intended to be used in conjuction with cql3d_gui.py and not as a
standalone function.

This code may be sparse on comments since I did a lot of copy-paste because I 
do not fully understand how PyQt5 works. If someone does, please feel free to 
add comments explaining some of the behaviour.

"""

import sys
import json
from PyQt5 import uic
from PyQt5.QtWidgets import *

#name for UI file
qtCreatorFile = "nbisetup_gui.ui"

#Load the .ui file into python
Ui_MainWindow,QtBaseClass=uic.loadUiType(qtCreatorFile)

class MyApp(QMainWindow, Ui_MainWindow):
    def __init__(self):
        QMainWindow.__init__(self)
        Ui_MainWindow.__init__(self)
        self.setupUi(self)
        
        if self.groupBox_2.isChecked():
            print("hello")
       

def get_nbi_params(self):
    """
   This function reads all the input parameters from the GUI and saves
   them as a dictionary in a text file.

    There is an if statement in here that makes the code different for whether you have one or two beams
    
    Returns
    -------
    None.

    """ 
    #Initlaize Dictionary
    nbiDict={}
    #this if statement is the case where you have one beam.
    if self.groupBox_2.isChecked() == False:
        
        #beam shape
        nbiDict['bshape']=str(bshape_W.currentText())
        #beam height
        nbiDict['bheigh']=self.get_float(self.bheigh_W)
        #beam width
        nbiDict['bwidth']=self.get_float(self.bwidth_W)
        #Focal Length, vertical then horizontal
        nbiDict['bvfoc']=self.get_float(self.bvfoc_W)
        nbiDict['bhfoc']=self.get_float(self.bhfoc_W)
        #Divergence, vertical then horizontal
        nbiDict['bvdiv']=self.get_float(self.bvdiv_W)
        nbiDict['bhdiv']=self.get_float(self.bhdiv_W)
        #Horizontal and Vertical angle
        nbiDict['angleh']=self.get_float(self.angleh_W)
        nbiDict['anglev']=self.get_float(self.anglev_W)
        #ofset, vertical then horizontal
        nbiDict['bvofset']=self.get_float(self.bvofset_W)
        nbiDict['bhofset']=self.get_float(self.bhofset_W)
        #Length along optical axis
        nbiDict['bleni']=self.get_float(self.bleni_W)
        #Distance from source to pivot point
        nbiDict['blenp']=self.get_float(self.blenp_W)
        #Total power through apeture into torus
        nbiDict['bptor']=self.get_float(self.bptor_W)
        #Pivot position, r then z
        nbiDict['rpivot']=self.get_float(self.rpivot_W)
        nbiDict['zpivot']=self.get_float(self.zpivot_W)
        #Current
        nbiDict['bcur']=self.get_float(self.bcur_W)
        #Max particle energy in source
        nbiDict['ebkev']=self.get_float(self.ebkev_W)
        #Fraction of current 1,2,3. I need to double check this:format is wonky
        nbiDict['fbcur(1,1)']=self.get_float(self.fbcur1_W)
        nbiDict['fbcur(2,1)']=self.get_float(self.fbcur2_W)
        nbiDict['fbcur(3,1)']=self.get_float(self.fbcur3_W)
        #Number of apetures
        nbiDict['naptr']=self.get_float(self.naptr_W)
        #Apeture height, width, length, shape
        # nbiDict['aheigh']=self.get_float(self.)
        # nbiDict['alen']=self.get_float(self.)
        # nbiDict['awidth']=self.get_float(self.)
        nbiDict['ashape']=str(ashape_W.currentText())
    
#below are just functions to get values from textfields or to quit program.
    def get_int(self,inputBox):
        """
        This function returns the integer value in a given QLineEdit input
        window.
        
        Default value is 0
        
        If a char or string is entered on accident, 0 is returned.
​
        Parameters
        ----------
        inputBox : QLineEdit
            Input window where a float value is expected.
​
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
​
        Parameters
        ----------
        inputBox : QLineEdit
            Input window where a float value is expected.
​
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
​
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
        
        
        