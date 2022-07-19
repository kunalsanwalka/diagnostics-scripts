# -*- coding: utf-8 -*-
"""
Created on Thu Nov  4 18:34:43 2021

@author: kunal
"""

import numpy as np
import scipy.constants as const
import scipy.special as sp
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 22})

#Radius
r=0.55 #m

#Magnetic field strength
bMag=1.7 #T

#k_perp
kPerp=2*np.pi/r
# kPerp=0.55

#Perpendicular energy array (keV)
ePerpArr=np.linspace(0,100,5000) #keV
#Convert to joules
eJoulesArr=ePerpArr*const.e*1e3 #Joules

#Larmor radius
rhoLArr=np.sqrt(2*2*const.m_p*eJoulesArr)/(const.e*bMag)

#Bessel function argument
besselArgArr=kPerp*rhoLArr

#Bessel functions
j0Arr=[]
j1Arr=[]
j2Arr=[]
j3Arr=[]

for i in range(len(besselArgArr)):
    
    j0Arr.append(sp.jv(0,besselArgArr[i]))
    j1Arr.append(sp.jv(1,besselArgArr[i]))
    j2Arr.append(sp.jv(2,besselArgArr[i]))
    j3Arr.append(sp.jv(3,besselArgArr[i]))
    
j0Arr=np.array(j0Arr)
j1Arr=np.array(j1Arr)
j2Arr=np.array(j2Arr)
j3Arr=np.array(j3Arr)

#Sum of the 1st 4 bessel functions
jSumArr=j1Arr+j2Arr+j3Arr

#Max value of J_1
j1Max=np.max(j1Arr)
#Location of max value
j1MaxLoc=np.argmax(j1Arr)
#E_perp at max value
ePerpMaxJ1=ePerpArr[j1MaxLoc]

#Max value of J_sum
jSumMax=np.max(jSumArr)
#Location of max value
jSumMaxLoc=np.argmax(jSumArr)
#E_perp at max value
ePerpMaxJMax=ePerpArr[jSumMaxLoc]

fig,ax=plt.subplots(figsize=(12,8))

ax.plot([0,ePerpArr[-1]],[0,0],color='black',linewidth=3)
ax.plot(ePerpArr,j0Arr,label=r'J$_0$',color='blue')
ax.plot(ePerpArr,j1Arr,label=r'J$_1$',color='red')
ax.plot(ePerpArr,j2Arr,label=r'J$_2$',color='green')
ax.plot(ePerpArr,j3Arr,label=r'J$_3$',color='purple')
# ax.plot(ePerpArr,jSumArr,label=r'J$_{sum}$',color='orange',linewidth=2)
ax.plot(ePerpArr,j1Arr+j3Arr,label=r'J$_1$+J$_3$')

#Maximum value of J_1
ax.scatter(ePerpMaxJ1,j1Max,color='red')
ax.text(ePerpMaxJ1-6,j1Max+0.03,r'E$_{\perp}$='+str(np.round(ePerpMaxJ1,2)))

#Maximum value of J_sum
ax.scatter(ePerpMaxJMax,jSumMax,color='orange')
ax.text(ePerpMaxJMax-7,jSumMax-0.14,r'E$_{\perp}$='+str(np.round(ePerpMaxJMax,2)))

ax.set_ylabel('f(x)')
ax.set_xlabel(r'E$_{\perp}$ [keV]')
ax.set_title('B='+str(bMag)+'T, r='+str(r)+'m')

ax.set_xlim(0,ePerpArr[-1])

plt.legend()
plt.grid()
plt.show()
plt.savefig('C:/Users/kunal/OneDrive - UW-Madison/WHAM/Plots/Bessel_functions_r_055.pdf')