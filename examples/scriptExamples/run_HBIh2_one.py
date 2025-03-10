# -*- coding: utf-8 -*-.
"""
The Huber_braun neuronal model function.

@author: porio
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import time as TM
import anarpy.models.netHBIh2 as HB
import anarpy.utils.Networks as Networks

Area = 15000 # um^2

print("Area = %g"%Area)
HB.Area=Area

#Seteo de variables gsd,gsr y numero de nodos
nnodes=35
HB.nnodes=nnodes
HB.CM=Networks.distCM(nnodes,density=0.1,rnd=0.1)

HB.gsd=np.random.uniform(0.18,0.23,size=nnodes)
HB.gsr=np.random.uniform(0.2,0.26,size=nnodes)
# HB.gh=np.random.uniform(0.4,0.6,size=nnodes)

HB.adaptTime=2000
HB.adaptInt=0.05

HB.Ggj=0.6 #  coupling conductance

HB.runTime=10000
HB.runInt=0.025  # intervalo para resolver la SDE
sampling=0.25
HB.sampling=sampling  # intervalo para guardar datos
cutfreq=50 #cut frequency for low pass
# parametros del filtro
b,a=signal.bessel(4,cutfreq*2*sampling/1000,btype='low')

#not needed anymore because now it occurs inside the Sim() function
# HB.HyB.recompile()
# HB.HyBdet.recompile()

t0=TM.time()

# SIMULATION

if Area==0:
    Y_t,Tplot=HB.Sim()
else:    
    Y_t,Tplot=HB.Sim(Stoch=True,verbose=True)


CPUtime=TM.time()-t0
print(TM.ctime(),CPUtime,'simulation ready')

# Filtering and sub-samplig
b,a=signal.bessel(4,cutfreq*2*sampling/1000,btype='low')
decimate=20
Vfilt=np.array([signal.filtfilt(b,a,Y)[::decimate] for Y in Y_t.T]).T

# These lines can be adapted to write the parameters to a text file
print(str(HB.ParamsNode()).replace(", '","\n '"))
print(str(HB.ParamsSim()).replace(", '","\n '"))
print(str(HB.ParamsNet()).replace(", '","\n '"))
       
LFP=np.mean(Vfilt,-1)        

# hacemos FFT para buscar la freq maxima
spec=np.abs(np.fft.fft(LFP-np.mean(LFP)))
freqs=np.fft.fftfreq(len(LFP),sampling*decimate/1000)

#%%
plt.figure(1,figsize=(12,8))
plt.clf()
subpl = [plt.subplot(5,3,i) for i in range(1,16)]

plot_i=int(5000/(sampling))  #ms
plot_e=int(8000/(sampling)) #ms
#        Tplot=np.arange(0,HB.runTime,sampling)

for volt,ax in zip(Y_t.T[:30:2],subpl):
    ax.plot(Tplot[plot_i:plot_e],volt[plot_i:plot_e])

plot_i=int(5000/(sampling*decimate))  #ms
plot_e=int(8000/(sampling*decimate)) #ms
Tplot2=np.arange(0,HB.runTime,sampling*decimate)

for volt,ax in zip(Vfilt.T[:30:2],subpl):
    ax.plot(Tplot2[plot_i:plot_e],volt[plot_i:plot_e])
    
    
plt.figure(2,figsize=(12,5))    
plt.subplot2grid((2,3),(0,1),colspan=2)
plt.plot(Tplot[::decimate],LFP)

plt.subplot2grid((2,3),(1,1),colspan=2)
plt.plot(freqs[:len(spec)//4],spec[:len(spec)//4])

plt.subplot2grid((2,3),(0,0),rowspan=2)
plt.imshow(HB.CM,cmap='gray_r')


#%%
