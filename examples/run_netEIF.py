# -*- coding: utf-8 -*-
"""
Created on Mon Oct  8 21:37:23 2018

@author: patri
"""

import numpy as np
import matplotlib.pyplot as plt
from models import netEIF_EImix as eif

# redefinimos el número de neuronas [Exc, Inh] en la red
eif.N_i=[75,25]
# Esta función reconstruye los vectores con parámetros
eif.InitPars()
# Calculamos matrices de conectividad aleatorias
eif.InitCM(Pee=0.2, Pei=0.6, Pie=0.4, Pii=0.4)

# Podemos usar la matriz que queramos definiendo las variables eif.CMe
# (excitatorias) y eif.CMi (inhibitorias)

# siempre recompilar la función decorada con numba
eif.LIF.recompile()

spikes,time,X_t=eif.runSim(eif.InitVars(-60),recordV=True)
  
print(eif.ParamsNet())
print(eif.ParamsNeuronByType())

#%%

# tstop=eif.tstop

if len(spikes)>0:
    spikes=np.array(spikes)
    neuron,fRates=np.unique(spikes[:,0],return_counts=True)
    fRates=fRates/(eif.tstop/1000)            
            
#%%
# Vplot=np.arange(-65,-25,0.1)
# plt.figure(2,figsize=(5,5))
# plt.clf()
# plt.plot(Vplot,F(Vplot))

Ne=eif.Ne
Ni=eif.Ni

plt.figure(1,figsize=(12,6))
plt.clf()
if len(spikes)>0:
    plt.subplot(231)
    plt.plot(spikes[:,1],spikes[:,0],'.',ms=1)
    plt.xlim((0,eif.tstop))
plt.subplot(232)
plt.plot(time[::10],X_t[::10,0,:Ne],'g')
plt.plot(time[::10],X_t[::10,0,Ne:],'r')
plt.subplot(233)
plt.plot(time[::10],X_t[::10,1,:Ne],'g')
plt.plot(time[::10],X_t[::10,1,Ne:],'r')
plt.subplot(235)
plt.plot(time[::10],X_t[::10,2,:Ne],'g')
plt.plot(time[::10],X_t[::10,2,Ne:],'r')


plt.subplot(234)
maxV=np.max(np.abs(np.c_[eif.CMe,eif.CMi]))
plt.imshow(np.c_[eif.CMe*1,eif.CMi*-1],cmap='bwr',vmin=-maxV,vmax=maxV)
plt.colorbar()

plt.tight_layout()
