# -*- coding: utf-8 -*-.
"""
The Huber_braun neuronal model function.

@author: porio
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import anarpy.models.netWilsonCowan as WC
import anarpy.utils.Networks as Networks

WC.tTrans=2  #transient removal (pre-simulation)
WC.tstop=20   # actual simulation
WC.G=0.05    #Connectivity strength
WC.D=0.002        #noise factor
WC.dt=0.002   # Sampling interval
# Note that WC.dtSim is the time interval for numerical solution

np.random.seed(15)
nnodes=45
WC.N=nnodes
WC.CM=Networks.distCM(nnodes,density=0.25,rnd=0.10,symmetrical=False)
# The nodes are heterogeneous
WC.P=np.random.uniform(0.35,0.5,nnodes)
# P=.4

# This line runs the simulation!! The following is analysis and plotting
Vtrace,time=WC.Sim(verbose=True)
#%%    

# These lines can be adapted to write the parameters to a text file
print(str(WC.ParamsNode()).replace(", '","\n '"))
print(str(WC.ParamsSim()).replace(", '","\n '"))
print(str(WC.ParamsNet()).replace(", '","\n '"))
       
E_t=Vtrace[:,0,:]

spec=np.abs(np.fft.fft(E_t-np.mean(E_t,0),axis=0))
freqs=np.fft.fftfreq(len(E_t),WC.dt)

analytic=signal.hilbert(E_t,axis=0)
envelope=np.abs(analytic)

FC=np.corrcoef(envelope,rowvar=False)

#%%

plt.figure(1,figsize=(10,8))
plt.clf()
plt.subplot(321)
plt.plot(time,Vtrace[:,0,::4])
plt.ylabel('E')

# plt.subplot(323)
# plt.plot(time,Vtrace[:,2,:])
# plt.xlabel('time')
# plt.ylabel('a_ie')

plt.subplot(325)
plt.plot(freqs[:1000],spec[:1000,::4])
plt.xlabel('frequency (Hz)')
plt.ylabel('abs')

plt.subplot(222)
plt.imshow(WC.CM,cmap='gray_r')
plt.title("structural connectivity")
# plt.yticks((0,5,10,15))

plt.subplot(224)
plt.imshow(FC,cmap='BrBG',vmin=-1,vmax=1)
plt.colorbar()
plt.title("Envelope correlation (FC)")
# plt.yticks((0,5,10,15))

plt.subplots_adjust(hspace=0.3)
