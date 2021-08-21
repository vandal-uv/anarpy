# -*- coding: utf-8 -*-
"""
Created on Fri Jul 12 11:07:52 2019

@author: porio
"""
#import matplotlib
#matplotlib.use('Agg')

import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from utils import wavelets
from utils import Networks

#import the model
#explore its variables using ParamsNode(), ParamsSim(), etc
from models import WangBuszakiNetworkE_I_Ext_GJ2 as wbn

# (re)define the number of neurons

wbn.Ne=600
wbn.Ni=150

wbn.Trun=2000
wbn.equil=1000
wbn.mGsynI=200

#%%Electrical Connectivity Matrix for Gap Junctions
N=wbn.Ne + wbn.Ni
Ni=wbn.Ni
Ne= wbn.Ne

CM2=np.ones((Ni,Ni))   # all-to-all

CM2=np.random.binomial(1,0.08,(Ni,Ni))
CM2[np.triu_indices(Ni,0)]=0
CM2+=CM2.T

Ggj=0.03  #Gap junction connectivity strength (conductance)
#CMgj[Ne:,Ne:]=CM2 * Ggj

wbn.CMelec = CM2 * Ggj

#%% Chemical connectivity matrix
wbn.Pi=0.3
wbn.iRate=7.5

# generate a connectivity matrix using the included random algorithm
np.random.seed(12)

Pee=wbn.Pe
Pie=wbn.Pe  # E to I
Pii=wbn.Pi
Pei=wbn.Pi # I to E

aleat=0.08
EEMat=Networks.distCM(Ne,P=Pee,rnd=aleat,symmetrical=False)#,directed=True)
IEMat=Networks.distCM(Ni,Ne,P=Pie,rnd=aleat,symmetrical=False)
CMe=np.r_[EEMat,IEMat]  # It's not the same as wbn.CMe

IIMat=Networks.distCM(Ni,P=Pii,rnd=aleat,symmetrical=False)
EIMat=Networks.distCM(Ne,Ni,P=Pei,rnd=aleat,symmetrical=False)

CMi=np.r_[EIMat,IIMat]  ## It's not the same as wbn.CMi

wbn.genRandomCM(AdjMe=CMe,AdjMi=CMi)
CMchem = np.c_[wbn.CMe,-wbn.CMi]
# if you want, you can generate your own CM,
# in that case do not use genRandomCM()
# wbn uses CMe and CMi, you have to give them separately

wbn.WB_network.recompile()
#spikes=wbn.runSim()
#spikes,LFP,Time=wbn.runSim(output='LFP')
spikes,V_t,Time=wbn.runSim(output='allV')  #full output mode. WARNING: uses some memory
 
print("terminado")


#%%  Here begins the analysis

# parameters for spike raster analysis
binsize = 0.5 # bin size for population activity in ms
tbase = np.arange(0,wbn.Trun, binsize) # raster time base
kernel=signal.gaussian(10*2/binsize+1,2/binsize)
kernel/=np.sum(kernel)

pop_spikes = spikes[:,1]  #spike times for ALL neurons
popact,binedge = np.histogram(pop_spikes, tbase)
conv_popact=np.convolve(popact,kernel,mode='same')
#
LFP=np.mean(V_t,-1)  #LFP is the mean of all voltages
sdLFP = np.std(LFP)
sdV_t = np.std(V_t,0)
chi_sq = wbn.N*sdLFP**2/(np.sum(sdV_t**2))  ## This is some synchrony measure

#Volatge trace(s) frequency analysis. We're going to filter and reduce the number of points first
decimate=100;cutfreq=100  #final sample rate will be 500/s; enough for analysis up to 100Hz
b,a=signal.bessel(4,cutfreq*2*wbn.dt/1000,btype='low')

freqs=np.arange(1,100,0.5)  #Desired frequencies for wavelet spectrogram
Periods=1/(freqs*(decimate*wbn.dt)/1000)    #Desired periods in sample untis
dScales=Periods/wavelets.Morlet.fourierwl  #desired Scales

#filter and downsample in one line
LFPfilt=signal.filtfilt(b,a,LFP)[::decimate]
#Continuous Wavelet transform
wavelT=wavelets.Morlet(LFPfilt,scales=dScales)
pwr=wavelT.getnormpower()  

spec=np.sum(pwr,-1)
maxinspec=np.argmax(spec)
peakspec=spec[maxinspec]
freq_peakspec=freqs[maxinspec]
maxinspec_20=np.argmax(spec[:38])
peakspec_20=spec[maxinspec_20]
freq_peakspec_20=freqs[maxinspec_20]

#%%
xlims = [500,700]
xlimsind = [int(x/wbn.dt) for x in xlims]
xlims2 = [0,1000]

fig=plt.figure(1,figsize=(14,8), dpi= 100, facecolor='w', edgecolor='k') # tamaño, resolucion, color de fondo y borde de la figura
plt.clf()
 
plt.subplot(311)
plt.plot(spikes[:,1],spikes[:,0],'k.',ms=1)
plt.title("spikes raster")
plt.xlim(xlims2)

plt.subplot(345)
plt.plot(Time[xlimsind[0]:xlimsind[1]],V_t[xlimsind[0]:xlimsind[1],::5],lw=0.5)
plt.xlim(xlims) 
plt.ylim(-80,50 ) 
plt.title("All voltage traces")

plt.subplot(346)
plt.plot(Time,LFP)
plt.xlim(xlims) 
plt.title("LFP")

plt.subplot(347)
plt.plot(spikes[:,1],spikes[:,0],'k.',ms=1)
plt.title("spikes raster")
plt.xlim(xlims) 

plt.subplot(348)
plt.plot(binedge[1:],popact)
plt.plot(binedge[1:],conv_popact)
plt.xlim(xlims) 
plt.title("Population firing rate")
 
plt.subplot(3,4,12)

lags,c,_,_=plt.acorr(conv_popact - np.mean(conv_popact),maxlags=500,usevlines=False,linestyle='-',ms=0)
plt.title("autocorrelation")

peaks_i=np.where(np.diff(1*(np.diff(c)>0))==-1)[0]+1  #peak detection in autocorrelation plot
if len(peaks_i)>1:
    firstPeak=peaks_i[lags[peaks_i]>0][0]
    netwFreq=1000/(binsize*lags[firstPeak])
    maxCorr=c[firstPeak]
else:
    netwFreq=np.nan
    maxCorr=np.nan

plt.subplot(3,4,11)
plt.imshow(pwr,aspect='auto',extent=(0,max(Time),min(freqs),max(freqs)),origin='lower')
plt.colorbar()
#cbax=plt.subplot(3,4,11)
#plt.axis('off')
#cb=plt.colorbar(ax=cbax,use_gridspec=False)
plt.title("spectrogram")

plt.subplot(349)
vmax=np.max(np.abs(CMchem))
plt.imshow(CMchem,cmap='seismic',vmax=vmax, vmin=-vmax)
plt.colorbar()
plt.title("chemical CM")

plt.subplot(3,4,10)
plt.imshow(CM2,cmap='gray_r')

plt.title("electrical CM")


plt.tight_layout()

fRates=np.histogram(spikes,range(wbn.N))[0] / wbn.Trun *1000
fRateEm=np.mean(fRates[:wbn.Ne])
fRateEsd=np.std(fRates[:wbn.Ne])
fRateIm=np.mean(fRates[wbn.Ne:])
fRateIsd=np.std(fRates[wbn.Ne:])
fRateAm=np.mean(fRates)
fRateAsd=np.std(fRates)

print("Synchrony: %g    Population Frequency: %g Hz"%(chi_sq,netwFreq))
#plt.savefig(f"plots/Pi_{valpi:.2f}_Rate_{valirate:.1f}.png",dpi=300) 


#%%
xlims2=[500,1500]

plt.figure(2,figsize=(12,6))
plt.clf()

ax1=plt.subplot2grid((3,3),(0,0),colspan=2)
inhSpikes=np.where(spikes[:,0]>Ne)
excSpikes=np.where(spikes[:,0]<=Ne)

plt.plot(spikes[inhSpikes,1],spikes[inhSpikes,0],'b.',ms=1)
plt.plot(spikes[excSpikes,1],spikes[excSpikes,0],'r.',ms=1)
plt.title("spikes raster")
plt.xlim(xlims2)
plt.ylabel("neuron index")
ax1.set_xticklabels(())

ax2=plt.subplot2grid((3,3),(1,0),colspan=2)
plt.plot(Time,LFP)
plt.xlim(xlims2) 
plt.ylabel("(mV)")
ax2.set_xticklabels(())
plt.title("Mean Voltage")

plt.subplot2grid((3,3),(2,0),colspan=2)
plt.imshow(pwr,aspect='auto',extent=(0,max(Time),min(freqs),max(freqs)),origin='lower')
plt.xlabel("Time (ms)")
plt.ylabel("Frequency (Hz)")
plt.title("Spectrogram")
plt.xlim(xlims2)

plt.subplot2grid((3,3),(1,2),rowspan=2)
plt.plot(freqs,np.sum(pwr,-1)/1000)
plt.xlabel("Frequency (Hz)")
plt.ylabel("Power (x 1000)")

plt.figtext(0.73,0.9,"Synch index: %0.3g\nPop. Freq: %0.1f Hz\nMean rate: %0.1f s\u207b\u00b9"%(chi_sq,netwFreq,fRateAm),
            va='top',ha='left',size='x-large')

plt.tight_layout()

#plt.savefig("Ggj%gR_Pi%g_rate%g.png"%(Ggj,wbn.Pi,wbn.iRate),dpi=300)