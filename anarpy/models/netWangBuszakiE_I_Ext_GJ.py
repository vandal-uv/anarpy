# -*- coding: utf-8 -*-
"""
Created on Tue Sep 26 17:34:11 2017

@author: Patricio
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from numba import jit,float64,vectorize,int64
#import Wavelets

@vectorize([float64(float64)])
def alphan(v):
    return -0.01*(v+34)/(np.exp(-0.1*(v+34))-1) # ok RH

@vectorize([float64(float64)])
def betan(v):
    return 0.125*np.exp(-(v+44)/80) # ok RH

@vectorize([float64(float64)])
def alpham(v):
    return -0.1*(v+35)/(np.exp(-0.1*(v+35))-1) # ok RH

@vectorize([float64(float64)])
def betam(v):
    return 4*np.exp(-(v+60)/18) # ok RH

@vectorize([float64(float64)])
def alphah(v):
    return 0.07*np.exp(-(v+58)/20) # ok RH

@vectorize([float64(float64)])
def betah(v):
    return 1/(np.exp(-0.1*(v+28))+1) # ok RH

def expnorm(tau1,tau2):
    if tau1>tau2:
        t2=tau2; t1=tau1
    else:
        t2=tau1; t1=tau2
    tpeak = t1*t2/(t1-t2)*np.log(t1/t2)
    return (np.exp(-tpeak/t1) - np.exp(-tpeak/t2))/(1/t2-1/t1)

# Neurons Parameters
gNa = 35.0; gK = 9.0;  gL=0.1  #mS/cm^2
ENa = 55.0; EK = -90.0; EL = -65.0 #mV
phi = 5.0
VsynE = 0; VsynI = -80  #reversal potential
tau1E = 3; tau2E = 1
tau1I = 4; tau2I = 1
theta=-20 #threshold for detecting spikes
Iapp = 0; # uA/cm^2, injected current

#Synaptic parameters
mGsynE = 5; mGsynI = 200; mGsynExt = 3  #mean
sGsynE = 1; sGsynI = 10; sGsynExt = 1

Pe=0.3; Pi=0.3
iRate = 3.5   #Rate of external input

mdelay=1.5; sdelay = 0.1 #ms synaptic delays, mean and SD
dt = 0.02 #ms

#Network parameters
Ne=100 #Numero de neuronas excitatorias
Ni=25 #Numero de neuronas inhibitorias

def genRandomCM(mode='all', AdjMe=None, AdjMi=None):
    global CMe,CMi,GsynExt,N
    if mode not in ('exc','inh','excinh','ext','all'):
        raise ValueError("mode has to be one of ['exc','inh','excinh','ext','all']")
    N=Ne+Ni

    factE = 1000*dt*expnorm(tau1E,tau2E)
    factI = 1000*dt*expnorm(tau1I,tau2I)
  
    
    if mode in ('exc','excinh','all'):
        GsynE = np.random.normal(mGsynE,sGsynE,size=(N,Ne))
        GsynE = GsynE*(GsynE>0) # remove negative values
        if AdjMe is None:
            AdjMe=np.random.binomial(1,Pe,size=(N,Ne))
        elif AdjMe.shape!=(N,Ne):
            raise ValueError("Check dimensions of AdjMe. It has to be N x Ne")
        CMe= AdjMe * GsynE / factE 
    
    if mode in ('inh','excinh','all'):
        GsynI = np.random.normal(mGsynI,sGsynI,size=(N,Ni))
        GsynI = GsynI*(GsynI>0) # remove negative values
        if AdjMi is None:
            AdjMi=np.random.binomial(1,Pi,size=(N,Ni))
        elif AdjMi.shape!=(N,Ni):
            raise ValueError("Check dimensions of AdjMe. It has to be N x Ni")
        CMi= AdjMi* GsynI / factI
    
    if mode in ('ext','all'):
    #Weigths for external random input
        GsynExt = np.random.normal(mGsynExt,sGsynExt,size=N)
        GsynExt = GsynExt*(GsynExt>0) / factE # remove negative values and normalize
    genDelays()
        
def genDelays():
    global delay,delay_dt
    delay = np.random.normal(mdelay,sdelay,size=N)
    delay_dt=(delay/dt).astype(int)


genRandomCM()

Ggj=0.001  # not so big gap junction conductance
CMelec=Ggj * np.random.binomial(1,0.3,(Ni,Ni))  #mock electric connectivity

#firing=np.zeros(N)


@jit(float64[:,:](float64[:,:],int64[:],int64),nopython=True)
def WB_network(X,ls,i):
    v=X[0,:]
    h=X[1,:]
    n=X[2,:]
    sex=X[3,:]
    sey=X[4,:]
    six=X[5,:]
    siy=X[6,:]
    sexe=X[7,:]
    seye=X[8,:]
    
    minf=alpham(v)/(betam(v)+alpham(v))
    INa=gNa*minf**3*h*(v-ENa) 
    IK=gK*n**4*(v-EK) 
    IL=gL*(v-EL) 
    
    ISyn= (sey + seye) * (v - VsynE) + siy * (v - VsynI)
    Igj = np.zeros(N)
    Igj[Ne:] = np.sum(CMelec * (np.expand_dims(v[Ne:],1) - v[Ne:]),-1)
    firingExt = np.random.binomial(1,iRate*dt,size=N)
    firing=1.*(ls==(i-delay_dt))

    return np.vstack((-INa-IK-IL-ISyn-Igj+Iapp,
                     phi*(alphah(v)*(1-h) - betah(v)*h),
                     phi*(alphan(v)*(1-n) - betan(v)*n),
                     -sex*(1/tau1E + 1/tau2E) - sey/(tau1E*tau2E) + np.dot(CMe,firing[0:Ne]),
                     sex,
                     -six*(1/tau1I + 1/tau2I) - siy/(tau1I*tau2I) + np.dot(CMi,firing[Ne:]),
                     six,
                     -sexe*(1/tau1E + 1/tau2E) - seye/(tau1I*tau2I) + firingExt*GsynExt,
                     sexe))

equil=400
Trun=2000    
#Total=Trun + equil #ms





#nsteps=len(Time)

def initVars(v=None):
    if v is None:
        v_init=np.random.uniform(-80,-60,size=N) #-70.0 * np.ones(N) # -70 is the one used in brian simulation
    h=1/(1+betah(v_init)/alphah(v_init))
    n=1/(1+betan(v_init)/alphan(v_init))
    sex=np.zeros_like(v_init)
    sey=np.zeros_like(v_init)
    six=np.zeros_like(v_init)
    siy=np.zeros_like(v_init)
    sexe=np.zeros_like(v_init)
    seye=np.zeros_like(v_init)
    return np.array([v_init,h,n,sex,sey,six,siy,sexe,seye])

#X=initVars()

def runSim(v_init=None,output='spikes'):
    global firing
    if v_init is None:
        X=initVars()
    elif len(v_init)==N:
        X=initVars(v_init)
    else:
        raise ValueError("v_init has to be None or an array of length N")
    
    if output not in ('spikes','LFP','allV'):
        raise ValueError("output has to be one of ['spikes','LFP','allV']")

    firing=np.zeros(N)     
    #adaptation simulation - not stored
    equil_dt=int(equil/dt)  #equilibrium time - in samples
    bufferl=100*(np.max(delay_dt)//100+1)
    V_t=np.zeros((bufferl,N))
    lastSpike=equil_dt*np.ones(N,dtype=np.int64)
    for i in range(equil_dt):
        ib=i%bufferl
        X+=dt*WB_network(X,lastSpike,i)
#        firing=1*(V_t[ib-delay_dt,range(N)]>theta)*(V_t[ib-delay_dt-1,range(N)]<theta)
    
    Time = np.arange(0,Trun,dt)
    
    if output=='spikes':
        spikes=[]
        bufferl=100*(np.max(delay_dt)//100+1)
        V_t=np.zeros((bufferl,N))
        lastSpike=lastSpike-equil_dt
        lastSpike[lastSpike==0]=int(Trun/dt)
        for i,t in enumerate(Time):
            ib=i%bufferl
            V_t[ib]=X[0]
            if np.any((V_t[ib]>theta)*(V_t[ib-1]<theta)):
                for idx in np.where((V_t[ib]>theta)*(V_t[ib-1]<theta))[0]:
                    spikes.append([idx,t])
                    lastSpike[idx]=i
            X+=dt*WB_network(X,lastSpike,i)
        return np.array(spikes)
    
    elif output=='LFP':
        spikes=[]
        bufferl=100*(np.max(delay_dt)//100+1)
        V_t=np.zeros((bufferl,N))
        LFP_t=np.zeros(len(Time))
        lastSpike=lastSpike-equil_dt
        lastSpike[lastSpike==0]=int(Trun/dt)        
        for i,t in enumerate(Time):
            ib=i%bufferl
            V_t[ib]=X[0]
            LFP_t[i]=np.mean(X[0])
            if np.any((V_t[ib]>theta)*(V_t[ib-1]<theta)):
                for idx in np.where((V_t[ib]>theta)*(V_t[ib-1]<theta))[0]:
                    spikes.append([idx,t])
                    lastSpike[idx]=i
            X+=dt*WB_network(X,lastSpike,i)
        return np.array(spikes),LFP_t,Time
    
    elif output=='allV':
        spikes=[]
        V_t=np.zeros((len(Time),N))
        lastSpike=lastSpike-equil_dt
        lastSpike[lastSpike==0]=int(Trun/dt)        
        for i,t in enumerate(Time):
            V_t[i]=X[0]
            if np.any((V_t[i]>theta)*(V_t[i-1]<theta)):
                for idx in np.where((V_t[i]>theta)*(V_t[i-1]<theta))[0]:
                    spikes.append([idx,t])
                    lastSpike[idx]=i
            X+=dt*WB_network(X,lastSpike,i)
        return np.array(spikes),V_t,Time
    
    
def ParamsNode():    
    pardict={}
    for var in ('gNa','gK','gL','ENa','EK','EL','phi','theta','Iapp'):
        pardict[var]=eval(var)
        
    return pardict

def ParamsSyn():
    pardict={}
    for var in ('VsynE','VsynI','tau1E','tau2E','tau1I','tau2I','mdelay','sdelay',
                'factE','factI'):
        pardict[var]=eval(var)
        
    return pardict

def ParamsNet():
    pardict={}
    for var in ('Ne','Ni','N','Pe','Pi','iRate'):
        pardict[var]=eval(var)
    return pardict


def ParamsNetMatrix():
    pardict={}
    for var in ('mGsynE','mGsynI','mGsynExt','sGsynE','sGsynI','sGsynExt',
                'GsynE','GsynI','GsynExt'):
        pardict[var]=eval(var)
    return pardict



def ParamsSim():
    pardict={}
    for var in ('equil','Trun','dt'):
        pardict[var]=eval(var)
        
    return pardict

    
#    V_t = np.zeros((nsteps,N))
#    for i in range(nsteps):
#        V_t[i]=X[0]    
#        X+=dt*WB_network(X,i)
#%%
if __name__=='__main__':
    
    
    Pi=0.3
    iRate = 3.
    genRandomCM()
    
    Ggj=0.1  # not so big gap junction conductance
    CMelec=Ggj * np.random.binomial(1,0.3,(Ni,Ni))  #mock electric connectivity
    
    WB_network.recompile()
        
    spikes=runSim()
    
    
    
#    spikes,V_t,Time=runSim(output='allV')
    
    binsize = 0.5 # bin size for population activity in ms
    tbase = np.arange(0,Trun, binsize) # raster time base
    
    kernel=signal.gaussian(10*2/binsize+1,2/binsize)
    kernel/=np.sum(kernel)
    
    #spikes=[(np.diff(1*(V_t[:,i]>-20))==1).nonzero()[0] for i in range(N)]
    #pop_spikes = np.asarray([item for sublist in spikes for item in sublist]) # todas las spikes de la red
    pop_spikes = spikes[:,1]
    popact,binedge = np.histogram(pop_spikes, tbase)
    
    conv_popact=np.convolve(popact,kernel,mode='same')
    
    #decimate=50;cutfreq=100
    #b,a=signal.bessel(4,cutfreq*2*dt/1000,btype='low')
    #
    #freqs=np.arange(10,100,0.5)  #Desired frequencies
    #Periods=1/(freqs*(decimate*dt)/1000)    #Desired periods in sample untis
    #dScales=Periods/Wavelets.Morlet.fourierwl  #desired Scales
    #
    #LFP=np.mean(V_t,-1)
    #sdLFP = np.std(LFP)
    #sdV_t = np.std(V_t,0)
    #chi_sq = N*sdLFP**2/(np.sum(sdV_t**2))
    
    #%%
    xlims = [900,1100]
    
    fig=plt.figure(2,figsize=(12,8), dpi= 80, facecolor='w', edgecolor='k') # tamaño, resolucion, color de fondo y borde de la figura
    plt.clf()
    
    #plt.subplot(341)
    #plt.plot(Time,V_t[:,::5],lw=0.5)
    #plt.ylim(-80,50 ) 
    ##
    #plt.subplot(342)
    #plt.plot(Time,LFP)
    
    plt.subplot(343)
    #for i in range(N):
    #    plt.plot(dt*np.array(spikes[i]),i*np.ones_like(spikes[i]),'k.',ms=0.5)
    plt.plot(spikes[:,1],spikes[:,0],'k.',ms=1)
    
    plt.subplot(344)
    plt.plot(popact)
    plt.plot(conv_popact)
    
    #plt.subplot(345)
    #plt.plot(Time,V_t[:,::5],lw=0.5)
    #plt.xlim(xlims) 
    #plt.ylim(-80,50 ) 
    #
    #plt.subplot(346)
    #plt.plot(Time,LFP)
    #plt.xlim(xlims) 
    
    plt.subplot(347)
    #for i in range(N):
    #    plt.plot(dt*np.array(spikes[i]),i*np.ones_like(spikes[i]),'.')
    plt.plot(spikes[:,1],spikes[:,0],'k.',ms=1)
    
    plt.xlim(xlims) 
    
    plt.subplot(348)
    plt.plot(popact)
    plt.plot(conv_popact)
    plt.xlim(xlims) 
    
    #plt.subplot(3,4,9)
    #spect,freqs,t,_=plt.specgram(LFP,Fs=1000/dt,interpolation=None,
    #                             detrend='mean')
    #plt.ylim(20,80)
    
    plt.subplot(3,4,12)
    
    lags,c,_,_=plt.acorr(conv_popact,maxlags=500,usevlines=False,linestyle='-',ms=0)
    peaks=lags[np.where(np.diff(1*(np.diff(c)>0))==-1)[0]+1]
    
    if len(peaks)>1:
        netwFreq=1000/(binsize*peaks[peaks>0][0])
    else:
        netwFreq=np.nan
    
    
    # Wavelet stuff
    
    #
    #Vfilt=signal.filtfilt(b,a,LFP)[::decimate]
    #
    #
    #
    ##wavel=Wavelets.Morlet(EEG,largestscale=10,notes=20,scaling='log')
    #wavelT=Wavelets.Morlet(Vfilt,scales=dScales)
    ##cwt=np.array([wavel.getdata() for wavel in wavelT])
    #pwr=wavelT.getnormpower()
    #
    ##
    #
    #plt.subplot(3,4,9)
    #s,f,t,_=plt.specgram(Vfilt,Fs=1000/(dt*decimate),interpolation=None,detrend='mean')
    #plt.ylim(0,100)
    #
    #
    #plt.subplot(3,4,11)
    #plt.plot(Vfilt)
    #
    #plt.subplot(3,4,10)
    #plt.imshow(pwr,aspect='auto',extent=(0,Trun,min(freqs),max(freqs)),origin='lower')
    ##cbax=plt.subplot(3,4,11)
    ##plt.axis('off')
    ##cb=plt.colorbar(ax=cbax,use_gridspec=False)
    
    plt.tight_layout()
    