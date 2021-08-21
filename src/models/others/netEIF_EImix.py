# -*- coding: utf-8 -*-
"""
Created on Mon Oct  8 21:37:23 2018

@author: patri
"""

import numpy as np
import matplotlib.pyplot as plt
from numba import jit,float64


Vr=-70
DT=2
Vth=-50
Vreset=-58
Vspk=0
gL= 6
C=200
tau_exc=5; Eexc=0 
tau_inh=20; Einh=-70

# Main function
@jit(float64[:,:](float64[:,:],float64[:]),nopython=True)
def LIF(X,I):
    """
    Network of Exponential-Integrate-and-Fire
    
    Calculates the derivatives of voltage and synaptic conductances at a given time.

    Parameters
    ----------
    X : array of float64 of size (3,N)
        N is the number of neurons. Accross the first dimension, the variables are Voltage, excitatory conductance, inhibitory conductance.
    I : array of float64, 1-D and length N 
        External current applied to each neuron.

    Returns
    -------
    array of float64 of size (3,N)
        The derivative at time _t_ for each variable.

    """
    v,ge,gi=X
    return np.vstack(((-gL*(v-Vr)+DT*np.exp((v-Vth)/DT)-ge*(v-Eexc)-gi*(v-Einh)+I)/C,
                     -ge / tau_exc,
                     -gi / tau_inh))

#names of parameters
    
N_i=[16,10] #Numero de neuronas [excitatorias,inhibitorias, ...]
N=sum(N_i) #numero de neuronas a simular

#individual parameter values
Vr_i=[-65,-62] #-70
DT_i=[0.8, 3] #2
Vth_i=[-52,-42] #-50
Vreset_i=[-53, -54] #-58
Vspk_i=0
gL_i=[4.3, 2.9]# 6
C_i=[104, 30] #200

Params=["Vr","DT","Vth","Vreset","Vspk","gL","C"]
"""
We will take the parameters as lists of lenght K (number of neuron
types) and convert them to arrays of length N (number of neurons).
"""    
def InitPars():
    """
    
    Creates arrays of parameters of length N (number of neurons, one parameter per neuron)
    from arrays of length K (number of neuron types). If any of the given values is a scalar, 
    it will remain a scalar. **This function does not take any argument**;
    it operates from the module-wide existing variables 
    
    Vr_i    Membrane (leak) reversal potential
    
    DT_i    ΔT parameter of EIF neuron
    
    Vth_i    V_th parameter of EIF neuron
    
    Vreset_i     Voltage value after a spike (reset voltage)
    
    Vspk_i     Voltage for detecting spikes
    
    gL_i    Membrane (leak) conductance
    
    C_i     Membrane capacitance

    These variables must exist either as a scalar or as a list of length K (number of neuron types)

    Raises
    ------
    ValueError
        If any of the variables is of a different length than the neuron types.

    Returns
    -------
    None. The arrays of variables become available module-wide.

    """
    global Vr,DT,Vth,Vreset,Vspk,gL,C
    
    for param in Params:
        if type(eval(param+"_i"))==list:
            if eval("len("+param+"_i)")==len(N_i):
                exec("%s = np.array([%s_i[0],]*N_i[0] + [%s_i[1],]*N_i[1])"%((param,)*3), globals())
            else:
                raise ValueError("check the length of variable"+param)
        else:
            exec("%s = %s_i"%((param,)*2))
    
#NETWORK parameters    
#Synaptic Parameters
tau_exc=5; Eexc=0 
tau_inh=20; Einh=-70

#CONNECTIVITY MATRIX
# By default, all excitatory and inhibitory weigths will be the same. This
# doesn't have to be the case; just use a heterogeneous Connectivity matrix.
WsynE = 0.2; WsynI = 0.3 #defaults synaptic weigths
#recordar que será siempre positivo porque son sinapsis de tipo conductancia

## DEFAULT CONNECTIVITY: Random connectivity matrices
def InitCM(Pee=0.4, Pei=0.6, Pie=0.8, Pii=0.4):
    """
    Calculate a random adjacency Matrix and store it in the module-wide variables CMe and CMi.
    
    CMe : Array of size (N,Ne). All excitatory inputs
    
    CMi : Array of size (N,Ni). All inhibitory inputs
    
    The (i,j) entry of CMe (CMi) matrix denotes the connection from the j-th excitatory
    (inhibitory) neuron to the i-th neuron

    Parameters
    ----------
    Pee : float, 0 <= Pee <= 1, optional
        Connection probability from excitatory to excitatory neurons. 
    Pei : float, 0 <= Pei <= 1, optional
        Connection probability from inhibitory to excitatory neurons. 
    Pie : float, 0 <= Pie <= 1, optional
        Connection probability from excitatory to inhibitory neurons. 
    Pii : float, 0 <= Pii <= 1, optional
        Connection probability from inhibitory to inhibitory neurons. 

    Returns
    -------
    None. CMe and CMi matrices are internally defined

    """
    global CMe,CMi,Ne,Ni
    Ne=N_i[0];Ni=N_i[1]
    CMee=WsynE * np.random.binomial(1,Pee,size=(Ne,Ne)) #from exc to exc
    CMie=WsynE * np.random.binomial(1,Pie,size=(Ni,Ne)) #from exc to inh
    CMei=WsynI * np.random.binomial(1,Pei,size=(Ne,Ni)) #from inh to exc
    CMii=WsynI * np.random.binomial(1,Pii,size=(Ni,Ni)) #from inh to exc
    CMe=np.r_[CMee,CMie]  #All inputs FROM exc
    CMi=np.r_[CMei,CMii]  #All inputs FROM inh

#simulation parameters
tstop=5000; dt=0.1

# Input (noisy) current
Ioffset=80 #media del estímulo
noise=2 #desviacion estandar del ruido

def InitVars(V0=-50):
    """
    Create a vector of initial variables. Initial voltage can be set, synaptic conductances
    will always be 0

    Parameters
    ----------
    V0 : float, optional
        Initial voltage. The default is -50.

    Returns
    -------
    X : array of floats, dimension (3,N)
        N is the number of neurons. 
        Accross the first dimension, the variables are Voltage, excitatory conductance, inhibitory conductance.

    """
    N=sum(N_i)
    v=V0*np.ones(N) #La variable v tiene dimensión nLIF
    ge=np.zeros_like(v)
    gi=np.zeros_like(v)
    X=np.array([v,ge,gi])
    return X

def runSim(X=None,recordV=False):
    """
    Run a simulation with the currently defined parameters.

    Parameters
    ----------
    X : 3xN Array of floats, optional
        Initial conditions. N is the number of neurons. Accross the first dimension,
        the variables are Voltage, excitatory conductance, inhibitory conductance.
        If None, ICs are set to V=-50, synaptic conductances=0. The default is None.
    recordV : boolean, optional
        If True, variables will be recorded for each neuron and returned at the end.
        Use with caution when simulating large networks. The default is False.

    Returns
    -------
    spikes : list of lists
        Each element of the list contains the index of the neuron and the time of firing.
    
    time : 1D array of floats
        Contains the times of the simulation
        
    X_t : T x 3 x N array of floats
        (Only returned if recordV==True) Time traces for voltage and synaptic conductances.
        T is the same length as time.

    """
    global N
    N=sum(N_i)
    if X is None:
        X=InitVars()
    time=np.arange(0,tstop,dt)
    spikes=[]
    # Istim=Ioffset+np.sqrt(noise/dt)*np.random.normal(size=(len(time),N)) #distinta corriente en cada neurona
    sqdt=np.sqrt(noise/dt)
    
    if recordV:
        X_t=np.zeros((len(time),3,N))
        for i,t in enumerate(time):
            Istim=Ioffset+sqdt*np.random.normal(size=N) #distinta corriente en cada neurona
            X_t[i,:]=X
            X+=dt*LIF(X,Istim)
            v=X[0]
            if any(v>=0):
                idx=np.where(v>=0)[0]
                X[0,idx] = Vreset[idx]
                for idx_i in idx:
                    spikes.append([idx_i,t])
                    if idx_i<Ne:
                        X[1] = X[1] + CMe[:,idx_i]
                    else:
                        X[2] = X[2] + CMi[:,idx_i-Ne]
        return spikes,time,X_t
    else:
        for i,t in enumerate(time):
            Istim=Ioffset+sqdt*np.random.normal(size=N) #distinta corriente en cada neurona
            X+=dt*LIF(X,Istim)
            v=X[0]
            if any(v>=0):
                idx=np.where(v>=0)[0]
                X[0,idx] = Vreset[idx]
                for idx_i in idx:
                    spikes.append([idx_i,t])
                    if idx_i<Ne:
                        X[1] = X[1] + CMe[:,idx_i]
                    else:
                        X[2] = X[2] + CMi[:,idx_i-Ne]
        return spikes,time


def ParamsNeuron():    
    pardict={}
    for var in ("Vr","DT","Vth","Vreset","Vspk","gL","C"):
        pardict[var]=eval(var)
        
    return pardict

def ParamsNeuronByType():
    pardict={}
    for var in ("Vr","DT","Vth","Vreset","Vspk","gL","C"):
        pardict[var]=eval(var+"_i")
        
    return pardict

def ParamsNet():
    pardict={}
    for var in ('N_i','N','tau_exc', 'Eexc', 'tau_inh', 'Einh', 'WsynE', 'WsynI', 'Ioffset', 'noise'):
        pardict[var]=eval(var)
    return pardict


def ParamsCMatrix():
    pardict={}
    for var in ('CMe','CMi'):
        pardict[var]=eval(var)
    return pardict

def ParamsSim():
    pardict={}
    for var in ('tstop','dt'):
        pardict[var]=eval(var)
        
    return pardict


#%%

if __name__=="__main__":

    N_i=[75,25]
    InitPars()
    InitCM()
    
    LIF.recompile()
    
    spikes,time,X_t=runSim(InitVars(),recordV=True)
      
    #%%
    
    
    if len(spikes)>0:
        spikes=np.array(spikes)
        neuron,fRates=np.unique(spikes[:,0],return_counts=True)
        fRates=fRates/(tstop/1000)            
                
    #%%
    # Vplot=np.arange(-65,-25,0.1)
    # plt.figure(2,figsize=(5,5))
    # plt.clf()
    # plt.plot(Vplot,F(Vplot))
    
    plt.figure(1,figsize=(12,6))
    plt.clf()
    if len(spikes)>0:
        plt.subplot(231)
        plt.plot(spikes[:,1],spikes[:,0],'.',ms=1)
        plt.xlim((0,tstop))
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
    maxV=np.max(np.abs(np.c_[CMe,CMi]))
    plt.imshow(np.c_[CMe*1,CMi*-1],cmap='bwr',vmin=-maxV,vmax=maxV)
    plt.colorbar()
    
    plt.tight_layout()
