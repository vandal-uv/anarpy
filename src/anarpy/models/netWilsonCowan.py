# -*- coding: utf-8 -*-
"""
Created on Tue Dec 22 14:56:13 2015
The Huber_braun neuronal model function
@author: porio
"""

import numpy as np
from numba import jit,float64, vectorize

from numba.core.errors import NumbaPerformanceWarning
import warnings
warnings.simplefilter('ignore', category=NumbaPerformanceWarning)

#Node parameters
# Any of them can be redefined as a vector of length nnodes
#excitatory connections
a_ee=3.5; a_ie=2.5
#inhibitory connections
a_ei=3.75; a_ii=0
#tau
tauE=0.010; tauI=0.020  # Units: seconds
#external input
P = 0.4 # 0.4
Q = 0
# inhibitory plasticity

rE,rI=0.5,0.5
mu = 1;sigma=0.25

D=0
# This is a mock value, it will get overwritten
sqdtD=0.1

#network parameters
N=50
G=0.001
CM=np.random.binomial(1,0.1,(N,N)).astype(np.float64)


@vectorize([float64(float64)],nopython=True)
def S(x):
    return (1/(1+np.exp(-(x-mu)/sigma)))

@jit(float64[:,:](float64,float64[:,:]),nopython=True)
def wilsonCowan(t,X):
    E,I = X
    noise=np.random.normal(0,sqdtD,size=N)
    return np.vstack(((-E + (1-rE*E)*S(a_ee*E - a_ie*I + G*np.dot(CM,E) + P + noise))/tauE,
                     (-I + (1-rI*I)*S(a_ei*E - a_ii*I ))/tauI))

@jit(float64[:,:](float64,float64[:,:]),nopython=True)
def wilsonCowanDet(t,X):
    E,I = X
    return np.vstack(((-E + (1-rE*E)*S(a_ee*E - a_ie*I + G*np.dot(CM,E) + P))/tauE,
                     (-I + (1-rI*I)*S(a_ei*E - a_ii*I ))/tauI))

"""
The main function is starting from here          
"""

E0=0.1
I0=0.1

### Time units are seconds  ###
tTrans=2
tstop=10
dt=0.001    #interval for points storage
dtSim=0.0001   #interval for simulation (ODE integration)
downsamp=int(dt/dtSim)

# # adaptation time - note that adaptation occurs twice
# timeTrans=np.arange(0,tTrans,dtSim)
# # Simulation time
# timeSim=np.arange(0,tstop,dtSim)
# # time for storing data
# time=np.arange(0,dt,tstop)

def SimAdapt(Init=None):
    """
    Runs a deterministic simulation of timeTrans. 
    """
    global timeTrans
    if Init is None:
        Var=np.array([E0,I0])[:,None]*np.ones((1,N))
    else:
        Var=Init
    # generate the vector again in case variables have changed
    timeTrans=np.arange(0,tTrans,dtSim)

    wilsonCowanDet.recompile()
    
    # Varinit=np.zeros((len(timeTrans),3,N))
    for i,t in enumerate(timeTrans):
        # Varinit[i]=Var
        Var+=dtSim*wilsonCowanDet(t,Var)
    
    return Var

def Sim(Var0=None,verbose=False):
    """
    Run a network simulation with the current parameter values.
    
    If D==0, run deterministic simulation.
    Note that the time unit in this model is seconds.

    Parameters
    ----------
    Var0 : ndarray (2,N), ndarray (2,), or None
        Initial values. Either one value per each node (2,N), 
        one value for all (2,) or None. If None, an equilibrium simulation
        is run with faster plasticity kinetics. The default is None.
    verbose : Boolean, optional
        If True, some intermediate messages are shown.
        The default is False.

    Raises
    ------
    ValueError
        An error raises if the dimensions of CM and the number of nodes
        do not match.

    Returns
    -------
    Y_t : ndarray
        Time trajectory for the three variables of each node.
    time : TYPE
        Values of time.

    """
    global CM,sqdtD,downsamp
    
    if CM.shape[0]!=CM.shape[1] or CM.shape[0]!=N:
        raise ValueError("check CM dimensions (",CM.shape,") and number of nodes (",N,")")
    
    if CM.dtype is not np.dtype('float64'):
        try:
            CM=CM.astype(np.float64)
        except:
            raise TypeError("CM must be of numeric type, preferred float")
    
    
    if type(Var0)==np.ndarray:
        if len(Var0.shape)==1:
            Var=Var0*np.ones((1,N))
        else:
            Var=Var0
    elif Var0 is None:
        Var=SimAdapt()

    timeSim=np.arange(0,tstop,dtSim)
    time=np.arange(0,tstop,dt)
    downsamp=int(dt/dtSim)
         
    Y_t=np.zeros((len(time),2,N))  #Vector para guardar datos

    if verbose:
        print("Simulating %g s dt=%g, Total %d steps"%(tstop,dtSim,len(timeSim)))

    if verbose and D==0:
        wilsonCowanDet.recompile()
        for i,t in enumerate(timeSim):
            if i%downsamp==0:
                Y_t[i//downsamp]=Var
            if t%10==0:
                print("%g of %g s"%(t,tstop))
            Var += dtSim*wilsonCowanDet(t,Var)

    if verbose and D>0:
        sqdtD=D/np.sqrt(dtSim)
        wilsonCowan.recompile()
        for i,t in enumerate(timeSim):
            if i%downsamp==0:
                Y_t[i//downsamp]=Var
            if t%10==0:
                print("%g of %g ms"%(t,tstop))
            Var += dtSim*wilsonCowan(t,Var)

    if not verbose and D==0:
        wilsonCowanDet.recompile()
        for i,t in enumerate(timeSim):
            if i%downsamp==0:
                Y_t[i//downsamp]=Var
            Var += dtSim*wilsonCowanDet(t,Var)

    if not verbose and D>0:
        sqdtD=D/np.sqrt(dtSim)
        wilsonCowan.recompile()
        for i,t in enumerate(timeSim):
            if i%downsamp==0:
                Y_t[i//downsamp]=Var
            Var += dtSim*wilsonCowan(t,Var)
            
    return Y_t,time
        
def ParamsNode():
    pardict={}
    for var in ('a_ee','a_ei','a_ie','a_ii','tauE','tauI',
                'P','Q','rE','rI','mu','sigma'):
        pardict[var]=eval(var)
        
    return pardict

def ParamsNet():
    pardict={}
    for var in ('N','G','CM'):
        pardict[var]=eval(var)
        
    return pardict

def ParamsSim():
    pardict={}
    for var in ('tTrans','tstop','dt','dtSim'):
        pardict[var]=eval(var)
        
    return pardict
#%%
if __name__=="__main__":
    
    import matplotlib.pyplot as plt
    
    tTrans=25
    tstop=10
    G=0.005
    D=0
    
    N=20
    CM=np.random.binomial(1,0.3,(N,N)).astype(np.float32)
    P=np.random.uniform(0.34,0.5,N)
    # P=.4
    
    Vtrace,time=Sim(verbose=True)
    ParamsNode()
#%%    
    plt.figure(1)
    plt.clf()
    plt.subplot(211)
    plt.plot(time,Vtrace[:,0,:])

    plt.subplot(212)
    plt.plot(time,Vtrace[:,1,:])
    
    
