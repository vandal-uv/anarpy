# -*- coding: utf-8 -*-
"""
Created on Tue Dec 22 14:56:13 2015
Wilson Cowan model with noise and plasticity
@author: porio
"""

from matplotlib.pyplot import getp
from numba.core.types.misc import Module
import numpy as np
from numba import jit,float64,boolean, vectorize

from numba.core.errors import NumbaPerformanceWarning
import warnings
warnings.simplefilter('ignore', category=NumbaPerformanceWarning)

#Node parameters
# Any of them can be redefined as a vector of length nnodes
#excitatory connections
a_ee=3.5; a_ie_0=2.5
#inhibitory connections
a_ei=3.75; a_ii=0
#tau
tauE=0.010; tauI=0.020  # Units: seconds
#external input
P = 0.4 # 0.4
Q = 0
# inhibitory plasticity
rhoE=0.14 # target mean value for E
tau_ip=2  #time constant for plasticity

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

@jit(float64[:,:](float64,float64[:,:],boolean,boolean),nopython=True)
def wilsonCowan(t,X, noisy, plasticity):
    
    E,I,a_ie = X
    noise=(noisy)*np.random.normal(0,sqdtD,size=N)
    
    exc = (-E + (1-rE*E)*S(a_ee*E - a_ie*I + G*np.dot(CM,E) + P + noise))/tauE
    inh = (-I + (1-rI*I)*S(a_ei*E - a_ii*I ))/tauI
    plast = (plasticity)*(I*(E-rhoE)/tau_ip)
    
    return np.vstack((exc,inh,plast))
    
   
"""
The main function is starting from here          
"""

E0=0.1
I0=0.1

### Time units are seconds  ###
tTrans=100
tstop=100
dt=0.001    #interval for points storage
dtSim=0.0001   #interval for simulation (ODE integration)
downsamp=int(dt/dtSim)



def SimAdapt(Init=None, plasticity=False):
    """
    Runs a deterministc simulation of duration 'timeTrans'
    Runs two deterministic simulations of timeTrans. 
    
    First with tau_ip=0.05
    Second with tau_ip=tau_ip/2
    """

    global tau_ip, timeTrans
    if Init is None:
        #print("initializing vars")
        Var=np.array([E0,I0,a_ie_0])[:,None]*np.ones((1,N))
    else:
        Var=Init
    # generate the vector again in case variables have changed
    timeTrans=np.arange(0,tTrans,dtSim)
    if plasticity:
        old_tau_ip=tau_ip
        tau_ip=0.05
    noisy = False
    wilsonCowan.recompile()

    
    
    # Varinit=np.zeros((len(timeTrans),3,N))
    for i,t in enumerate(timeTrans):
        # Varinit[i]=Var 
        # at the begining,  Var is boolean for soome reason
        Var+=dtSim*wilsonCowan(t,Var,noisy, plasticity)
    
    if plasticity:
        tau_ip=old_tau_ip/2
        wilsonCowan.recompile()
        
        for i,t in enumerate(timeTrans):
            Var+=dtSim*wilsonCowan(t,Var,noisy,plasticity)
        tau_ip=old_tau_ip
    
    return Var

def Sim(Var0=None,verbose=False, noisy = False, plasticity = False):
    """
    Run a network simulation with the current parameter values.
    
    If D==0, run deterministic simulation.
    Note that the time unit in this model is seconds.

    Parameters
    ----------
    Var0 : ndarray (X,N), ndarray (X,), or None
        Initial values. Either one value per each node (X,N), 
        one value for all (X,) or None. If None, an equilibrium simulation
        is run with faster plasticity kinetics. The default is None.
        X is the dimension of the variables. If X==2, the model is without
        plasticity. If X==3, model is with plasticity
    verbose : Boolean, optional
        If True, some intermediate messages are shown.
        The default is False.
    noisy: boolean

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
    
    #Connectivity matrix assertion
    if CM.shape[0]!=CM.shape[1] or CM.shape[0]!=N:
        raise ValueError("check CM dimensions (",CM.shape,") and number of nodes (",N,")")
    
    if CM.dtype is not np.dtype('float64'):
        try:
            CM=CM.astype(np.float64)
        except:
            raise TypeError("CM must be of numeric type, preferred float")
    
    if plasticity:
        if (rhoE==None) or (tau_ip==None):
            raise ValueError("Plastic variables are not iniatilized!")
    
    #variable initialization
    if type(Var0)==np.ndarray:
        if len(Var0.shape)==1:
            Var=Var0*np.ones((1,N))
        else:
            Var=Var0
    elif Var0 is None:
        Var=SimAdapt(None,plasticity)

    timeSim=np.arange(0,tstop,dtSim)
    time=np.arange(0,tstop,dt)
    downsamp=int(dt/dtSim)
    
    Y_t=np.zeros((len(time),3,N))

    if verbose:
        print("Simulating %g s dt=%g, Total %d steps"%(tstop,dtSim,len(timeSim)))

    if noisy:
        sqdtD=D/np.sqrt(dtSim)
    wilsonCowan.recompile()
    
    for i,t in enumerate(timeSim):
        if i%downsamp==0:
            Y_t[i//downsamp]=Var
        Var += dtSim*wilsonCowan(t,Var, noisy, plasticity)
        if verbose:
            if t%10==0:
                print("%g of %g ms"%(t,tstop))
            
    return Y_t,time

def getParams(pars = 'node'):

    if(pars=='network'):
        params = ('N','G','CM')
    elif(pars=='simulation'):
        params = ('tTrans','tstop','dt','dtSim')
    elif(pars=='node'):
        params = ('a_ee','a_ei','a_ii','tauE','tauI',
                'P','Q','rhoE','tau_ip','rE','rI','mu','sigma')
    pardict={}
    for var in params:
        pardict[var]=eval(var)
    return pardict


#%%
if __name__=="__main__":
    
    import matplotlib.pyplot as plt
    
    tTrans=25
    tstop=10
    G=0.005
    D=0
    rhoE=0.14
    
    N=20
    CM=np.random.binomial(1,0.3,(N,N)).astype(np.float32)
    P=np.random.uniform(0.3,0.5,N)
    # P=.4
    
    Vtrace,time=Sim(verbose=True)
    getParams(pars='node')
#%%    
    plt.figure(1)
    plt.clf()
    plt.subplot(211)
    plt.plot(time,Vtrace[:,0,:])

    plt.subplot(212)
    plt.plot(time,Vtrace[:,2,:])
    
    
