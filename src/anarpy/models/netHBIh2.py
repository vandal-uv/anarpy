# -*- coding: utf-8 -*-
"""
Created on Tue Dec 22 14:56:13 2015
The Huber_braun neuronal model function
@author: porio
"""

import numpy as np
from numba import jit,float64

# PARAMETERS -- Must be defined first to allow numba compile
# Node parameters
# Any of them can be redefined as a vector of length nnodes
gd = 2.5; gr = 2.8; gsd = 0.21; gsr = 0.28;
gl = 0.06; gh = 0.4;
V0d = -25.; V0r = -25.; zd = 0.25; zr = 0.25;tr = 2.;
V0sd = -40.; zsd = 0.11; tsd = 10.;
eta = 0.014; kappa = 0.18; tsr = 35.;
V0h= -85.; zh = -0.14; th=125.;
Ed = 50.; Er = -90.; El = -80.; Eh = -30.;
temp=36.

rho = 1.3**((temp-25.)/10)
phi = 3**((temp-25.)/10)

Area=15000
gusd = gusr = 20. #pS
guh = 2.  #pS

Nsd = 10*gsd*Area/gusd
Nsr = 10*gsr*Area/gusr
Nh = 10*gh*Area/guh

# Network parameters
nnodes=50
Ggj=0.01

# Mock connectivity only for the purpose of compilation
CM=np.random.binomial(1,0.1,(nnodes,nnodes))

def InitVars(v=None):
    """
    Calculate initial values of variables, starting from a initial voltage.
    
    Initial voltage can be given as a single value (equal for all nodes), one
    value per node or random.

    Parameters
    ----------
    v : float, 1D ndarray or None, optional.
    
        If float, the same initial values are calculated per each node.
        If array or list (length must be nnodes), a different set of initial
        condition is calculated per node.         
        If None, a random voltage in [-65,60] is calculated for each node.
        The default is None.

    Returns
    -------
    y0 : ndarray, 5 x nnodes
    
        Contains the initial values for the _nnodes_ nodes. The order is
        voltage, asd, asr, ah, ar.

    """
    global rho,phi
    if v is None:
        v=np.random.uniform(low=-65,high=-60,size=nnodes)
    
    #Luego calculamos el valor de las variables a ese voltaje
    ar = 1/(1+np.exp(-zr*(v-V0r)));
    asd = 1/(1+np.exp(-zsd*(v-V0sd)));
    ah= 1/(1+np.exp(-zh*(v-V0h)))
    
    rho = 1.3**((temp-25.)/10)
    phi = 3**((temp-25.)/10)
    asr = -eta*rho*gsd*asd*(v - Ed)/kappa;
    
    #initial values Vector
    y0=np.array([v,asd,asr,ah,ar])
    
    return y0

y=InitVars()
runInt=0.025  # intervalo para resolver la SDE

@jit(float64[:,:](float64[:,:],float64),nopython=True)
def HyB(y,t):
    """
    Calculate the derivatives (deterministic) of the HBIh model at time t.

    Parameters
    ----------
    y : ndarray, 5 x nnodes
        Variables at time t.
    t : float
        Time.

    Returns
    -------
    Det : ndarray, 5 x nnodes
        Derivatives.

    """
    v=y[0,:]
    asd=y[1,:]
    asr=y[2,:]
    ah=y[3,:]
    ar=y[4,:]
    
    ad = 1/(1+np.exp(-zd*(v-V0d)))
    isd = rho*gsd*asd*(v - Ed)
    psr = (asr**2)/(asr**2+0.4**2)
    Imemb=isd + rho*gd*ad*(v - Ed) + rho*(gr*ar + gsr*psr)*(v-Er) \
                + rho*gh*ah*(v - Eh)+ rho*gl*(v - El)
    arinf = 1/(1+np.exp(-zr*(v-V0r)))
    asdinf = 1/(1+np.exp(-zsd*(v-V0sd)))
    ahinf= 1/(1+np.exp(-zh*(v-V0h)))

    Igj = np.sum(CM * Ggj * (np.expand_dims(v,1) - v),-1)

    Det=np.vstack((-Imemb - Igj,
                phi*(asdinf - asd)/tsd,
                phi*(-eta*isd - kappa*asr)/tsr,
                phi*(ahinf-ah)/th,
                phi*(arinf - ar)/tr))
    
    Stoch=np.vstack((rho*gsr*(v-Er)*np.sqrt(psr*(1-psr)/Nsr),
                    np.sqrt((np.abs(asdinf*(1-asd)+asd*(1-asdinf))*phi)/(tsd*Nsd)),
                    np.zeros(nnodes),
                    np.sqrt((np.abs(ahinf*(1-ah)+ah*(1-ahinf))*phi)/(th*Nh)),
                    np.zeros(nnodes))) * np.random.randn(5,nnodes) / np.sqrt(runInt)
    
    Det += Stoch
    
    return Det

@jit(float64[:,:](float64[:,:],float64),nopython=True)
def HyBdet(y,t):
    """
    Calculate the derivatives (stochastic) of the HBIh model at time t.
    
    Noise intensity is given by the module-wide variables Nsd, Nsr, Nh

    Parameters
    ----------
    y : ndarray, 5 x nnodes
        Variables at time t.
    t : float
        Time.

    Returns
    -------
    Det : ndarray, 5 x nnodes
        Derivatives.

    """
    v=y[0,:]
    asd=y[1,:]
    asr=y[2,:]
    ah=y[3,:]
    ar=y[4,:]

    ad = 1/(1+np.exp(-zd*(v-V0d)))
    isd = rho*gsd*asd*(v - Ed)
    psr = (asr**2)/(asr**2+0.4**2)
    Imemb=isd + rho*gd*ad*(v - Ed) + rho*(gr*ar + gsr*psr)*(v-Er) \
                + rho*gh*ah*(v - Eh)+ rho*gl*(v - El)
    arinf = 1/(1+np.exp(-zr*(v-V0r)))
    asdinf = 1/(1+np.exp(-zsd*(v-V0sd)))
    ahinf= 1/(1+np.exp(-zh*(v-V0h)))

    Igj = np.sum(CM * Ggj * (np.expand_dims(v,1) - v),-1)

    Det=np.vstack((-Imemb - Igj,
                phi*(asdinf - asd)/tsd,
                phi*(-eta*isd - kappa*asr)/tsr,
                phi*(ahinf-ah)/th,
                phi*(arinf - ar)/tr))
    
    return Det

"""
The main function is starting from here          
"""



#Simulation parameters
adaptTime=30000  #length (in ms) of adaptation simulation for transient removal
adaptInt=0.05    #time interval (in ms) of adaptation simulation

runTime=202000  # length (in ms) of simulation 
runInt=0.025  # interval (in ms) for solving the ODE/SDE
sampling=0.25  # interval (in ms) for data storage and return


def SimAdapt():
    """
    Rus a deterministic simulation of length HB.adaptTime and dt=HB.adaptInt.
    
    Initial conditions are given by the the function HB.Initvars().
    """
    # calculate initial conditions. We are not giving any argument, thus the
    # ICs will be random
    y=InitVars()
    
    Tadapt=np.arange(0,adaptTime,adaptInt)
    print("Adaptation simulation...  %g ms %d steps"%(adaptTime,len(Tadapt)))
    for t in Tadapt:
        y+=adaptInt*HyBdet(y,t)
    return y

def Sim(y0=None,Stoch=False,verbose=False):
    """
    Simulate the model with the defined parameters.

    Parameters
    ----------
    y0 : float or ndarray of shape (nnodes, ) or (5,nnodes) , optional
        Initial conditions. If float, this value is taken as the initial 
        voltage for all nodes and the rest of variables is calculated as
        the equilibrium at that voltage. If ndarray of single dimension
        (must be length nnodes), each value is given for each node and the 
        rest of variables is calculated at equilibrium. If ndarray of shape
        (5, nnodes), it specifies the whole set of initial conditions.
        If None, random initial voltages in [-65,-60] are drawn for each node.
        The default is None.
    Stoch : Boolean, optional
        If True, the simulation is stochastic. The default is False.
    verbose : Boolean, optional
        If True, messages are printed every 1000 ms of simulation.
        The default is False.

    Raises
    ------
    ValueError
        An initial check is performed to check the size of CM and nnodes.
        If they don't match a ValueError is raised

    Returns
    -------
    Y_t : ndarray
        The calculated variable trajectories. Variables (v, asd, asr, ah, ar) 
        go in the first dimension and time in the last dimension
    T_run
        Vector containing the time values of the trajectories.

    """
    if CM.shape[0]!=CM.shape[1] or CM.shape[0]!=nnodes:
        raise ValueError("check CM dimensions (",CM.shape,") and number of nodes (",nnodes,")")
    
    global Nsd,Nsr,Nh
    
    Nsd = 10*gsd*Area/gusd
    Nsr = 10*gsr*Area/gusr
    Nh = 10*gh*Area/guh
    
    # ********************************************************************
    # *** The following 2 lines are REALLY important before simulation ***
    # *** No parameter can be changed after                            ***
    # ********************************************************************
    HyB.recompile()
    HyBdet.recompile()

    if type(y0)==np.ndarray:
        if len(y0.shape)==1:
            y=InitVars(y0)
        else:
            y=y0
    elif y0 is None:
        y=SimAdapt()
    else:
        y=InitVars(y0*np.ones(nnodes))
         
    s_ratio=int(sampling/runInt)  # sampling ratio
    Trun=np.arange(0,runTime,runInt)
    Y_t=np.zeros((len(Trun)//s_ratio,nnodes))  #Vector para guardar datos

    print("Simulating %g ms dt=%g, Total %d steps"%(runTime,runInt,len(Trun)))

    if verbose and not Stoch:
        for i,t in enumerate(Trun):
            if i%s_ratio==0:
                Y_t[i//s_ratio]=y[0]
            if t%1000==0:
                print("%g of %g ms"%(t,runTime))
            y += runInt*HyBdet(y,t)

    if verbose and Stoch:
        for i,t in enumerate(Trun):
            if i%s_ratio==0:
                Y_t[i//s_ratio]=y[0]
            if t%1000==0:
                print("%g of %g ms"%(t,runTime))
            y += runInt*HyB(y,t)            

    if not verbose and not Stoch:
        for i,t in enumerate(Trun):
            if i%s_ratio==0:
                Y_t[i//s_ratio]=y[0]
            y += runInt*HyBdet(y,t)

    if not verbose and Stoch:
        for i,t in enumerate(Trun):
            if i%s_ratio==0:
                Y_t[i//s_ratio]=y[0]
            y += runInt*HyB(y,t)           
            
    return Y_t,Trun[::s_ratio]
        
def ParamsNode():
    """
    Return current node parameters.

    Returns
    -------
    pardict : Dictionary

    """
    pardict={}
    for var in ('gd','gr','gsd','gsr','gl','gh','V0d','V0r','zd','zr',
                'tr','V0sd','zsd','tsd','eta','kappa','tsr','V0h','zh',
                'th','Ed','Er','El','Eh','temp','Area','gusd','gusr','guh'):
        pardict[var]=eval(var)
        
    return pardict

def ParamsNet():
    """
    Return current network parameters.

    Returns
    -------
    pardict : Dictionary
        Number of nodes, Coupling constant, Connectivity matrix.

    """
    pardict={}
    for var in ('nnodes','Ggj','CM'):
        pardict[var]=eval(var)
        
    return pardict

def ParamsSim():
    """
    Return simulation parameters.

    Returns
    -------
    pardict : Dictionary
        Adaptation and simulation times and intervals.

    """
    pardict={}
    for var in ('adaptTime','adaptInt','runTime','runInt','sampling'):
        pardict[var]=eval(var)
        
    return pardict

#%%
if __name__=="__main__":
    
    import matplotlib.pyplot as plt
    
    adaptTime=1000
    runTime=2000
    Ggj=0.02
    
    nnodes=30
    CM=np.random.binomial(1,0.3,(nnodes,nnodes))
    gsd=np.random.uniform(0.18,0.23,size=nnodes)
    gh=np.random.uniform(0.3,0.6,size=nnodes)
    
    Vtrace,time=Sim(verbose=True,Stoch=True)
    
    plt.figure(1)
    plt.clf()
    plt.plot(time,Vtrace)
    
    
