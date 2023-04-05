import numpy as np
from numba import jit,float64,int32,int64

class HuberBraun():
    
    def __init__(self):
        # Node parameters
        self.gd = 2.5; self.gr = 2.8; self.gsd = 0.21; self.gsr = 0.28
        self.gl = 0.06; self.gh = 0.4
        self.V0d = -25.; self.V0r = -25.; self.zd = 0.25; self.zr = 0.25;self.tr = 2.
        self.V0sd = -40.; self.zsd = 0.11; self.tsd = 10.
        self.eta = 0.014; self.kappa = 0.18; self.tsr = 35.
        self.V0h= -85.; self.zh = -0.14; self.th = 125.
        self.Ed = 50.; self.Er = -90.; self.El = -80.; self.Eh = -30.
        self.temp = 36.
        self.Area = 15000
        self.gusd = self.gusr = 20. #pS
        self.guh = 2.  #pS

        # Computed parameters
        self.rho = 1.3**((self.temp-25.)/10)
        self.phi = 3**((self.temp-25.)/10)
        self.Nsd = 10*self.gsd*self.Area/self.gusd
        self.Nsr = 10*self.gsr*self.Area/self.gusr
        self.Nh = 10*self.gh*self.Area/self.guh

        # Network parameters
        self.nnodes = 50
        self.Ggj = 0.01
        self.CM=np.random.binomial(1,0.1,(self.nnodes,self.nnodes))


        # Simulation parameters
        self.runInt = 0.025 # SDE interval
        self.adaptTime=30000  #length (in ms) of adaptation simulation for transient removal
        self.adaptInt=0.05    #time interval (in ms) of adaptation simulation
        self.runTime=202000  # length (in ms) of simulation
        self.runInt=0.025  # interval (in ms) for solving the ODE/SDE
        self.sampling=0.25  # interval (in ms) for data storage and return


    #use the attributes of the model instad of global scope variables
    def InitVars(self,v=None):
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
        if v is None:
            v=np.random.uniform(low=-65,high=-60,size=self.nnodes)
        
        #Luego calculamos el valor de las variables a ese voltaje
        ar = 1/(1+np.exp(-self.zr*(v-self.V0r)))
        asd = 1/(1+np.exp(-self.zsd*(v-self.V0sd)))
        ah= 1/(1+np.exp(-self.zh*(v-self.V0h)))
        
        self.rho = 1.3**((self.temp-25.)/10)
        self.phi = 3**((self.temp-25.)/10)
        asr = -self.eta*self.rho*self.gsd*asd*(v - self.Ed)/self.kappa
        
        #initial values Vector
        y0=np.array([v,asd,asr,ah,ar])
        
        return y0
    
    @staticmethod
    @jit(float64[:, :](float64[:, :], float64, float64, float64, float64, float64[:], float64, float64, float64, \
                       float64, float64, float64[:], float64, float64, float64, float64, float64, float64, \
                       float64, float64, float64, float64, float64, float64, float64, float64, int64[:,:], float64, \
                       int32, float64, float64, float64, float64, float64, float64), nopython=True)
    def HyB(y, t, zd, V0d, rho, gsd, Ed, gd, gr, gsr, Er, gh, Eh, gl, El, zr, V0r, zsd, V0sd, tsd, eta, \
            kappa, tsr, phi, tr, Nsr, CM, Ggj, nnodes, th, Nh,zh,V0h,Nsd, runInt):
        v = y[0, :]
        asd = y[1, :]
        asr = y[2, :]
        ah = y[3, :]
        ar = y[4, :]

        ad = 1 / (1 + np.exp(-zd * (v - V0d)))
        isd = rho * gsd * asd * (v - Ed)
        psr = (asr ** 2) / (asr ** 2 + 0.4 ** 2)
        Imemb = isd + rho * gd * ad * (v - Ed) + rho * (gr * ar + gsr * psr) * (v - Er) + rho * gh * ah * (v - Eh) + rho * gl * (v - El)
        arinf = 1 / (1 + np.exp(-zr * (v - V0r)))
        asdinf = 1 / (1 + np.exp(-zsd * (v - V0sd)))
        ahinf = 1 / (1 + np.exp(-zh * (v - V0h)))

        Igj = np.sum(CM * Ggj * (np.expand_dims(v, 1) - v), -1)

        Det = np.vstack((-Imemb - Igj,
                        phi * (asdinf - asd) / tsd,
                        phi * (-eta * isd - kappa * asr) / tsr,
                        phi * (ahinf - ah) / th,
                        phi * (arinf - ar) / tr))

        Stoch = np.vstack((rho * gsr * (v - Er) * np.sqrt(psr * (1 - psr) / Nsr),
                        np.sqrt((np.abs(asdinf * (1 - asd) + asd * (1 - asdinf)) * phi) / (tsd * Nsd)),
                        np.zeros(nnodes),
                        np.sqrt((np.abs(ahinf * (1 - ah) + ah * (1 - ahinf)) * phi) / (th * Nh)),
                        np.zeros(nnodes))) * np.random.randn(5, nnodes) / np.sqrt(runInt)

        Det += Stoch

        return Det
    
    
    def HyBCall(self,y,t):
        return self.HyB(y,t,self.zd,self.V0d,self.rho,self.gsd,self.Ed,self.gd,self.gr,self.gsr,self.Er,
                        self.gh,self.Eh,self.gl,self.El,self.zr,self.V0r,self.zsd,self.V0sd,self.tsd,
                        self.eta,self.kappa,self.tsr,self.phi,self.tr,self.Nsr,self.CM,self.Ggj,self.nnodes,
                        self.th,self.Nh,self.zh,self.V0h,self.Nsd,self.runInt)
    
    @staticmethod
    @jit(float64[:,:](float64[:,:],float64,float64,float64,float64,float64[:],float64,float64,float64,float64,\
                      float64,float64[:],float64,float64,float64,float64,float64,float64,float64,float64,\
                      float64,float64,float64,float64,float64,float64,float64,float64,int64[:,:],float64),nopython=True)
    def HyBdet(y, t, zd, V0d, rho, gsd, Ed, gd, gr, gsr, Er, gh, Eh, gl, El, zr, V0r, zsd, V0sd, tsd, eta, kappa, tsr, phi, th, tr,zh,V0h,CM,Ggj):
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

    def HyBdetCall(self,y,t):
        return self.HyBdet(y,t,self.zd,self.V0d,self.rho,self.gsd,self.Ed,self.gd,self.gr,self.gsr,self.Er,
                        self.gh,self.Eh,self.gl,self.El,self.zr,self.V0r,self.zsd,self.V0sd,self.tsd,
                        self.eta,self.kappa,self.tsr,self.phi,self.th,self.tr,self.zh,self.V0h,self.CM,self.Ggj)
    
    def SimAdapt(self):
        """
        Run a deterministic simulation of length HB.adaptTime and dt=HB.adaptInt.
        
        Initial conditions are given by the the function HB.Initvars().
        """
        # calculate initial conditions. We are not giving any argument, thus the
        # ICs will be random
        y=self.InitVars()
        
        Tadapt=np.arange(0,self.adaptTime,self.adaptInt)
        print("Adaptation simulation...  %g ms %d steps"%(self.adaptTime,len(Tadapt)))
        for t in Tadapt:
            y+=self.adaptInt*self.HyBdetCall(y,t)
        return y
    

    #Rewrite Sim method using class attributes
    def Sim(self,y0=None,Stoch=False,verbose=False):
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
        if self.CM.shape[0]!=self.CM.shape[1] or self.CM.shape[0]!=self.nnodes:
            raise ValueError("check CM dimensions (",self.CM.shape,") and number of nodes (",self.nnodes,")")
        
        
        self.Nsd = 10*self.gsd*self.Area/self.gusd
        self.Nsr = 10*self.gsr*self.Area/self.gusr
        self.Nh = 10*self.gh*self.Area/self.guh

        if type(y0)==np.ndarray:
            if len(y0.shape)==1:
                y=self.InitVars(y0)
            else:
                y=y0
        elif y0 is None:
            y=self.SimAdapt()
        else:         
            y=self.InitVars(y0*np.ones(self.nnodes))
        
        s_ratio=int(self.sampling/self.runInt)  # sampling ratio
        Trun=np.arange(0,self.runTime,self.runInt)
        Y_t=np.zeros((len(Trun)//s_ratio,self.nnodes))  # data vector

        print("Simulating %g ms dt=%g, Total %d steps"%(self.runTime,self.runInt,len(Trun)))

        if verbose and not Stoch:
            for i,t in enumerate(Trun):
                if i%s_ratio==0:
                    Y_t[i//s_ratio]=y[0]
                if t%1000==0:
                    print("%g of %g ms"%(t,self.runTime))
                y += self.runInt*self.HyBdetCall(y,t)
        
        if verbose and Stoch:
            for i,t in enumerate(Trun):
                if i%s_ratio==0:
                    Y_t[i//s_ratio]=y[0]
                if t%1000==0:
                    print("%g of %g ms"%(t,self.runTime))
                y += self.runInt*self.HyBCall(y,t)
        
        if not verbose and not Stoch:
            for i,t in enumerate(Trun):
                if i%s_ratio==0:
                    Y_t[i//s_ratio]=y[0]
                y += self.runInt*self.HyBdetCall(y,t)

        if not verbose and Stoch:
            for i,t in enumerate(Trun):
                if i%s_ratio==0:
                    Y_t[i//s_ratio]=y[0]
                y += self.runInt*self.HyBCall(y,t)

        return Y_t,Trun[::s_ratio]
    
    def params_node(self):
        # returns a dictionary with the node parameters of the model
        return {
            'gd': self.gd,
            'gr': self.gr,
            'gsd': self.gsd,
            'gsr': self.gsr,
            'gl': self.gl,
            'gh': self.gh,
            'V0d': self.V0d,
            'V0r': self.V0r,
            'zd': self.zd,
            'zr': self.zr,
            'tr': self.tr,
            'V0sd': self.V0sd,
            'zsd': self.zsd,
            'tsd': self.tsd,
            'eta': self.eta,
            'kappa': self.kappa,
            'tsr': self.tsr,
            'V0h': self.V0h,
            'zh': self.zh,
            'th': self.th,
            'Ed': self.Ed,
            'Er': self.Er,
            'El': self.El,
            'Eh': self.Eh,
            'temp': self.temp,
            'Area': self.Area,
            'gusd': self.gusd,
            'gusr': self.gusr,
            'guh': self.guh
        }

    def params_net(self):
        # returns a dictionary with the network parameters of the model
        return {
            'CM': self.CM,
            'nnodes': self.nnodes,
            'Ggj': self.Ggj,
        }
    
    def params_sim(self):
        # returns a dictionary with the simulation parameters of the model
        return {
            'runTime': self.runTime,
            'runInt': self.runInt,
            'sampling': self.sampling,
            'adaptTime': self.adaptTime,
            'adaptInt': self.adaptInt
        }

    def params(self):
        # returns a joint dictionary with all the parameters of the model
        return {**self.params_node(), **self.params_net(), **self.params_sim()}
    
    def set_params(self, params):
        # sets the parameters of the model from a dictionary
        for key, value in params.items():
            setattr(self, key, value)

    def params_save(self, filename):
        # saves the parameters of the model in a npy file
        np.save(filename, self.params())
    
    def set_params_from_file(self, filename):
        # sets the parameters of the model from a npy file
        self.set_params(np.load(filename, allow_pickle=True).item())

if __name__=='__main__':
    import matplotlib.pyplot as plt
    # example of use
    model = HuberBraun()
    model.adaptTime=1000
    model.runTime=2000
    model.Ggj=0.02

    nnodes = 30
    model.nnodes=30
    model.CM=np.random.binomial(1.0,0.3,(nnodes,nnodes))
    model.gsd=np.random.uniform(0.18,0.23,size=nnodes)
    model.gh=np.random.uniform(0.3,0.6,size=nnodes)
    
    Vtrace,time=model.Sim(verbose=True,Stoch=True)
    plt.figure(1)
    plt.clf()
    plt.plot(time,Vtrace)
    plt.savefig('Vtrace.png')