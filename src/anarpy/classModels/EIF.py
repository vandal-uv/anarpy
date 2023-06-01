import numpy as np
from numba import jit,float64,int64

class EIF:


    def __init__(self):

        # Neuron Parameters
        self.Vr_i = [-65, -62]  # -70
        self.DT_i = [0.8, 3]  # 2
        self.Vth_i = [-52, -42]  # -50
        self.Vreset_i = [-53, -54]  # -58
        self.Vspk_i = 0
        self.gL_i = [4.3, 2.9]  # 6
        self.C_i = [104, 30]  # 200
        self.Ioffset_i = [80, 40]
        self.noise = 2

        # Pre-initialize parameters
        self.parlist = ["Vr", "DT", "Vth",
                        "Vreset", "Vspk", "gL", "C", "Ioffset"]
        self.Vr = None
        self.DT = None
        self.Vth = None
        self.Vreset = None
        self.Vspk = None
        self.gL = None
        self.C = None
        self.Ioffset = None

        self.N_i = [16, 10]  # Number of neurons per neuron type (exc, inh)
        self.Ne = self.N_i[0]
        self.Ni = self.N_i[1]
        self.N = sum(self.N_i)   # Total number

        # Synaptic Parameters
        self.tau_exc = 5
        self.Eexc = 0
        self.tau_inh = 20
        self.Einh = -70
        # Synaptic Weights
        self.WsynE = 0.2
        self.WsynI = 0.3
        # Simulation parameters
        self.tstop = 5000
        self.dt = 0.1

        # Connectivity matrices
        self.CMe = np.zeros((self.N, self.N))
        self.CMi = np.zeros((self.N, self.N))

    def params_neuron(self):
        params = {'Vr_i': self.Vr_i, 'DT_i': self.DT_i, 'Vth_i': self.Vth_i, 'Vreset_i': self.Vreset_i, 'Vspk_i': self.Vspk_i,
                  'gL_i': self.gL_i, 'C_i': self.C_i, 'Ioffset_i': self.Ioffset_i, 'N_i': self.N_i, 'N': self.N}
        return params

    def params_syn(self):
        params = {'tau_exc': self.tau_exc, 'Eexc': self.Eexc, 'tau_inh': self.tau_inh, 'Einh': self.Einh, 'WsynE': self.WsynE, 'WsynI': self.WsynI}
        return params

    def params_sim(self):
        params = {'tstop': self.tstop, 'dt': self.dt}
        return params

    def set_params(self,params):
        for key, value in params.items():
            setattr(self, key, value)

    def init_params(self):
        # Initialize parameters with correct N lenght values
        for par in self.parlist:
            if type(getattr(self, par+"_i")) == list:
                if len(getattr(self, par+"_i")) == 1:
                    setattr(self, par, np.ones(self.N)
                            * getattr(self, par+"_i")[0])
                elif len(getattr(self, par+"_i")) == 2:
                    setattr(self, par, np.concatenate((np.ones(
                        self.N_i[0])*getattr(self, par+"_i")[0], np.ones(self.N_i[1])*getattr(self, par+"_i")[1])))
                else:
                    raise ValueError(
                        "The lenght of the parameter "+par+"_i is not valid")
            else:
                setattr(self, par, getattr(self, par+"_i"))

    @staticmethod
    @jit(float64[:, :](float64[:, :], float64[:], float64[:], float64[:], \
                       float64[:], float64[:], float64[:],int64, int64,int64, int64), nopython=True)
    def LIF(X, I, Vr, DT, Vth, gL, C, tau_exc, Eexc, tau_inh, Einh):
        v, ge, gi = X    # voltage, excitatory g, inhibitory g
        return np.vstack(((-gL*(v-Vr)+DT*np.exp((v-Vth)/DT)-ge*(v-Eexc)-gi*(v-Einh)+I)/C,
                        -ge / tau_exc,
                        -gi / tau_inh))

    def LIFCall(self, X, I):
        """
        Network of Exponential-Integrate-and-Fire

        Calculates the derivatives of voltage and synaptic conductances at a given time.

        Parameters
        ----------
        X : array of float64 of size (3,N)
            N is the number of neurons. Accross the first dimension, the variables
            are Voltage, excitatory conductance, inhibitory conductance.
        I : array of float64, 1-D and length N 
            External current applied to each neuron.

        Returns
        -------
        array of float64 of size (3,N)
            The derivative at time _t_ for each variable.

        """
        return self.LIF(X,I,self.Vr,self.DT,self.Vth,self.gL,self.C,self.tau_exc,self.Eexc,self.tau_inh,self.Einh)


    #the previous function as a method of the class EIF
    def InitVars(self, V0=-50):
        """
        Create a vector of initial variables.

        Initial voltage can be set, synaptic conductances will always be 0.

        Parameters
        ----------
        V0 : float, optional
            Initial voltage. The default is -50.

        Returns
        -------
        X : array of floats, dimension (3,N)
            N is the number of neurons. 
            Accross the first dimension, the variables are Voltage,
            excitatory conductance, inhibitory conductance.

        """
        v = V0*np.ones(self.N)
        ge = np.zeros_like(v)
        gi = np.zeros_like(v)
        X = np.array([v, ge, gi])
        return X

    def InitCM(self, Pee=0.4, Pei=0.6, Pie=0.8, Pii=0.4):
        """
        Calculate a random adjacency Matrix.

        The Matrix will be stored it in the module-wide variables CMe and CMi.
        
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
        Cmees = self.WsynE * np.random.binomial(1, Pee, size=(self.Ne, self.Ne))  # from exc to exc
        Cmies = self.WsynE * np.random.binomial(1, Pie, size=(self.Ni, self.Ne))  # from exc to inh
        Cmeis = self.WsynI * np.random.binomial(1, Pei, size=(self.Ne, self.Ni))  # from inh to exc
        Cmiis = self.WsynI * np.random.binomial(1, Pii, size=(self.Ni, self.Ni))  # from inh to inh
        self.CMe = np.r_[Cmees, Cmies]  # All inputs FROM exc
        self.CMi = np.r_[Cmeis, Cmiis]  # All inputs FROM inh

    def runSim(self, X=None, recordV=False):
        if X is None:
            X = self.InitVars()
        time = np.arange(0, self.tstop, self.dt)
        spikes = []
        sqdt = np.sqrt(self.noise/self.dt)
        
        if recordV:
            X_t = np.zeros((len(time), 3, self.N))
            for i, t in enumerate(time):
                Istim = self.Ioffset+sqdt*np.random.normal(size=self.N)
                #distinta corriente en cada neurona
                X_t[i, :] = X
                X += self.dt*self.LIFCall(X, Istim)
                v = X[0]
                if any(v >= 0):
                    idx = np.where(v >= 0)[0]
                    X[0, idx] = self.Vreset[idx]
                    for idx_i in idx:
                        spikes.append([idx_i, t])
                        if idx_i < self.Ne:
                            X[1] = X[1] + self.CMe[:, idx_i]
                        else:
                            X[2] = X[2] + self.CMi[:, idx_i-self.Ne]
            return spikes, time, X_t
        else:
            for i, t in enumerate(time):
                Istim = self.Ioffset + sqdt*np.random.normal(size=self.N)
                # distinta corriente en cada neurona
                X += self.dt*self.LIF(X, Istim)
                v = X[0]
                if any(v >= 0):
                    idx = np.where(v >= 0)[0]
                    X[0, idx] = self.Vreset[idx]
                    for idx_i in idx:
                        spikes.append([idx_i, t])
                        if idx_i < self.Ne:
                            X[1] = X[1] + self.CMe[:, idx_i]
                        else:
                            X[2] = X[2] + self.CMi[:, idx_i-self.Ne]
            return spikes, time


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    eif = EIF()
    eif.init_params()
    eif.InitCM()
    spikes, time, X_t = eif.runSim(eif.InitVars(),recordV=True)
    print(eif.params_neuron())
    print(eif.params_syn())
    print(eif.params_sim())
    print(eif.InitVars())

    if len(spikes)>0:
        spikes = np.array(spikes)
        neuron, fRates=np.unique(spikes[:,0],return_counts=True)
        fRates = fRates/(eif.tstop/1000)    

    plt.figure(1,figsize=(12,6))
    plt.clf()
    if len(spikes) > 0:
        plt.subplot(231)
        plt.plot(spikes[:, 1],spikes[:,0],'.',ms=1)
        plt.xlim((0, eif.tstop))
    plt.subplot(232)
    plt.plot(time[::10],X_t[::10, 0,:eif.Ne], 'g')
    plt.plot(time[::10],X_t[::10, 0,eif.Ne:], 'r')
    plt.subplot(233)
    plt.plot(time[::10],X_t[::10, 1,:eif.Ne], 'g')
    plt.plot(time[::10],X_t[::10, 1,eif.Ne:], 'r')
    plt.subplot(235)
    plt.plot(time[::10],X_t[::10, 2,:eif.Ne], 'g')
    plt.plot(time[::10],X_t[::10, 2,eif.Ne:], 'r')

    plt.subplot(234)
    maxV = np.max(np.abs(np.c_[eif.CMe, eif.CMi]))
    plt.imshow(np.c_[eif.CMe*1, eif.CMi*-1], cmap='bwr', vmin=-maxV, vmax=maxV)
    plt.colorbar()

    plt.tight_layout()
    plt.savefig('EIF.png')
