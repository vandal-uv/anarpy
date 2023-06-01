import numpy as np
from numba import jit, float64, int32, vectorize
from numba.core.errors import NumbaPerformanceWarning
import warnings

warnings.simplefilter('ignore', category=NumbaPerformanceWarning)

@jit(nopython=True)
def S(x, mu, sigma):
    return (1 / (1 + np.exp(-(x - mu) / sigma)))

class WilsonCowan:
    def __init__(self):
        # Node parameters
        # Any of them can be redefined as a vector of length nnodes
        # excitatory connections
        self.a_ee = 3.5
        self.a_ei = 2.5
        # inhibitory connections
        self.a_ie = 3.75
        self.a_ii = 0
        # tau
        self.tauE = 0.010  # Units: seconds
        self.tauI = 0.020  # Units: seconds
        # external input
        self.P = 0.4 
        self.Q = 0
        # inhibitory plasticity
        self.rE = 0.5
        self.rI = 0.5
        self.mu = 1.0
        self.sigma = 0.25
        self.D = 0
        # initialization variable value, it will get overwritten
        self.sqdtD = 0.1
        # network parameters
        self.N = 50
        self.G = 0.001
        self.CM = np.random.binomial(1, 0.1, (self.N, self.N)).astype(np.float64)
        # simulation parameters
        self.E0=0.1
        self.I0=0.1
        # Time units are seconds
        self.tTrans=2
        self.tstop=100
        self.dt=0.001       #interval for points storage
        self.dtSim=0.0001   #interval for simulation (ODE integration)
        self.downsamp=int(self.dt/self.dtSim)

    
    @staticmethod
    @jit(float64[:, :](float64, float64[:, :], float64[:, :], float64, int32, float64, \
                float64, float64, float64, float64, float64, float64, float64, float64, float64, float64, float64), nopython=True)
    def wilsonCowan(t, X, CM, sqdtD, N, G, P, tauE, tauI, a_ee, a_ei, a_ie, a_ii, rE, rI, mu, sigma):
        E, I = X
        noise = np.random.normal(0, sqdtD, size=N)
        return np.vstack(((-E + (1 - rE * E) * S(a_ee * E - a_ei * I + G * np.dot(CM, E) + P + noise, mu, sigma)) / tauE,
                    (-I + (1 - rI * I) * S(a_ie * E - a_ii * I, mu, sigma)) / tauI))

    def wilsonCowanCall(self, t, X):
        return self.wilsonCowan(t, X, self.CM, self.sqdtD, self.N, self.G, self.P, self.tauE, self.tauI, 
                                self.a_ee, self.a_ei, self.a_ie, self.a_ii, self.rE, self.rI, self.mu, self.sigma)
    

    @staticmethod
    @jit(float64[:, :](float64, float64[:, :], float64[:, :], float64, float64, float64, float64, float64, float64, float64, float64[:], \
                       float64, float64, float64, float64), nopython=True)
    def wilsonCowanDet(t, X, CM, G, a_ee, a_ei, a_ie, a_ii, tauE, tauI, P, rE, rI, mu, sigma):
        E, I = X
        return np.vstack(((-E + (1 - rE * E) * S(a_ee * E - a_ei * I + G * np.dot(CM,E) + P, mu, sigma)) / tauE,
                        (-I + (1 - rI * I) * S(a_ie * E - a_ii * I, mu, sigma)) / tauI))

    def wilsonCowanDetCall(self, t, X):
        return self.wilsonCowanDet(t, X, self.CM, self.G, self.a_ee, self.a_ei, self.a_ie, self.a_ii, self.tauE, self.tauI, 
                                   self.P, self.rE, self.rI, self.mu, self.sigma)


    def sim_adapt(self, Init=None):
        """
        Runs a simulation of timeTrans.
        """
        if Init is None:
            Var = np.array([self.E0, self.I0])[:, None] * np.ones((1,self.N))
        else:
            Var = Init
        # Generate the vector again in case variables have changed
        timeTrans = np.arange(0, self.tTrans, self.dtSim)

        if self.D == 0:
            for _, t in enumerate(timeTrans):
                Var += self.dtSim * self.wilsonCowanDetCall(t, Var)
        else:
            sqdtD = self.D/np.sqrt(self.dtSim)
            for _, t in enumerate(timeTrans):
                Var += self.dtSim * self.wilsonCowanCall(t, Var)

        return Var

    def sim(self, Var0=None, verbose=False):
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
        if self.CM.shape[0] != self.CM.shape[1] or self.CM.shape[0] != self.N:
            raise ValueError("check CM dimensions (", self.CM.shape, ") and number of nodes (", self.N, ")")

        if self.CM.dtype is not np.dtype('float64'):
            try:
                self.CM = self.CM.astype(np.float64)
            except:
                raise TypeError("CM must be of numeric type, preferred float")

        if Var0 is not None:
            if len(Var0.shape) == 1:
                Var = Var0 * np.ones((1, self.N))
            else:
                Var = Var0
        else:
            Var = self.sim_adapt()

        timeSim = np.arange(0, self.tstop, self.dtSim)
        time = np.arange(0, self.tstop, self.dt)
        self.downsamp = int(self.dt / self.dtSim)

        Y_t = np.zeros((len(time), Var.shape[0], self.N))  # save data vector

        if verbose:
            print("Simulating %g s dt=%g, Total %d steps" % (self.tstop, self.dtSim, len(timeSim)))

        if verbose and self.D == 0:
            for i, t in enumerate(timeSim):
                if i % self.downsamp == 0:
                    Y_t[i // self.downsamp] = Var
                if t % 10 == 0:
                    print("%g of %g s" % (t, self.tstop))
                Var += self.dtSim * self.wilsonCowanDetCall(t, Var)

        if verbose and self.D > 0:
            self.sqdtD = self.D / np.sqrt(self.dtSim)
            for i, t in enumerate(timeSim):
                if i % self.downsamp == 0:
                    Y_t[i // self.downsamp] = Var
                if t % 10 == 0:
                    print("%g of %g s" % (t, self.tstop))
                Var += self.dtSim * self.wilsonCowanCall(t, Var)

        if not verbose and self.D == 0:
            for i, t in enumerate(timeSim):
                if i % self.downsamp == 0:
                    Y_t[i // self.downsamp] = Var
                Var += self.dtSim * self.wilsonCowanDetCall(t, Var)

        if not verbose and self.D > 0:
            self.sqdtD = self.D / np.sqrt(self.dtSim)
            for i, t in enumerate(timeSim):
                if i % self.downsamp == 0:
                    Y_t[i // self.downsamp] = Var
                Var += self.dtSim * self.wilsonCowanCall(t, Var)

        return Y_t, time

    def params_node(self):
        # Returns a dictionary with the parameters of the model
        pardict = {
            'a_ee': self.a_ee,
            'a_ie': self.a_ie,
            'a_ei': self.a_ei,
            'a_ii': self.a_ii,
            'tauE': self.tauE,
            'tauI': self.tauI,
            'P': self.P,
            'Q': self.Q,
            'rE': self.rE,
            'rI': self.rI,
            'mu': self.mu,
            'sigma': self.sigma
        }
        return pardict

    def params_net(self):
        # Returns a dictionary with the network parameters of the model
        pardict = {
            'N': self.N,
            'G': self.G,
            'CM': self.CM
        }
        return pardict

    def params_sim(self):
        # Returns a dictionary with the simulation parameters of the model
        pardict = {
            'tTrans': self.tTrans,
            'tstop': self.tstop,
            'dt': self.dt,
            'dtSim': self.dtSim
        }
        return pardict

    def params(self):
        # Returns a dictionary with all the parameters of the model
        pardict = self.params_node()
        pardict.update(self.params_net())
        pardict.update(self.params_sim())
        return pardict

    def set_params(self, pardict):
        # Loads a dictionary and sets attributes according to keys
        for key in pardict:
            setattr(self, key, pardict[key])

    def set_params_from_file(self, fname):
        # Loads a dictionary from a file and sets attributes according to keys
        pardict = np.load(fname, allow_pickle=True).item()
        self.set_params(pardict)

    def save_params(self, fname):
        # Saves a dictionary with the parameters of the model
        pardict = self.params()
        np.save(fname, pardict)
    

if __name__=="__main__":
    import matplotlib.pyplot as plt

    wc = WilsonCowan()

    wc.tTrans = 25
    wc.tstop = 10
    wc.G = 0.005
    wc.D = 0

    wc.N = 20
    wc.CM = np.random.binomial(1, 0.3, (wc.N, wc.N)).astype(np.float32)
    wc.P = np.random.uniform(0.34, 0.5, wc.N)

    Vtrace, time = wc.sim(verbose=True)
    print(wc.params_node())
    plt.figure(1)
    plt.clf()
    plt.subplot(211)
    plt.plot(time,Vtrace[:,0,:])

    plt.subplot(212)
    plt.plot(time,Vtrace[:,1,:])
    plt.savefig('test.png')