import numpy as np
from numba import jit, float64, int32, vectorize
from numba.core.errors import NumbaPerformanceWarning
from WilsonCowan import WilsonCowan
import warnings

warnings.simplefilter('ignore', category=NumbaPerformanceWarning)

@jit(nopython=True)
def S(x, mu, sigma):
    return (1 / (1 + np.exp(-(x - mu) / sigma)))

class WilsonCowanPlastic(WilsonCowan):
    def __init__(self):
        super().__init__()
        # Plasticity parameters
        ## Inhibitory plasticity
        self.rhoE = 0.14 # target mean value for E
        self.tau_ip = 2  #time constant for plasticity
        self.a_ei_0 = 2.5

    @staticmethod
    @jit(float64[:, :](float64, float64[:, :], float64[:, :], float64, int32, float64, float64, float64, \
                float64, float64, float64, float64, float64, float64, float64, float64, float64, float64, float64), nopython=True)
    def wilsonCowan(t, X, CM, sqdtD, N, G, P, tauE, tauI, a_ee, a_ei, a_ie, a_ii, rE, rI, rhoE, tau_ip, mu, sigma):
        E, I, a_ei = X
        noise = np.random.normal(0, sqdtD, size=N)
        return np.vstack(((-E + (1 - rE * E) * S(a_ee * E - a_ei * I + G * np.dot(CM, E) + P + noise, mu, sigma)) / tauE,
                    (-I + (1 - rI * I) * S(a_ie * E - a_ii * I, mu, sigma)) / tauI,
                    (I*(E-rhoE))/tau_ip))

    def wilsonCowanCall(self, t, X):
        return self.wilsonCowan(t, X, self.CM, self.sqdtD, self.N, self.G, self.P, self.tauE, self.tauI, 
                                self.a_ee, self.a_ei, self.a_ie, self.a_ii, self.rE, self.rI, self.rhoE, self.tau_ip, self.mu, self.sigma)
    

    @staticmethod
    @jit(float64[:, :](float64, float64[:, :], float64[:, :], float64, float64, float64, float64, float64, float64, float64, float64[:], \
                       float64, float64, float64, float64, float64, float64), nopython=True)
    def wilsonCowanDet(t, X, CM, G, a_ee, a_ei, a_ie, a_ii, tauE, tauI, P, rE, rI, rhoE, tau_ip, mu, sigma):
        E, I, a_ei = X
        return np.vstack(((-E + (1 - rE * E) * S(a_ee * E - a_ei * I + G * np.dot(CM, E) + P, mu, sigma)) / tauE,
                        (-I + (1 - rI * I) * S(a_ie * E - a_ii * I, mu, sigma)) / tauI,
                        (I*(E-rhoE))/tau_ip))

    def wilsonCowanDetCall(self, t, X):
        return self.wilsonCowanDet(t, X, self.CM, self.G, self.a_ee, self.a_ei, self.a_ie, self.a_ii, self.tauE, self.tauI, 
                                   self.P, self.rE, self.rI, self.rhoE, self.tau_ip, self.mu, self.sigma)

    def sim_adapt(self, Init=None):
        """
        Runs a deterministic simulation of timeTrans.
        """
        if Init is None:
            Var = np.array([self.E0, self.I0, self.a_ei_0])[:, None] * np.ones((1,self.N))
        else:
            Var = Init
        # Generate the vector again in case variables have changed
        timeTrans = np.arange(0, self.tTrans, self.dtSim)

        for _, t in enumerate(timeTrans):
            Var += self.dtSim * self.wilsonCowanDetCall(t, Var)

        return Var   

if __name__=="__main__":
    import matplotlib.pyplot as plt

    wc = WilsonCowanPlastic()

    wc.tTrans = 25
    wc.tstop = 10
    wc.G = 0.005
    wc.D = 0

    wc.N = 20
    wc.CM = np.random.binomial(1, 0.3, (wc.N, wc.N)).astype(np.float32)
    wc.P = np.random.uniform(0.34, 0.5, wc.N)
    Init = np.array([wc.E0,wc.I0,wc.a_ei_0])[:,None]*np.ones((1,wc.N))

    Vtrace, time = wc.sim(verbose=True)
    print(wc.params_node())
    plt.figure(1)
    plt.clf()
    plt.subplot(211)
    plt.plot(time,Vtrace[:,0,:])

    plt.subplot(212)
    plt.plot(time,Vtrace[:,1,:])
    plt.savefig('testPlastic.png')