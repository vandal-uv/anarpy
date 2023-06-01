import numpy as np
from scipy import signal
from numba import jit,float64,vectorize,int64

#isolated functions
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



class WangBuszaki:
    def __init__(self):

        #Node Parameters
        self.gNa = 35.0; self.gK = 9.0;  self.gL=0.1  #mS/cm^2
        self.ENa = 55.0; self.EK = -90.0; self.EL = -65.0 #mV
        self.phi = 5.0
        self.Iapp = 0; # uA/cm^2, injected current

        #Synaptic Parameters
        self.VsynE = 0; self.VsynI = -80  #reversal potential
        self.tau1E = 3; self.tau2E = 1
        self.tau1I = 4; self.tau2I = 1
        self.theta=-20 #threshold for detecting spikes
        self.mdelay=1.5; self.sdelay = 0.1 #ms synaptic delays, mean and SD

        #NetworkMatrix parameters
        self.mGsynE = 5; self.mGsynI = 200; self.mGsynExt = 3  #mean
        self.sGsynE = 1; self.sGsynI = 10; self.sGsynExt = 1
        self.GsynExt = None; self.GsynE = None; self.GsynI = None

        #Network parameters
        self.Ne=100 #Numero de neuronas excitatorias
        self.Ni=25 #Numero de neuronas inhibitorias
        self.N = self.Ne + self.Ni
        self.Pe=0.3; self.Pi=0.3
        self.iRate = 3.5   #Rate of external input

        #Simulation parameters
        #Total=Trun + equil [ms]
        self.equil=400
        self.Trun=2000
        self.dt = 0.02 #ms


        self.CMe,self.CMi,self.CMext = None,None,None
        self.Ggj=0.001  # not so big gap junction conductance
        self.CMelec=self.Ggj * np.random.binomial(1,0.3,(self.Ni,self.Ni))  #mock electric connectivity


        self.delay = np.random.normal(self.mdelay,self.sdelay,size=self.N)
        self.delay_dt=(self.delay/self.dt).astype(int)


    
    def initVars(self,v=None):
        if v is None:
            v_init=np.random.uniform(-80,-60,size=self.N)
        h=1/(1+betah(v_init)/alphah(v_init))
        n=1/(1+betan(v_init)/alphan(v_init))
        sex=np.zeros_like(v_init)
        sey=np.zeros_like(v_init)
        six=np.zeros_like(v_init)
        siy=np.zeros_like(v_init)
        sexe=np.zeros_like(v_init)
        seye=np.zeros_like(v_init)
        return np.array([v_init,h,n,sex,sey,six,siy,sexe,seye])
        

    def genRandomCM(self, mode='all', AdjMe=None, AdjMi=None):
        if mode not in ('exc','inh','excinh','ext','all'):
            raise ValueError("mode has to be one of ['exc','inh','excinh','ext','all']")
        self.N=self.Ne+self.Ni

        factE = 1000*self.dt*expnorm(self.tau1E,self.tau2E)
        factI = 1000*self.dt*expnorm(self.tau1I,self.tau2I)
        
        if mode in ('exc','excinh','all'):
            GsynE = np.random.normal(self.mGsynE,self.sGsynE,size=(self.N,self.Ne))
            GsynE = GsynE*(GsynE>0)
            if AdjMe is None:
                AdjMe=np.random.binomial(1,self.Pe,size=(self.N,self.Ne))
            elif AdjMe.shape!=(self.N,self.Ne):
                raise ValueError("Check dimensions of AdjMe. It has to be N x Ne")
            self.CMe= AdjMe * GsynE / factE
        
        if mode in ('inh','excinh','all'):
            GsynI = np.random.normal(self.mGsynI,self.sGsynI,size=(self.N,self.Ni))
            GsynI = GsynI*(GsynI>0)
            if AdjMi is None:
                AdjMi=np.random.binomial(1,self.Pi,size=(self.N,self.Ni))
            elif AdjMi.shape!=(self.N,self.Ni):
                raise ValueError("Check dimensions of AdjMe. It has to be N x Ni")
            self.CMi= AdjMi* GsynI / factI
        
        if mode in ('ext','all'):
            GsynExt = np.random.normal(self.mGsynExt,self.sGsynExt,size=self.N)
            GsynExt = GsynExt*(GsynExt>0) / factE
            self.GsynExt=GsynExt

    #                [<'ndarray'>, <'ndarray'>, <'int'>, <'float'>, <'float'>, <'float'>, <'float'>, <'float'>,<'float'>, 
    #                 <'float'>, <'int'>, <'int'>, <'int'>, <'int'>, <'int'>, <'int'>,<'ndarray'>, <'ndarray'>, <'ndarray'>, 
    #<'float'>, <'ndarray'>, <'int'>, <'int'>, <'int'>, <'ndarray'>, <'float'>]
    #                  array(float64, 2d, C), array(int64, 1d, C), int64, float64, float64, float64, float64, float64, float64, 
    #                   float64, int64, int64, int64, int64, int64, int64, array(float64, 2d, C), array(float64, 2d, C), array(int64, 1d, C), 
    #                   float64, array(float64, 1d, C), int64, int64, int64, array(float64, 2d, C), float64
    #@jit(float64[:, :](float64[:, :], int64[:], int64, float64, float64, float64, float64, float64, float64, \
    #                   float64, int64, int64, int64, int64, int64, int64, float64[:,:], float64[:,:], float64[:], \
    #                    float64, float64[:], int64, int64, int64, float64[:,:], float64), nopython=True)
    @staticmethod
    def WB_network(X, ls, i, gNa, ENa, gK, EK, gL, EL, phi, VsynE, VsynI, tau1E, tau2E, tau1I, tau2I, CMe, CMi, \
                   delay_dt, iRate, GsynExt, Iapp, N, Ne, CMelec, dt):
        v = X[0, :]
        h = X[1, :]
        n = X[2, :]
        sex = X[3, :]
        sey = X[4, :]
        six = X[5, :]
        siy = X[6, :]
        sexe = X[7, :]
        seye = X[8, :]

        minf = alpham(v) / (betam(v) + alpham(v))
        INa = gNa * minf**3 * h * (v - ENa)
        IK = gK * n**4 * (v - EK)
        IL = gL * (v - EL)

        ISyn = (sey + seye) * (v - VsynE) + siy * (v - VsynI)
        Igj = np.zeros(N)
        Igj[Ne:] = np.sum(CMelec * (np.expand_dims(v[Ne:], 1) - v[Ne:]), -1)
        firingExt = np.random.binomial(1, iRate*dt, size=N)
        firing = 1. * (ls == (i - delay_dt))

        return np.vstack((-INa - IK - IL - ISyn - Igj + Iapp,
                        phi*(alphah(v)*(1-h) - betah(v)*h),
                        phi*(alphan(v)*(1-n) - betan(v)*n),
                        -sex*(1/tau1E + 1/tau2E) - sey/(tau1E*tau2E) + np.dot(CMe, firing[0:Ne]),
                        sex,
                        -six*(1/tau1I + 1/tau2I) - siy/(tau1I*tau2I) + np.dot(CMi, firing[Ne:]),
                        six,
                        -sexe*(1/tau1E + 1/tau2E) - seye/(tau1I*tau2I) + firingExt*GsynExt,
                        sexe))
    
    def WB_networkCall(self,X,ls,i):
        return self.WB_network(X,ls,i,self.gNa,self.ENa,self.gK,self.EK,self.gL,self.EL,self.phi,self.VsynE,self.VsynI,\
                               self.tau1E,self.tau2E,self.tau1I,self.tau2I,self.CMe,self.CMi,self.delay_dt,self.iRate,\
                               self.GsynExt,self.Iapp,self.N,self.Ne,self.CMelec,self.dt)

    def runSim(self, v_init=None, output='spikes'):
        if v_init is None:
            X=self.initVars()
        elif len(v_init)==self.N:
            X=self.initVars(v_init)
        else:
            raise ValueError("v_init has to be None or an array of length N")
        
        if output not in ('spikes','LFP','allV'):
            raise ValueError("output has to be one of ['spikes','LFP','allV']")

        #adaptation simulation - not stored
        equil_dt=int(self.equil / self.dt)  #equilibrium time - in samples
        bufferl=100*(np.max(self.delay_dt)//100+1)
        V_t=np.zeros((bufferl,self.N))
        lastSpike=equil_dt*np.ones(self.N,dtype=np.int64)
        for i in range(equil_dt):
            ib=i%bufferl
            X+=self.dt*self.WB_networkCall(X,lastSpike,i)
    #        firing=1*(V_t[ib-delay_dt,range(N)]>theta)*(V_t[ib-delay_dt-1,range(N)]<theta)
        
        Time = np.arange(0,self.Trun,self.dt)
        
        if output=='spikes':
            spikes=[]
            bufferl=100*(np.max(self.delay_dt)//100+1)
            V_t=np.zeros((bufferl,self.N))
            lastSpike=lastSpike-equil_dt
            lastSpike[lastSpike==0]=int(self.Trun/self.dt)
            for i,t in enumerate(Time):
                ib=i%bufferl
                V_t[ib]=X[0]
                if np.any((V_t[ib]>self.theta)*(V_t[ib-1]<self.theta)):
                    for idx in np.where((V_t[ib]>self.theta)*(V_t[ib-1]<self.theta))[0]:
                        spikes.append([idx,t])
                        lastSpike[idx]=i
                X+=self.dt*self.WB_networkCall(X,lastSpike,i)
            return np.array(spikes)
        
        elif output=='LFP':
            spikes=[]
            bufferl=100*(np.max(self.delay_dt)//100+1)
            V_t=np.zeros((bufferl,self.N))
            LFP_t=np.zeros(len(Time))
            lastSpike=lastSpike-equil_dt
            lastSpike[lastSpike==0]=int(self.Trun/self.dt)        
            for i,t in enumerate(Time):
                ib=i%bufferl
                V_t[ib]=X[0]
                LFP_t[i]=np.mean(X[0])
                if np.any((V_t[ib]>self.theta)*(V_t[ib-1]<self.theta)):
                    for idx in np.where((V_t[ib]>self.theta)*(V_t[ib-1]<self.theta))[0]:
                        spikes.append([idx,t])
                        lastSpike[idx]=i
                X+=self.dt*self.WB_networkCall(X,lastSpike,i)
            return np.array(spikes),LFP_t,Time
        
        elif output=='allV':
            spikes=[]
            V_t=np.zeros((len(Time),self.N))
            lastSpike=lastSpike-equil_dt
            lastSpike[lastSpike==0]=int(self.Trun/self.dt)        
            for i,t in enumerate(Time):
                V_t[i]=X[0]
                if np.any((V_t[i]>self.theta)*(V_t[i-1]<self.theta)):
                    for idx in np.where((V_t[i]>self.theta)*(V_t[i-1]<self.theta))[0]:
                        spikes.append([idx,t])
                        lastSpike[idx]=i
                X+=self.dt*self.WB_networkCall(X,lastSpike,i)
            return np.array(spikes),V_t,Time

    def params_node(self):
        # returns a dictionary with the node parameters
        return {'gNa':self.gNa, 'ENa':self.ENa, 'gK':self.gK, 'EK':self.EK, 'gL':self.gL, 'EL':self.EL, 'phi':self.phi, 'Iapp':self.Iapp}
    
    def params_syn(self):
        # returns a dictionary with the synapse parameters
        return {'VsynE':self.VsynE, 'VsynI':self.VsynI, 'tau1E':self.tau1E, 'tau2E':self.tau2E, 'tau1I':self.tau1I, 'tau2I':self.tau2I,
                'theta':self.theta, 'mdelay':self.mdelay, 'sdelay':self.sdelay,}
    
    def params_netmatrix(self):
        # returns a dictionary with the network parameters
        return {'mGsynE':self.mGsynE, 'sGsynE':self.sGsynE, 'mGsynI':self.mGsynI, 'sGsynI':self.sGsynI, 'mGsynExt':self.mGsynExt, 'sGsynExt':self.sGsynExt,
                'GsynE':self.GsynE, 'GsynI':self.GsynI, 'GsynExt':self.GsynExt}
    
    def params_net(self):
        # returns a dictionary with the network parameters
        return {'N':self.N, 'Ne':self.Ne, 'Ni':self.Ni, 'Pi':self.Pi, 'iRate':self.iRate}
    
    def params_sim(self):
        # returns a dictionary with the simulation parameters
        return {'dt':self.dt, 'equil':self.equil, 'Trun':self.Trun}
    
    def params(self):
        # returns a dictionary with all the parameters
        return {**self.params_node(), **self.params_syn(), **self.params_netmatrix(), **self.params_net(), **self.params_sim()}
    
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


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    model = WangBuszaki()
    model.Pi = 0.3
    model.iRate = 3.0
    model.genRandomCM()

    model.Ggj = 0.1
    model.CMelec = model.Ggj * np.random.binomial(1,0.3,(model.Ni,model.Ni))
    
    print(model.params_node())
    print(model.params_syn())
    print(model.params_net())
    print(model.params_netmatrix())
    print(model.params_sim())
    spikes = model.runSim()
    
    print(model.CMe.shape)
    print(model.CMi.shape)

    binsize = 0.5 # bin size for population activity in ms
    tbase = np.arange(0,model.Trun, binsize) # raster time base
    
    kernel=signal.gaussian(10*2/binsize+1,2/binsize)
    kernel/=np.sum(kernel)
    
    pop_spikes = spikes[:,1]
    popact,binedge = np.histogram(pop_spikes, tbase)
    
    conv_popact=np.convolve(popact,kernel,mode='same')
    xlims = [900,1100]
    
    fig=plt.figure(2,figsize=(12,8), dpi= 80, facecolor='w', edgecolor='k') # tamaño, resolucion, color de fondo y borde de la figura
    plt.clf()
    
    plt.subplot(343)
    plt.plot(spikes[:,1],spikes[:,0],'k.',ms=1)
    
    plt.subplot(344)
    plt.plot(popact)
    plt.plot(conv_popact)

    plt.subplot(347)
    #for i in range(N):
    #    plt.plot(dt*np.array(spikes[i]),i*np.ones_like(spikes[i]),'.')
    plt.plot(spikes[:,1],spikes[:,0],'k.',ms=1)
    
    plt.xlim(xlims) 
    
    plt.subplot(348)
    plt.plot(popact)
    plt.plot(conv_popact)
    plt.xlim(xlims) 

    
    plt.subplot(3,4,12)
    
    lags,c,_,_=plt.acorr(conv_popact,maxlags=500,usevlines=False,linestyle='-',ms=0)
    peaks=lags[np.where(np.diff(1*(np.diff(c)>0))==-1)[0]+1]
    
    if len(peaks)>1:
        netwFreq=1000/(binsize*peaks[peaks>0][0])
    else:
        netwFreq=np.nan

    plt.tight_layout()
    plt.savefig('WangBuszaki.png',dpi=300)