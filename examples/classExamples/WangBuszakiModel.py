import matplotlib.pyplot as plt
import numpy as np
from scipy import signal
from anarpy.classModels.WangBuszaki import WangBuszaki


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