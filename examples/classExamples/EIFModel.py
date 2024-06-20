import matplotlib.pyplot as plt
import numpy as np
from anarpy.classModels.EIF import EIF

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
