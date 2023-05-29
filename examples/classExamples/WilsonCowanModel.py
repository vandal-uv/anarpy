import matplotlib.pyplot as plt
import numpy as np
from anarpy.classModels.WilsonCowan import WilsonCowan

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