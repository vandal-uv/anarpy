import matplotlib.pyplot as plt
import numpy as np
from anarpy.classModels.HuberBraun import HuberBraun

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