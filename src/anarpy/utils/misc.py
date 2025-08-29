# -*- coding: utf-8 -*-
"""
Created on Mon Aug 18 17:46:26 2025

@author: Patricio Orio
"""
import numpy as np

def randomShift(data,N=50,cols=True):
    """
    Random shifting of time series

    Parameters
    ----------
    data : 2D numpy array
        T x S numpy array (S x T if cols == False).
    N : int, optional
        Number of surrogates to generate. The default is 50.
    cols : boolean, optional
        If true, the first dimension of the array is time. The default is True.

    Returns
    -------
    outSeries : numpy array
        N x T x S.  (N x S x T if cols==False)
        N = number of surrogates, T = time points, S = number of series

    """
    if cols:
        data2=data.T
    else:
        data2=np.copy(data)
    D,L=data2.shape
    outSeries=[]
    for i in range(N):
        shifts=np.random.randint(1,L,D)
        serie=[np.r_[d[s:],d[:s]] for d,s in zip(data2,shifts)]
        outSeries.append(serie)
    outSeries=np.array(outSeries)
    if cols:
        outSeries=np.swapaxes(outSeries,1,2)
    return outSeries 
   