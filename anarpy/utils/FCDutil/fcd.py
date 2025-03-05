# -*- coding: utf-8 -*-
"""
Created on Wed Apr 27 15:57:38 2016

@author: jmaidana
@author: porio
"""
from __future__ import division
import numpy as np
#import seaborn as sns ## is used to plot some of the results
#import pandas as pd ## pandas has a function to make the correlation between multiple time series
import matplotlib.pylab as plt
from numba import njit

from numpy import linalg as LA

@njit
def phaseFC(phase_T, cols=True):
    """
    Calculate a FC matrix based on phase synchrony

    Parameters
    ----------
    phase_T : numpy array of floats in (0,1)
        Phase in time for each node. If cols==True, the first axis (0) is time
        and the second axis contains the nodes.
    cols : Boolean, optional
        The default is True.

    Returns
    -------
    FCphase : 2D numpy array
        NxN matrix containing the pair-wise phase synch values. N is the number 
        of nodes (2nd dimension of phase_t if cols==True)

    """
    if cols:
        # if series are in columns (time in rows)
        phase_T=phase_T.T

    nnodes=phase_T.shape[0]
    
    FCphase=np.zeros((nnodes,nnodes))
    for ii in range(nnodes):
        for jj in range(ii):
            FCphase[ii,jj]=np.mean(np.abs((np.exp(1j*phase_T[ii])+np.exp(1j*phase_T[jj]))/2))
    FCphase = FCphase + FCphase.T + np.identity(nnodes)
    return FCphase

def ccorrcoef(alpha1, alpha2, axis=None):
    """
    Circular correlation coefficient

    Parameters
    ----------
    alpha1 : ndarray
        Angles or phases of first time series.
    alpha2 : ndarry
        Angles or phases of second time series.
    axis : integer, optional
        DESCRIPTION. The default is None.

    Returns
    -------
    rho : float
        The circular correlation coefficient.

    """
    
    if axis is not None and alpha1.shape[axis] != alpha2.shape[axis]:
        raise(ValueError, "shape mismatch")
    # compute mean directions
    if axis is None:
        n = alpha1.size #toma el largo de la matriz
    else:
        n = alpha1.shape[axis]#En caso de operar sobre fila o columna
    #################################################################
    c1 = np.cos(alpha1)
    c1_2 = np.cos(2*alpha1)
    c2 = np.cos(alpha2)
    c2_2 = np.cos(2*alpha2)
    s1 = np.sin(alpha1)
    s1_2 = np.sin(2*alpha1)
    s2 = np.sin(alpha2)
    s2_2 = np.sin(2*alpha2)

    sumfunc = lambda x: np.sum(x, axis=axis)
    num = 4 * (sumfunc(c1*c2) * sumfunc(s1*s2) -
               sumfunc(c1*s2) * sumfunc(s1*c2))
    
    den = np.sqrt((n**2 - sumfunc(c1_2)**2 - sumfunc(s1_2)**2) *
                  (n**2 - sumfunc(c2_2)**2 - sumfunc(s2_2)**2))

    rho = num / den

    return rho


def phaseScramble(data,Nsurr=10):
    len_d=len(data)
    fftdata=np.fft.fft(data)
    angles=np.angle(fftdata)
    amplitudes=np.abs(fftdata)
    
    surrAngles=np.random.uniform(low=-np.pi,high=np.pi,size=(Nsurr,len(angles)))
    surrAngles[:,1:len_d//2]=surrAngles[:,-1:len_d//2:-1]
    surrAngles[:,len_d//2]=0
    
    fftSurr=amplitudes*(np.cos(surrAngles) + 1j*np.sin(surrAngles))
    surrData=np.fft.ifft(fftSurr,axis=-1)
    
    return surrData


def extract_FCD(data,wwidth=1000,maxNwindows=100,olap=0.9,coldata=False,
                mode='corr',modeFCD='corr',LEV=False):
    """
    Performs FC and FCD computation from a groups of time series
    
    Parameters
    ----------
    
    data : array
        Collection of time series. If coldata==False (default) rows 
        (first dimension) are nodes and columns are time points.
        
    wwidth : int
        Length of sliding windows, in data points.
        
    maxNwindows : int
        Maximum number of sliding windows to compute. If needed, wwidth will be
        increased to enforce this number. olap will be respected.
        
    olap : float. (0<= olap < 1)
        Overlap between consecutive sliding windows, as a fraction of wwidth.
        
    coldata : Boolean
        If True, data is interpreted as time points in the rows and nodes in
        the columns. Default: False
        
    mode : Measure to be employed within each window to calculate the FCs
        "corr" : Pearson correlation.
        "tdcorr" : Time-delayed correlation. Maximum of correlation with varying 
        time delay, to a maximum of wwidth/2.
        "circcorr" : Circular Correlation Coeficient.
        "psync" : Phase synchrony. Needs the data to come as instantaeous phase.
        "pcoher" : Phase coherence or PLV. Needs the data to come as instantaeous phase.
        "clark" : Angular distance between the vectors defined by the elements
        of the time series inside the window.
    
    modeFCD : Measure to be employed to compare the FCs and build the FCD
        "corr" : Pearson correlation between FCs. Note that this is a similarity measure, not distance.  
        "angdist" : Angular distance, defined as the cosine between vectors.  
        "clarksondist" : Clarkson's angular distance.  
        "euclidean" : Euclidean (2-norm) distance betwen unfolded FCs.
        
    Returns
    -------
    
    FCD : MxM array
        FCD matrix. M is the number of sliding windows (FCs)
    FCs : MxN array
        Matrix containing the unfolded FCs. M is the number of FCs and N is 
        the number of node pairs. Note that N = n(n-1)/2, where n is the number
        of nodes
    shift : int
        Number of points between the beggining of consecutive FCs.
     
    """
        
    if olap>=1:
        raise ValueError("olap must be lower than 1")
    if coldata:
        data=data.T    
    
    all_corr_matrix = []
    lenseries=len(data[0])
    Nwindows=min(((lenseries-wwidth*olap)//int(wwidth*(1-olap)),maxNwindows))
    shift=int((lenseries-wwidth)//(Nwindows-1))
    if Nwindows==maxNwindows:
        wwidth=int(shift//(1-olap))
    
    indx_start = range(0,(lenseries-wwidth+1),shift)
    indx_stop = range(wwidth,(1+lenseries),shift)
         
    nnodes=len(data)
    #    mat_ones=np.tril(-10*np.ones((n_ts,n_ts))) #this is a condition to then eliminate the diagonal
    for j1,j2 in zip(indx_start,indx_stop):
        aux_s = data[:,j1:j2]
        if mode=='corr':
            corr_mat = np.corrcoef(aux_s) 
        elif mode=='psync':
            # corr_mat=np.mean(np.abs((np.exp(1j*aux_s[:,None,:])+np.exp(1j*aux_s[None,:,:]))/2),-1)
            corr_mat=phaseFC(aux_s, cols=False)
        elif mode=='pcoher': #PLV phase locking value
            corr_mat=np.zeros((nnodes,nnodes))
            for ii in range(nnodes):
                for jj in range(ii):
                    corr_mat[ii,jj]=np.abs(np.mean(np.exp(1j*np.diff(aux_s[[ii,jj],:],axis=0))))
        elif mode=='tdcorr': #time-delayed correlation
            corr_mat=np.zeros((nnodes,nnodes))
            maxlags = wwidth//2
            for ii in range(nnodes):
                for jj in range(ii):
                    x = aux_s[ii,:] - np.mean(aux_s[ii,:])
                    y = aux_s[jj,:] - np.mean(aux_s[jj,:])
                    correls = np.correlate(x, y, mode='full')
                    correls = correls[wwidth - 1 - maxlags:wwidth + maxlags]
                    maxCorr=np.max(correls)
                    corr_mat[ii,jj]=maxCorr/np.sqrt(np.dot(aux_s[ii,:],aux_s[ii,:])*np.dot(aux_s[jj,:],aux_s[jj,:]))
        elif mode == 'circcorr':
            corr_mat=np.zeros((nnodes,nnodes))
            for ii in range(nnodes):
                for jj in range(ii):   
                    corr_mat[ii,jj] = ccorrcoef(aux_s[ii,:],aux_s[jj,:])
        elif mode == 'clark':#clarkson para las FC
            corr_mat=np.zeros((nnodes,nnodes))
            for ii in range(nnodes):
                for jj in range(ii):
                    corr_mat[ii,jj]=LA.norm(aux_s[ii,:]/LA.norm(aux_s[ii,:]) - aux_s[jj,:]/LA.norm(aux_s[jj,:]))  

        all_corr_matrix.append(corr_mat)
        
    corr_vectors=np.array([allPm[np.tril_indices(nnodes,k=-1)] for allPm in all_corr_matrix])
    L = np.shape(corr_vectors)[0]
    FCD = np.zeros((L,L))

    if modeFCD == 'corr':
        CV_centered=corr_vectors - np.mean(corr_vectors,-1)[:,None]
        FCD = np.abs(np.corrcoef(CV_centered))
    elif modeFCD == 'angdist':#angular distance
        for ii in range(L):
            for jj in range(ii):
                FCD[ii,jj]=np.arccos((np.dot(corr_vectors[ii,:],corr_vectors[jj,:]))/(LA.norm(corr_vectors[ii,:])*LA.norm(corr_vectors[jj,:])))/np.pi  
                FCD[jj,ii]=FCD[ii,jj]
    elif modeFCD == 'clarksondist':#angular distance by clarkson
        for ii in range(L):
            for jj in range(ii):
                FCD[ii,jj]= LA.norm(corr_vectors[ii,:]/LA.norm(corr_vectors[ii,:]) - corr_vectors[jj,:]/LA.norm(corr_vectors[jj,:]))  
                FCD[jj,ii]=FCD[ii,jj]
    elif modeFCD == 'euclidean':
        for ii in range(L):
            for jj in range(ii):
                FCD[ii,jj]=LA.norm(corr_vectors[ii,:]-corr_vectors[jj,:])
                FCD[jj,ii]=FCD[ii,jj]
           
    
    #return CV_centered,corr_vectors,shift
    return FCD,corr_vectors,shift


def plotBig(FCD,FCs,PStot,phasesynch,Trun=None,minmax=[0,1],CM=None,
              cmap=None,saveFig=False,fileName='FCD',fcdMAX=None):
    """
    Parameters
    ----------
    
    FCD : MxM array
        FCD matrix
    
    FCs : array
        MxN Matrix containing FCs. M (rows) = FCs. N (columns) = node pairs. 
        Note that N = n(n-1)/2 where n is the number of nodes.
        
    PStot : array
        Matrix with the time average of the pair-wise synchronies.
    
    phasesynch :
        Vector containing the network (global) phase synch in time.
        
    Trun : float
        Total time of the simulation, for plotting purposes. If not given or None,
        the number of FCs is taken as total time.

    minmax : list or tuple
        minimum and maximum value for FC plots
        
    CM : Connectivity matrix
        If given, it will be plotted
        
    cmap : None
        colormap for FC plots. If None, either 'seismic' or 'jet' is automatically selected.

    saveFig : boolean
        Whether or not save the Figure.
        
    fileName : string
        Basename to save Figure. "-Plots.png" will be appended.
        
    fcdMAX : float or none
        Maximum value to normalize the color scale of the FCD plot. If None, the max
        is automatic.
        
    """
    
    if len(minmax)!=2:
        raise ValueError("minmax must be a list with two values")
    if cmap==None:
        if minmax[0]==0:
            cmap='jet'
        else:
            cmap='seismic'
            
    L = FCs.shape[-1]
    nnodes = int(np.sqrt(8*L+1)+1)//2
    
    if Trun==None:
        Trun=FCs.shape[0]
       
    time=np.linspace(0,Trun,len(phasesynch))
    
    plt.figure(104,figsize=(10,8))
    plt.clf()
        
    plt.subplot2grid((5,5),(0,0),rowspan=2,colspan=5)
    plt.plot(time,phasesynch)
    plt.title('mean P sync')
        
    plt.subplot2grid((5,5),(2,0),rowspan=2,colspan=2)
    plt.imshow(FCD,vmin=0,vmax=fcdMAX,extent=(0,Trun,Trun,0),interpolation='none',cmap='jet')
    plt.title('Sync FCD')
    plt.colorbar()
    plt.grid()
       
    plt.subplot2grid((5,5),(2,2),rowspan=2,colspan=2)
    plt.imshow(PStot,cmap='jet',vmax=1,vmin=0,interpolation='none')
    plt.gca().set_xticks(())
    plt.gca().set_yticks(())
    plt.title('Static FC')
    plt.colorbar()
    plt.grid()
        
    ax=plt.subplot2grid((5,5),(2,4))
    FCDvals=FCD[np.tril_indices(len(FCs),k=-4)]
    ax.hist(FCDvals,range=(0,0.6),color='C1')
    ax.text(0.5,0.97,'%.4g'%np.var(FCDvals),transform=ax.transAxes,ha='center',va='top',fontsize='small')
        
    if type(CM)==np.ndarray:
        CMsh=CM.shape
        if len(CMsh)==2 and CMsh[0]==CMsh[1]:
            plt.subplot2grid((5,5),(2,4))
            plt.imshow(CM,cmap='gray_r')
    
    windows=[int(len(FCs)*f) for f in (0.18, 0.36, 0.54, 0.72, 0.9)]
    axes2=[plt.subplot2grid((5,5),pos) for pos in ((4,0),(4,1),(4,2),(4,3),(4,4))]
    for axi,ind in zip(axes2,windows):
        corrMat=np.zeros((nnodes,nnodes))
        corrMat[np.tril_indices(nnodes,k=-1)]=FCs[ind]
        corrMat+=corrMat.T
        corrMat+=np.eye(nnodes)
            
        axi.imshow(corrMat,vmin=minmax[0],vmax=minmax[1],interpolation='none',cmap=cmap)
            
        axi.set_xticks(())
        axi.set_yticks(())
        
        axi.set_title('t=%.4g'%(ind*Trun/len(FCs)))
        axi.grid()
        
    plt.tight_layout(pad=0.5,w_pad=0.5,h_pad=0.5)
    
#    plt.show()
    
    if saveFig:    
        plt.savefig(fileName+"-Plots.png",dpi=200)
        

    
#%%


def plotFC(FCs,interval=5,fig=None,saveFig=False,fileName='FCD',
           minmax=[0,1],cmap=None,rows=None,deltaT=1):
    """
    Plots some FCs, given an interval
    
    Parameters
    ----------
    
    FCs : array
        MxN Matrix containing FCs. M (rows) = FCs. N (columns) = node pairs. 
        Note that N = n(n-1)/2 where n is the number of nodes.
        
    interval : integer
        Interval between plotted FCs
    
    fig : figure instance
        If given, use the existing figure. If not given or None, a new figure
        will be created (100)
        
    saveFig : boolean
        Whether or not save the Figure.
        
    fileName : string
        Basename to save Figure. "-Plots.png" will be appended.

    minmax : list or tuple
        minimum and maximum value for FC plots
        
    cmap : None
        colormap for FC plots. If None, either 'seismic' or 'jet' is automatically selected.

    """
    if fig==None:
        fig=plt.figure(100)
    fig.clf()
    if len(minmax)!=2:
        raise ValueError("minmax must be a list with two values")
    if cmap==None:
        if minmax[0]==0:
            cmap='jet'
        else:
            cmap='seismic'
        
    L = FCs.shape[-1]
    nnodes = int(np.sqrt(8*L+1)+1)//2

    IndexFc = np.arange(0,(FCs.shape[0]),interval)
    ColFc = 10
    RowFc = int(IndexFc.shape[0]/ColFc)+1
    if rows!=None:
        RowFc=max(RowFc,rows)    
    
    for k,n in enumerate(IndexFc):
        axi = fig.add_subplot(RowFc,ColFc,k+1)
        FCt = np.zeros((nnodes,nnodes))
        FCt[np.tril_indices(nnodes,k=-1)] = FCs[n]
        FCreconst = FCt+FCt.T
        axi.imshow(FCreconst,cmap=cmap,vmin=minmax[0],vmax=minmax[1])
        axi.set_title('t=%g'%(n*deltaT),fontsize='x-small')
        axi.set_xticks(())
        axi.set_yticks(())
        axi.grid(False)
    plt.tight_layout(pad=0.2,w_pad=0.2,h_pad=0)
    
    if saveFig:
        plt.savefig(fileName+'-FCs.png',dpi=200)
    
#    plt.show()
    
    # return fig



#%%
## As an example it takes the next time series:

if __name__=='__main__':    
    import Wavelets      
#    data_series=np.loadtxt("Vfilt-FR30to45noIh-50nodes-seed619-g0.316228.txt.gz")
    data_series=np.loadtxt("Vfilt-FR30to45chaos2C-50nodes-seed213-g0.01.txt.gz")

#    data_series=data_series[::10,:]
    dt=0.004
    runTime=27
    nnodes=50
    Trun=np.arange(0,runTime,dt)
    
    freqs=np.arange(2,15,0.2)  #Desired frequencies
    Periods=1/(freqs*dt)    #Desired periods in sample untis
    dScales=Periods/Wavelets.Morlet.fourierwl  #desired Scales
    
    #wavel=Wavelets.Morlet(EEG,largestscale=10,notes=20,scaling='log')
    wavelT=[Wavelets.Morlet(y1,scales=dScales) for y1 in data_series.T]
    cwt=np.array([wavel.getdata() for wavel in wavelT])
    pwr=np.array([wavel.getnormpower() for wavel in wavelT])
    
    phase=np.array([np.angle(cwt_i) for cwt_i in cwt])
    
    spec=np.sum(pwr,-1)
    maxFind=np.argmax(spec,-1)
    maxFreq=freqs[maxFind]
    
    bound1=int(1/dt)
    bound2=int((runTime-1)/dt)
    phaseMaxF=phase[range(nnodes),maxFind,bound1:bound2]
    phasesynch=np.abs(np.mean(np.exp(1j*phaseMaxF),0))

    Pcoher=np.zeros((nnodes,nnodes))
    
    for ii in range(nnodes):
        for jj in range(ii):
            Pcoher[ii,jj]=np.abs(np.mean(np.exp(1j*np.diff(phaseMaxF[[ii,jj],:],axis=0))))

    #sns.clustermap(corr_data_series) #to see how it cluster the time series, seaborn has the function clustermap
    ##############################################################################
    #%%

    PcorrFCD,Pcorr,shift=extract_FCD(phaseMaxF[:,::],wwidth=100,olap=0.5,mode='psync')
    Tini=1
    Tfin=Trun[-1]/1000 - 1
    plt.figure(4,figsize=(10,12))
    plt.clf()
    
    plt.subplot2grid((5,5),(0,0),rowspan=2,colspan=5)
    plt.plot(Trun[bound1:bound2],phasesynch)
    plt.title('mean P sync')
    
    plt.subplot2grid((5,5),(2,0),rowspan=2,colspan=2)
    plt.imshow(PcorrFCD,vmin=0,vmax=1,extent=(Tini,Tfin,Tfin,Tini),interpolation='none',cmap='jet')
    plt.title('P coher FCD')
    plt.grid()

#    plt.subplot2grid((5,5),(3,4))
#    plt.imshow(Psynch+Psynch.T+np.eye(nnodes),cmap='jet',vmax=1,vmin=0,interpolation='none')
#    plt.gca().set_xticklabels((),())
#    plt.gca().set_yticklabels((),())
#    plt.title('P sync')
#    plt.grid()
    
    plt.subplot2grid((5,5),(3,4))
    plt.imshow(Pcoher+Pcoher.T+np.eye(nnodes),cmap='jet',vmax=1,vmin=0,interpolation='none')
    plt.gca().set_xticklabels((),())
    plt.gca().set_yticklabels((),())
    plt.title('P coher')
    plt.grid()
    
    axes2=[plt.subplot2grid((5,5),pos) for pos in ((4,0),(4,1),(4,2),(4,3),(4,4))]
    for axi,ind in zip(axes2,(20,35,50,75,90)):
        corrMat=np.zeros((nnodes,nnodes))
        corrMat[np.tril_indices(nnodes,k=-1)]=Pcorr[ind]
        corrMat+=corrMat.T
        corrMat+=np.eye(nnodes)
        
        axi.imshow(corrMat,vmin=0,vmax=1,interpolation='none',cmap='jet')
        
        axi.set_xticklabels((),())
        axi.set_yticklabels((),())
        
        axi.set_title('t=%.2g'%(ind*Tfin/len(Pcorr)))
        axi.grid()


    #correlations,corrV,delta = extract_FCD(np.unwrap(data_series,axis=0),coldata=True,maxNwindows=100,wwidth=1000,mode='corr')
    #correlations,corrV = extract_FCD(data_series,coldata=True,maxNwindows=150,wwidth=200,mode='corr')
    
