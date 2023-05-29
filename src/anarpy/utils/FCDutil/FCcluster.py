#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 20 12:59:05 2018

@author: porio

"""

## Testing clustering by density peaks on retina data
# import matplotlib
# matplotlib.use('Agg')

import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from sklearn.decomposition import PCA

from . import dclus
from scipy.spatial import distance
from mpl_toolkits.mplot3d import Axes3D

#%%
def FCcluster(FCs,Distance='PCA',npcs=5,varexp=0.6,dc=0.05,minDist=0,minmax=[0,1],
              cmap=None,Trun=None,saveFig=False,fileName='FCD'):
    """
    Performs the clustering of a series of Functional Connectivity matrices, plus
    some plots.
    
    Parameters
    ----------
    
    FCs : array
        MxN Matrix containing FCs. M (rows) = FCs. N (columns) = node pairs. 
        Note that N = n(n-1)/2 where n is the number of nodes.
        
    Distance : 'PCA' or MxM matrix (M=number of FCs)
        If 'PCA' or not given, a PCA reduction of the FCs will be performed and the
        euclidean distance between the first npcs components will be used (default method).
        Otherwise, an arbitrary distance matrix can be given. Note that if you want to use
        a correlation matrix, you have to use 1-corr to have a distance.

    npcs : integer
        Number of principal components to consider for clustering and Data projection.
        Only used if Distance='PCA'
           
    varexp : float (0-1)
        Explained variance threshold. If the 10 first Principal Components
        explain less than varexp(fraction) of the variance, no clustering is performed.
        This is independent of the Distance method.
             
    dc : float 
        d_c parameter of the clustering algorithm.
        
    minDist : float
        Distance threshold for perfoming clustering. If less than 10 distances
        (in the PCA space) are greater than minDist, then no clustering is
        performed and only one cluster is returned.
        
    minmax : list or tuple
        minimum and maximum value for FC plots
        
    cmap : None
        colormap for FC plots. If None, either 'seismic' or 'jet' is automatically selected.
            
    Trun : float
        Total time of the simulation, for plotting purposes. If not given or None,
        the number of FCs is taken as total time.
            
    saveFig : boolean
        Whether or not save the Figures.
        
    fileName : string
        Basename to save Figures. "-clustering.png" and "-centroids.png" will be appended.
    
    Returns
    -------

    nclusters
        Number of clusters found (0 if varexp criterion is not fullfiled)
    
    n_eig80
        Number of Principal Components that explain 80% of the variance
    
    cluslabels
        Array of len M, giving the cluster to which each FC was assigned to.
    
    centidx
        Index of FCs identified as centroids
    
    projData
        M x npcs array, containing the FCs projected into the first npcs principal components.
    
    """
    if Trun==None:
        Trun=FCs.shape[0]
    if len(minmax)!=2:
        raise ValueError("minmax must be a list with two values")
    if cmap==None:
        if minmax[0]==0:
            cmap='jet'
        else:
            cmap='seismic'
    time=np.linspace(0,Trun,len(FCs))
    L = FCs.shape[-1]
    nnodes = int(np.sqrt(8*L+1)+1)//2
    N=len(FCs)
    
    if Distance!='PCA':
        if type(Distance)!=np.ndarray:
            raise TypeError("Distance must be 2D array or 'PCA'")
        if Distance.shape!=(N,N):
            raise ValueError("Distance matrix must be NxN, N=number of FCs")

# Calculo de PCA sobre la matriz de FCs
    pcac = PCA()
    pcac.fit(FCs)
    U = pcac.components_.T #en caso de que quede negativo revisar y multiplicar por -1
    expVar10 = np.sum(pcac.explained_variance_ratio_[0:10])
    allvar = pcac.explained_variance_ratio_
    
    cummVarexp=np.cumsum(allvar)
    n_eig80=np.where(cummVarexp>0.8)[0][0]  #Num de EigVals que explican 80%

# Grafico de las 'ncomp' primeros eigenvectors (en forma de matriz)
    ncomp = 6
    
    vmax=np.max(np.abs(U[:,:ncomp]))
    
    plt.figure(101,figsize=(10,1.5));
    plt.clf()
    for k in range(ncomp):
        Conect1 = np.zeros((nnodes,nnodes))
        cont = 0
        for i in range(1,nnodes):
           for j in range(0,i):
               Conect1[i,j] = U[cont,k]
               cont+=1
        ConectF = Conect1 + Conect1.T 
        axi = plt.subplot(1,ncomp+1,k+1)
        axi.set_xticks(())
        axi.set_yticks(())
        plt.imshow(ConectF,cmap='seismic',vmin=-vmax,vmax=vmax)  
        plt.title("Var expl %.4g"%allvar[k],size='x-small')
    axcb=plt.subplot(1,ncomp+1,k+2)
    axcb.set_aspect(6)
    plt.colorbar(mappable=axi.get_images()[0],cax=axcb,panchor=(0,0.2))
                
    plt.tight_layout(pad=0.2,w_pad=0.2)

    alpha = 0.02   #Umbral de confianza para el fit powerlaw
    # Computing PCA
    if Distance=='PCA':
        pca = PCA(n_components=npcs)
    
        pca.fit(FCs)#pca to FCs
        projData=pca.fit_transform(FCs)
        distmat = distance.cdist(projData, projData, 'euclidean')  #Matriz de Distancias
    else:
        distmat=Distance
        projData=pcac.fit_transform(FCs)

    # Calculo de rhos y deltas para clustering
    rho = dclus.compute_rho(distmat,dc)
    delta = dclus.compute_delta(distmat, rho)

    if expVar10 <varexp:
        #Si los primeros PCs explican menos de varexp, no hay clusters
        nclusters = 0
        cluslabels = list(range(len(FCs)))
        centidx=None

        fig4=plt.figure(102,figsize=(6,10))
        plt.clf()
        
        ax1=plt.subplot(311,projection='3d') # pc space
        ax1.plot(projData[:,0],projData[:,1],projData[:,2],'.-')
        
        plt.subplot(312)
        plt.imshow(distmat,cmap='jet')
        plt.colorbar()
        
        axFC=fig4.add_axes((0.15,0.15,0.7,0.2))
        axFC.imshow(FCs.T,vmin=minmax[0],vmax=minmax[1],cmap=cmap,aspect='auto')
        axFC.set_xticklabels(())
        
        axPCA=fig4.add_axes((0.15,0.08,0.7,0.05))
        axPCA.plot(time,projData)
        axPCA.set_xlim((0,Trun))

        if saveFig:
            plt.savefig(fileName+"-clustering.png",dpi=200)

#        plt.show()
        
    elif np.sum(distmat>minDist)<10:
        # If less than 10 distances are greater than minDist, there is only 1 cluster
        nclusters = 1
        cluslabels = np.ones(len(FCs))
        centidx=None        
        
        fig4=plt.figure(102,figsize=(6,10))
        plt.clf()
        
        
        ax1=plt.subplot(311,projection='3d') # pc space
        ax1.plot(projData[:,0],projData[:,1],projData[:,2],'.-')
        plt.title(str(nclusters)+' clusters')
        
        plt.subplot(312)
        plt.imshow(distmat,cmap='jet')
        plt.colorbar()
        
        axFC=fig4.add_axes((0.15,0.15,0.7,0.2))
        axFC.imshow(FCs.T,vmin=minmax[0],vmax=minmax[1],cmap=cmap,aspect='auto')
        axFC.set_xticklabels(())
        
        axPCA=fig4.add_axes((0.15,0.08,0.7,0.05))
        axPCA.plot(time,projData)
        axPCA.set_xlim((0,Trun))

        if saveFig:
            plt.savefig(fileName+"-clustering.png",dpi=200)        

    else:
        # Clustering: Computing thresholds, finding centroids and assigning variables to clusters
        nclusters,cluslabels,centidx,threshold = dclus.find_centroids_and_cluster(distmat,rho,delta,alpha)    
        

        fig4=plt.figure(102,figsize=(8,10))
        plt.clf()
        plt.subplot(321) # delta vs rho
        plt.plot(rho,delta,'b.')
        for i in centidx:
            plt.plot(rho[i],delta[i],'o')
        plt.plot(threshold[0,:],threshold[1,:],'k.')
        plt.title(str(nclusters)+' clusters')
        plt.xlabel(R'$\rho$')
        plt.ylabel(R'$\delta$')
        
        plt.subplot(323)
        #plt.plot(np.arange(len(cluslabels)),cluslabels,'.')
        for i in range(nclusters):
            plt.plot(np.where(cluslabels==i+1)[0],cluslabels[cluslabels==i+1],'o')
        
        ax1=plt.subplot(322,projection='3d') # pc space
        for i in range(nclusters):
            plt.plot(projData[cluslabels==i+1,0],projData[cluslabels==i+1,1],projData[cluslabels==i+1,2],'.')
        ax1.plot(projData[:,0],projData[:,1],projData[:,2],'k:',lw=0.5,alpha=0.5)
        ax1.plot(projData[centidx,0],projData[centidx,1],projData[centidx,2],'ko')
        
        axFCD=plt.subplot(324)
        axFCD.imshow(distmat,cmap='jet',extent=(0,Trun,Trun,0))
        plt.colorbar(mappable=axFCD.images[0],label='Euclid. Distance')
        xpos,_,wid,_=axFCD.get_position().bounds
        axClus2=fig4.add_axes((xpos,0.61,wid,0.015))
        axClus2.imshow(cluslabels[None,:]*np.ones((3,1)),cmap='tab10',aspect='auto',vmin=1,vmax=10)
        axClus2.axis('off')
        
        
        axFC=fig4.add_axes((0.15,0.15,0.7,0.2))
        axFC.imshow(FCs.T,vmin=minmax[0],vmax=minmax[1],cmap=cmap,aspect='auto')
        axFC.set_xticklabels(())
        
        axClus=fig4.add_axes((0.15,0.12,0.7,0.02))
        axClus.imshow(cluslabels[None,:]*np.ones((3,1)),cmap='tab10',aspect='auto',vmin=1,vmax=10)
        axClus.axis('off')
        
        axPCA=fig4.add_axes((0.15,0.07,0.7,0.04))
        # axPCA.imshow(projData.T,cmap='jet',aspect='auto')
        axPCA.plot(time,projData)
        axPCA.set_xlim((0,Trun))
        
#        plt.show()
        
        if saveFig:
            plt.savefig(fileName+"-clustering.png",dpi=200)  

        plt.figure(103)
        plt.clf()
        for k in range(1,nclusters+1):
            Indclus = np.where(cluslabels==k)[0]
            # Indcol = 10
            # Indrow = int(Indclus.shape[0]/Indcol)+1
            Matrixclus = np.zeros((nnodes,nnodes,len(Indclus)))
            for i,j in enumerate(Indclus):
                Clusi = np.zeros((nnodes,nnodes))
                Clusi[np.tril_indices(nnodes,k=-1)]=FCs[j]
                ClusiF = Clusi+Clusi.T+np.eye(nnodes)
                Matrixclus[:,:,i] = ClusiF
            MedianMatrix = np.median(Matrixclus,axis=2)
            axi = plt.subplot(1,nclusters,k)
            plt.imshow(MedianMatrix,cmap=cmap,vmin=minmax[0],vmax=minmax[1])
            axi.set_xticklabels(())
            axi.set_yticklabels(())
        
#        plt.show()
            
        if saveFig:
            plt.savefig(fileName+"-centroids.png",dpi=200)  
        # np.savetxt(Path+fileName+'Cluslabels%g.txt'%WW,cluslabels,fmt='%d')

    # =============================================================================
    #             Create .txt to quantify clusters and eigenvals to pca
    # =============================================================================
    
    print('hay', nclusters,' clusters en',fileName)

    return nclusters,n_eig80,cluslabels,centidx,projData
#%% 


#%%

if __name__=='__main__':

    import fcd

    Path='datasets/'
    fileName="HBihC-Det-s3-10-g0.0177828"
    
    Vfilt=np.load(Path+fileName+".npy")
    nnodes=Vfilt.shape[1]
    
    runInt=0.004  # Intervalo de muestreo en segundos
    runTime=Vfilt.shape[0]*runInt  #Tiempo total de simulación
    
    # Calculo de fase instantanea (ojo que la señal ya viene filtrada en la banda de ~10 Hz,
    # si no, habría que filtrar primero)
    phase=np.angle(signal.hilbert(Vfilt,axis=0))  
    
    bound1=int(1/runInt)
    bound2=int((runTime-1)/(runInt))
    phase=phase[bound1:bound2].T  #Eliminamos el primer y ultimo segundo de la señal
    
    runTime=Vfilt.shape[0]*runInt  #Volvemos a calcular el Tiempo total de simulación
    Trun=np.arange(0,runTime,runInt)  # Vector con valores de tiempo
    
    PStot=np.zeros((nnodes,nnodes))        
    for ii in range(nnodes):
        for jj in range(ii):
            PStot[ii,jj]=np.abs(np.mean(np.exp(1j*np.diff(phase[[ii,jj],:],axis=0))))
            
    phasesynch=np.abs(np.mean(np.exp(1j*phase),0))
    MPsync=np.mean(phasesynch)  #Media de la fase en el tiempo
    VarPsync=np.var(phasesynch)  #Varianza de la fase en el tiempo
    
    
    #%% Calculo de FCs y FCD
    # FCD = Matriz de FCD
    # Pcorr = matriz conteniendo las FCs estiradas
    
    # Parámetros para el cálculo de las FCs y FCD
    WW = 200 #Ancho de ventana a utilizar (en muestras)
    mode = 'psync'# medida para las FCs. 'psync' = sincroná de fase (en pares)
    modeFCD='clarksondist' # medida para la FCD. 'clarksondist' = distancia angular
    
    FCD,Pcorr,shift=fcd.extract_FCD(phase[:,::],maxNwindows=2000,wwidth=WW,olap=0.5,mode=mode,modeFCD=modeFCD)
    
    #Varianza de los valores encontrados en la FCD
    VarFCD = np.var(FCD[np.tril_indices(len(Pcorr),k=-5)])
    plotFC(Pcorr)
    
    FCcluster(Pcorr)
