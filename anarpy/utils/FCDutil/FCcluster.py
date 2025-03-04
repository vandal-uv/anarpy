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
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn import metrics

from . import dclus
from scipy.spatial import distance
from mpl_toolkits.mplot3d import Axes3D

#%%

def split_random_range(N):
    list1 = list(range(N))
    np.random.shuffle(list1)
    splitpoint = N//2
    g1,g2 = list1[:splitpoint], list1[splitpoint:]
    g1.sort()
    g2.sort()
    return g1,g2 

def clustering_stability(data,n_clus=2,repeats=30):
    kmeans = KMeans(n_clusters=n_clus, random_state=0, n_init="auto").fit(data)
    labels = kmeans.labels_
    centers = kmeans.cluster_centers_
    Nsamples = len(labels)
    
    scores=[]
    
    for ii in range(repeats):
        group1,group2 = split_random_range(Nsamples)
        
        kmeansHalf = KMeans(n_clusters=n_clus, random_state=0, n_init="auto").fit(data[group1])
        labels1 = kmeansHalf.labels_
        labels2 = kmeansHalf.predict(data[group2])
        
        labelsTest = np.ones(Nsamples)*-1
        labelsTest[group1] = labels1
        labelsTest[group2] = labels2
        
        scores.append(metrics.adjusted_rand_score(labels,labelsTest))
        # scores.append(metrics.adjusted_mutual_info_score(labels,labelsTest))
        # scores.append(metrics.fowlkes_mallows_score(labels,labelsTest))
        
    return np.mean(scores),labels,centers

def fig_noclusters(projData,tsneData,FCs,distmat,minmax,cmap,Trun=None,filename='FCD'):
    if Trun is None:
        Trun = len(FCs)
    
    time=np.linspace(0,Trun,len(FCs))
    fig4=plt.figure(102,figsize=(6,10))
    plt.clf()
    
    ax1=plt.subplot(321,projection='3d') # pc space
    ax1.plot(projData[:,0],projData[:,1],projData[:,2],'.-')
    
    ax2=plt.subplot(322) # pc space
    ax2.plot(tsneData[:,0],tsneData[:,1],'.-')
    
    plt.subplot(312)
    plt.imshow(distmat,cmap='jet')
    plt.colorbar()
    
    axFC=fig4.add_axes((0.15,0.15,0.7,0.2))
    axFC.imshow(FCs.T,vmin=minmax[0],vmax=minmax[1],cmap=cmap,aspect='auto')
    axFC.set_xticklabels(())
    
    axPCA=fig4.add_axes((0.15,0.08,0.7,0.05))
    axPCA.plot(time,projData)
    axPCA.set_xlim((0,Trun))

    plt.savefig(filename+"-clustering.png",dpi=200)
    
def fig_cluster(projData,tsneData,FCs,nclusters, cluslabels, centers, distmat,
                minmax,cmap,Trun=None,filename='FCD', rho=None, delta=None, threshold=None,):
    
    if Trun is None:
        Trun = len(FCs)
        
    N,L = FCs.shape
    nnodes = int(np.sqrt(8*L+1)+1)//2
    
    time=np.linspace(0,Trun,len(FCs))
    
    fig4=plt.figure(102,figsize=(8,10))
    plt.clf()
    
    if rho is not None:
        plt.subplot(321) # delta vs rho
        plt.plot(rho,delta,'b.')
        for i in centers:
            plt.plot(rho[i],delta[i],'o')
        plt.plot(threshold[0,:],threshold[1,:],'k.')
        plt.title(str(nclusters)+' clusters')
        plt.xlabel(R'$\rho$')
        plt.ylabel(R'$\delta$')
    else:
        plt.subplot(321)
        plt.scatter(tsneData[:,0], tsneData[:,1],c=cluslabels,cmap='tab10',vmin=0,vmax=9)
        plt.plot(tsneData[:,0], tsneData[:,1],'-', lw=0.5, color='grey')
    
    plt.subplot(323)
    #plt.plot(np.arange(len(cluslabels)),cluslabels,'.')
    for i in range(nclusters):
        plt.plot(np.where(cluslabels==i)[0],cluslabels[cluslabels==i],'o')
    
    ax1=plt.subplot(322,projection='3d') # pc space
    for i in range(nclusters):
        plt.plot(projData[cluslabels==i,0],projData[cluslabels==i,1],projData[cluslabels==i,2],'.')
    ax1.plot(projData[:,0],projData[:,1],projData[:,2],'k:',lw=0.5,alpha=0.5)
    
    if type(centers) is list or centers.ndim==1:
        ax1.plot(projData[centers,0],projData[centers,1],projData[centers,2],'ko')
    else:
        ax1.plot(centers[:,0],centers[:,1],centers[:,2],'ko')
            
    axFCD=plt.subplot(324)
    axFCD.imshow(distmat,cmap='jet',extent=(0,Trun,Trun,0))
    plt.colorbar(mappable=axFCD.images[0],label='Euclid. Distance')
    xpos,_,wid,_=axFCD.get_position().bounds
    axClus2=fig4.add_axes((xpos,0.61,wid,0.015))
    axClus2.imshow(cluslabels[None,:]*np.ones((3,1)),cmap='tab10',aspect='auto',vmin=0,vmax=9)
    axClus2.axis('off')
    
    axFC=fig4.add_axes((0.15,0.15,0.7,0.2))
    axFC.imshow(FCs.T,vmin=minmax[0],vmax=minmax[1],cmap=cmap,aspect='auto')
    axFC.set_xticklabels(())
    
    axClus=fig4.add_axes((0.15,0.12,0.7,0.02))
    axClus.imshow(cluslabels[None,:]*np.ones((3,1)),cmap='tab10',aspect='auto',vmin=0,vmax=9)
    axClus.axis('off')
    
    axPCA=fig4.add_axes((0.15,0.07,0.7,0.04))
    # axPCA.imshow(projData.T,cmap='jet',aspect='auto')
    axPCA.plot(time,projData)
    axPCA.set_xlim((0,Trun))
    
    plt.savefig(filename+"-clustering.png",dpi=200)  

    plt.figure(103)
    plt.clf()
    for k in range(nclusters):
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
        axi = plt.subplot(1,nclusters,k+1)
        plt.imshow(MedianMatrix,cmap=cmap,vmin=minmax[0],vmax=minmax[1])
        axi.set_xticklabels(())
        axi.set_yticklabels(())
    
    plt.savefig(filename+"-centroids.png",dpi=200)  
    
def FCclusterKmeans(FCs,npcs=15, varexp=0.6, max_clusters=8, min_clusters=2,
                    minDist = 0.2, 
                    minmax=[0,1], cmap=None, Trun=None, saveFig=False, fileName='FCD'):
    """

    Parameters
    ----------
    FCs : array
        MxN Matrix containing FCs. M (rows) = FCs. N (columns) = node pairs. 
        Note that N = n(n-1)/2 where n is the number of nodes.  
    npcs : integer, optional
        Number of principal components to consider for clustering and Data projection.
        Only used if Distance='PCA'. The default is 15.
    varexp : float, optional
        Explained variance threshold. If the 10 first Principal Components
        explain less than varexp(fraction) of the variance, no clustering is performed.
        This is independent of the Distance method. The default is 0.6.
    max_clusters : int, optional
        Maximum number of clusters to test. The default is 8.
    min_clusters : int, optional
        Minimum number of clusters to test. The default is 2.
    minDist : float, optional
        Distance threshold for perfoming clustering. If less than 10 distances
        (in the PCA space) are greater than minDist, then no clustering is
        performed and only one cluster is returned.. The default is 0.2.
    minmax : list or tuple, optional
        minimum and maximum value for FC plots. The default is [0,1].
    cmap : string, optional
        colormap for FC plots. If None, either 'seismic' or 'jet' is
        automatically selected based on minmax. The default is None.
    Trun : float, optional
        Total time of the simulation, for plotting purposes. If None,
        the number of FCs is taken as total time.. The default is None.
    saveFig : Boolean, optional
        Whether or not save the Figures.. The default is False.
    fileName : string, optional
        Basename to save Figures. "-clustering.png" and "-centroids.png" will be appended.
        The default is 'FCD'.

    Raises
    ------
    ValueError
        If minmax is other than 2 values.

    Returns
    -------
    nclusters
        Number of clusters found (0 if varexp criterion is not fullfiled)
    
    n_eig80
        Number of Principal Components that explain 80% of the variance
    
    labels
        Array of len M, giving the cluster to which each FC was assigned to.
    
    centers
        centroids
    
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
            
            
    pcac = PCA()
    pcac.fit(FCs)
    expVar10 = np.sum(pcac.explained_variance_ratio_[0:10])
    allvar = pcac.explained_variance_ratio_
    
    cummVarexp=np.cumsum(allvar)
    n_eig80=np.where(cummVarexp>0.8)[0][0]  #Num de EigVals que explican 80%

    pcaP = PCA(n_components=npcs)
    projData=pcaP.fit_transform(FCs)
    # U = pcaP.components_.T #en caso de que quede negativo revisar y multiplicar por -1

    tsneData = TSNE(n_components=2).fit_transform(FCs)
    
    clust_scores=[]
    clust_labels = []
    clust_centers = []

    clust_to_test = list(range(max_clusters,min_clusters-1,-1))
    for nclus in clust_to_test:
        sc,lb,cn = clustering_stability(projData, n_clus=nclus, repeats=30)
        clust_scores.append(sc)
        clust_labels.append(lb)
        clust_centers.append(cn)

    labels = clust_labels[np.argmax(clust_scores)]
    centers = clust_centers[np.argmax(clust_scores)]
    distmat = distance.cdist(projData, projData, 'euclidean')

    if expVar10 <varexp:
        #Si los primeros PCs explican menos de varexp, no hay clusters
        nclusters = 0
        labels = list(range(len(FCs)))
        centers=None

        if saveFig:
            fig_noclusters(projData, tsneData,FCs,distmat,minmax,cmap,Trun,fileName)

#        plt.show()
        
    elif np.sum(distmat>minDist)<10 or np.max(clust_scores) < 0.5:
        # If less than 10 distances are greater than minDist, there is only 1 cluster
        nclusters = 1
        labels = np.ones(len(FCs))
        centers=None        
        
        if saveFig:
            fig_noclusters(projData, tsneData,FCs,distmat,minmax,cmap,Trun,fileName)
            
    else:
        nclusters = len(set(labels))
        if saveFig:
            fig_cluster(projData, tsneData, FCs, nclusters, labels, centers, 
                        distmat, minmax, cmap, Trun, fileName)
    
    return nclusters, n_eig80,labels, centers, projData, clust_scores


def FCcluster(FCs,Distance='PCA',npcs=5,varexp=0.6,dc=0.05, minDist=0,minmax=[0,1],
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
    N,L = FCs.shape
    nnodes = int(np.sqrt(8*L+1)+1)//2
    
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
    
    tsneData = TSNE(n_components=2).fit_transform(FCs)

    if expVar10 <varexp:
        #Si los primeros PCs explican menos de varexp, no hay clusters
        nclusters = 0
        cluslabels = list(range(len(FCs)))
        centidx=None

        if saveFig:
            fig_noclusters(projData,tsneData,FCs,distmat,minmax,cmap,Trun,fileName)

#        plt.show()
        
    elif np.sum(distmat>minDist)<10:
        # If less than 10 distances are greater than minDist, there is only 1 cluster
        nclusters = 1
        cluslabels = np.ones(len(FCs))
        centidx=None        

        if saveFig:
            fig_noclusters(projData,tsneData,FCs,distmat,minmax,cmap,Trun,fileName)

    else:
        # Clustering: Computing thresholds, finding centroids and assigning variables to clusters
        nclusters,cluslabels,centidx,threshold = dclus.find_centroids_and_cluster(distmat,rho,delta,alpha)    
        
        if saveFig:
            fig_cluster(projData, tsneData, FCs, nclusters, cluslabels, centidx, distmat,
                        minmax, cmap, Trun, fileName, rho, delta, threshold)
    
    print('hay', nclusters,' clusters en',fileName)

    return nclusters,n_eig80,cluslabels,centidx,projData

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
