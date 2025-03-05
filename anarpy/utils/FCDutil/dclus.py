# Conjunto de funciones para detectar asambleas neuronales 
import numpy as np
import scipy.stats
import statsmodels.api as sm
from statsmodels.sandbox.regression.predstd import wls_prediction_std


def clustering_by_density(dist,dc,alpha,ishalo):
    """
    # Searches for clusters on the 'dist' distance matrix using the density
    # peaks clustering methods by Rodriguez and Laio 2014.
    #
    # INPUTS
    #
    # 'dist' is a nbins x nbins matrix of distances between the significant bins
    # found on data
    # 'percNeigh' average percentage of neighbours, ranging from [0, 1]
    # 'clusmet' is the method used for centroid search
    # 'prct' is the percentile to compute prediction bounds on the centroid detection
    # 'ishalo' is 1 if halo cluster refinement is used and 0 otherwise
    
    #
    # OUTPUTS
    #
    # 'Nens' is the number of ensembles found
    # 'ensId' is the corresponding ensemble of each bin of dist
    # 'centInd' is the moste representative bin of the ensemble
    # 'delta' is the minimum distance between point it and any other point with
    # higher density
    # 'rho' is the local density of point i
    """
    npts = len(dist)
    dist = dist*(1-np.eye(npts)) # removing diagonal
    rho = compute_rho(dist,dc)
    delta = compute_delta(dist,rho)
    nclus,cluslabels,centid,threshold = find_centroids_and_cluster(dist,rho,delta,prct)
    halolabels = halo_assign(dist,cluslabels,centid,ishalo)
    return rho,delta,centid,cluslabels,halolabels,threshold
    
    
def compute_rho(dist,dc):
    npts = len(dist)
    prctneig = int(np.round(dc*npts)) # number of closest neighbors to compute density
    dist_sorted = np.sort(dist,axis=0) # sorting each row in asciending order
    rho =  1/np.mean(dist_sorted[1:prctneig+1,:],axis=0) # density computation
    rho[np.isnan(rho)]=0;
    return rho
    
def compute_delta(dist, rho):
    """
    % DENSITYCLUST Clustering by fast search and find of density peaks.
    %   SEE the following paper published in *SCIENCE* for more details:
    %       Alex Rodriguez & Alessandro Laio: Clustering by fast search and find of density peaks,
    %       Science 344, 1492 (2014); DOI: 10.1126/science.1242072.
    %   INPUT:
    %       dist: [NE, NE] distance matrix
    %       rho: local density [row vector]
    %   OUTPUT:
            delta: minimun distance to the next higher density point
    
    %
    % MODIFIED BY RUBEN HERZOG 2018.
    % Uses matrix operations instead of several for. Optimized for GPUs
    """
    rho_sort_id = np.argsort(rho) # index to sort
    rho_sort_id = (rho_sort_id[::-1]) # reversing sorting indexes
    sort_rho = rho[rho_sort_id] # sortig rho in ascending order
    gtmat = np.greater_equal(sort_rho,sort_rho[:,None]) # gtmat(i,j)=1 if rho(i)>=rho(j) and 0 otherwise
    
    sortdist = np.zeros_like(dist)
    sortdist = dist[rho_sort_id,:]
    sortdist = sortdist[:,rho_sort_id]
    
    seldist = gtmat*sortdist # keeping only distance to points with highest or equal rho 
    seldist[seldist==0] = float("inf") 
              
    auxdelta = np.min(seldist,axis=1)
    delta=np.zeros_like(auxdelta) 
    delta[rho_sort_id] = auxdelta 
    delta[rho==np.max(rho)] = np.max(delta[np.logical_not(np.isinf(delta))]) # assigns max delta to the max rho
    delta[np.isinf(delta)] = 0
    return delta
    
def find_centroids_and_cluster(dist,rho,delta,alpha):
    """
    Finds centroids based on the rho vs delta relationship. 
    Fits a power law to the rhos vs delta function and uses the prct-th
    prediction bound as threshold for choosing the centroids.
    Then, the non-centroids are assigned to their closest centroid.
    
    INPUTS
    
    delta
    rho
    prct
    
    OUTPUT
    nclus
    cluslabels
    centid
    threshold
    """

    npnts = len(rho)    
    centid = np.zeros((npnts))    

    # fitting a power law to the rho vs delta relationship
    # preparing data
    mindelta = 10**(-6) # min delta to be considered, improves fit
    nzind = np.where(np.logical_and(delta>mindelta,rho>0))[0] # delta different from 0 and rhos higher than 0
    nzdelta = delta[nzind] # y of fit
    nzrho = rho[nzind] # x of fit
    
    # fitting a line in log space
    threshold = estimate_threshold(np.log(nzrho),np.log(nzdelta),alpha)
    threshold = np.maximum(np.exp(threshold),np.ones_like(threshold)*np.max(delta)*0.3) # to linear form
    
    # selecting centroids
    selid = (nzdelta>threshold)    
    auxid = nzind[selid] # centroids on original basis
    if len(auxid)==0:
        auxid=(np.argmax(rho),)
    nclus = len(auxid)
    centid[auxid] = np.arange(0,nclus,1)+1 # assigning labels to centroids
    threshold = np.vstack((nzrho,threshold)) # saving the x and y
    
    # assigning points to clusters based on their distance to the centroids
    if nclus==1:
        cluslabels = np.ones(npnts)
        centidx = np.where(centid)[0]
    else:
        centidx = np.where(centid)[0] # index of centroids
        dist2cent = dist[centidx,:]
        cluslabels = np.argmin(dist2cent,axis=0)+1
        _,cluscounts = np.unique(cluslabels,return_counts=True) # number of elements of each cluster
        one_mem_clus = np.where((cluscounts==1) | (cluscounts==0))[0] # index of 1 or 0 members clusters
        if one_mem_clus.size>0: # if there one or more 1 or 0 member cluster
            # cluslab = centid[centidx] # cluster labels
            # id2rem = np.where(np.in1d(cluslab,one_mem_clus)) # ids to remove
            clusidx=np.delete(centidx,one_mem_clus) # removing
            centid = np.zeros(len(centid))
            nclus = nclus-len(one_mem_clus)
            centid[clusidx]=np.arange(0,nclus,1)+1 # re labeling centroids            
            dist2cent = dist[centidx,:]# re compute distances from centroid to any other point
            cluslabels = np.argmin(dist2cent,axis=0)+1 # re assigns clusters 
            
    return nclus,cluslabels,centidx,threshold
    

def halo_assign(dist,cluslabels,centid,op):
    """ If op==1, removes from each cluster all the points that are farther than
    the average distance to the cluster centroid
    
    INPUTS
    
    dist
    cluslabels
    centid
    op
    
    OUTPUTS
    
    halolabels
    """
    halolabels = cluslabels.copy()
    if op:
        # sameclusmat[i,j]=1 is i and j belongs to the same cluster and 0 otherwise
        sameclusmat = np.equal(cluslabels,cluslabels[:,None]) #
        sameclus_cent = sameclusmat[centid>0,:] # selects only centroids
        dist2cent = dist[centid>0,:] # distance to centroids
        dist2cluscent = dist2cent*sameclus_cent # preserves only distances to the corresponding cluster centroid
        nclusmem = np.sum(sameclus_cent,axis=1) # number of cluster members
        
        meandist2cent = np.sum(dist2cluscent,axis=1)/nclusmem # mean distance to corresponding centroid
        gt_meandist2cent = np.greater(dist2cluscent,meandist2cent[:,None]) # greater than the mean dist to centroid
        remids = np.sum(gt_meandist2cent,axis=0)
        halolabels[remids>0] = 0 # setting to 0 the removes points
        return halolabels


def estimate_threshold(x,y,alpha):
    x = sm.add_constant(x)
    model = sm.OLS(y,x)
    results = model.fit()
    _,_,threshold = wls_prediction_std(results,alpha=alpha)
    return threshold

    
    
    
def linear_regression(x, y, prob):
    """
    Return the linear regression parameters and their <prob> confidence intervals.
    ex:
    >>> linear_regression([.1,.2,.3],[10,11,11.5],0.95)
    b0 is intercetp, b1 is slope, bb0 is interval for b0 and bb1 for b1
    NOT REALLY GOOD AT ESTIMATING BOUNDS
    """
    x = np.array(x)
    y = np.array(y)
    n = len(x)
    xy = x * y
    xx = x * x

    # estimates
    b1 = (xy.mean() - x.mean() * y.mean()) / (xx.mean() - x.mean()**2)
    b0 = y.mean() - b1 * x.mean()
    s2 = 1./n * sum([(y[i] - b0 - b1 * x[i])**2 for i in range(n)])
    #print 'b0 = ',b0
    #print 'b1 = ',b1
    #print 's2 = ',s2
    
    #confidence intervals    
    alpha = 1 - prob   
    c = -1 * scipy.stats.t.ppf(alpha/2.,n-2)
    bb1 = c * (s2 / ((n-2) * (xx.mean() - (x.mean())**2)))**.5
    #print 'the confidence interval of b1 is: ',[b1-bb1,b1+bb1]
    
    bb0 = c * ((s2 / (n-2)) * (1 + (x.mean())**2 / (xx.mean() - (x.mean())**2)))**.5
    #print 'the confidence interval of b0 is: ',[b0-bb0,b0+bb0]
    return b1,b0,bb1,bb0


if __name__=='__main__':
    
    find_centroids_and_cluster(distmat,rho,delta,alpha)











    

    
    
