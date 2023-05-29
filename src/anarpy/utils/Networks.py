# -*- coding: utf-8 -*-
"""
@author:Pato
"""

import numpy as np

def reversebits(n, bit_size=None):
    """
    Reverse the bits of an integer number

    Parameters
    ----------
    n : integer
        
    bit_size : TYPE, optional
        Total number of bits to consider when reversing. If None (default),
        the bits of the input are considered

    Returns
    -------
    integer
        Number that results when the bits of the input are reversed in order.

    """
    bin_number = bin(n)
    reverse_number = bin_number[-1:1:-1]
    if bit_size is not None:
        reverse_number = reverse_number + (bit_size - len(reverse_number))*'0'
    return int(reverse_number, 2)

def distCM(M,N=None,density=0.2,rnd=0,directed=False,symmetrical=True,diag=None,seed=None):
    """
    Distance-based algorithm for small-world network. The probability of connection
    between two neurons is 
    
    :math:`P_{conn}=\\frac{1}{(1+exp(-(2^{-distance/K}-0.5)/rnd))}`
        
    where K is the average number of connections per neuron. if rnd=0, there is a step function
    at distance=K and the result is a regular lattice.
    
    Parameters
    ----------
    
    M : int 
        Number of receiving neurons (rows)
    
    N : int, optional
        Number of emmiting neurons (columns). Default = M
    
    density : float. 0 < P < 1.
        Mean probability or density of connections. Default=0.2
    
    rnd : float
        randomness. Default=0 (regular lattice)
    
    directed : Boolean
        if True, connections are established only to one side of the diagonal. Default = False
        
    symmetrical : Boolean
        if True, CM[i,j]=CM[j,i]. Note that symmetrical cannot be true if directed or M!=N. 
    
    diag : Boolean
        if False or None, the diagonal is set to 0 when M==N. No effect if M!=N. Default: None
    
    seed : integer or None
        random seed. default: None
    """    
    if N is None:
        N = M
        
    if diag is None:
        if M==N:
            diag=False
        else:
            diag=True
    rng = np.random.default_rng(seed)
        
    if (directed or M!=N) and symmetrical:
        print("Warning:  Network cannot be symmetrical when directed or not square.\n symmetrical set to False.")
        symmetrical=False
        
    if symmetrical or not directed:
        K=N*density/2
    else:
        K=N*density

    if directed:
        DistMat=((np.arange(N) - np.linspace(0,N-1,M)[:,None] ) + N)%N        
    else:
        DistMat1=np.abs(np.arange(N) - np.linspace(0,N-1,M)[:,None] )
        DistMat2=np.abs(DistMat1-N)
        DistMat = np.minimum(DistMat1,DistMat2)
    ProbMat=2**(-DistMat/K)
    
    if rnd==0:
        CM=1*(ProbMat>0.5)
    else:
        sigma=1/rnd
        ConnProb=1/(1+np.exp(-sigma*(ProbMat-0.5)))

        if directed:
            ConnProb=ConnProb*K/np.sum(ConnProb,1)[:,None]
        else:
            ConnProb=ConnProb*2*K/np.sum(ConnProb,1)[:,None]
        CM=1*(ConnProb > rng.uniform(size=(M,N)))
        
    if not diag and M==N:
        CM[np.diag_indices(N)]=0
    
    if symmetrical:
        CM[np.tril_indices_from(CM)]=0
        CM+=CM.T
        
    return CM

def SW(N,M=None,density=0.2,Prw=0,directed=False,symmetrical=True,diag=None,seed=None):
    """
    Watts-Strogatz algorithm for generating a small-world network.
    
    Parameters
    ----------
    
    N : Number of receiving neurons (rows)
    
    M : Number of emmiting neurons (columns). Default = N
    
    density : Probability or density of connections. Default=0.2
    
    Prw : Probability of random re-wiring. Default=0 (regular lattice)
    
    directed : if True, connections are established only to one side of
    the diagonal. Default = False
    
    symmetrical : if True, CM[i,j]=CM[j,i]. Note that symmetrical cannot be true
    if directed is True or if M!=N
    
    diag : if False or None, the diagonal is set to 0 when M==N. No effect if M!=N. 
    Default: None

    seed : integer or Nonr
        random seed. default: None
    """    
    if M is None:
        M = N
        
    if diag is None:
        if M==N:
            diag=False
        else:
            diag=True

    if (directed or M!=N) and symmetrical:
        print("Warning:  Network cannot be symmetrical when directed or not square.\n symmetrical set to False.")
        symmetrical=False
    if symmetrical or not directed:
        K=int(np.sqrt(N*M)*density/2)
    else:
        K=int(np.sqrt(N*M)*density)
        
    rng = np.random.default_rng(seed)
    
    CM=np.zeros((N,M))

    if not symmetrical and not directed:
        for i in range(N):
            for j in range(int(i*(M/N))-K,int(i*(M/N))+K+1):
                if rng.uniform()<Prw:
                    ii=rng.integers(M)
                    while CM[i,(i-ii)%M]==1:
                        ii=rng.integers(M)
                    CM[i,(i-ii)%M]=1
                else:
                    CM[i,j%M]=1
    else:
        for i in range(N):
            for j in range(int(i*(M/N))+1,int(i*(M/N))+K+1):
                if rng.uniform()<Prw:
                    ii=rng.integers(M)
                    while CM[i,(i-ii)%M]==1:
                        ii=rng.integers(M)
                    CM[i,(i-ii)%M]=1
                else:
                    CM[i,j%M]=1
        if symmetrical:
#            CM[np.tril_indices_from(CM)]=0
            CM+=CM.T
            CM=np.minimum(CM,np.ones_like(CM))
    
    if not diag and M==N:
        CM[np.diag_indices(N)]=0
    
    return CM

def ModularNet(N, N_mod=4, density=0.08, P_inter=0.01, rand_Nintra=0, sw_intra = 1,
               directed=False, symmetrical_inter=True, symmetrical_intra=True,
               diag=False, seed=None):
    """
    Modular Network.
    
    Intra-module connections are random, with an initial probability of 
    density*N_mod (this product must be lower than 1). Intra-module connections
    are then removed and replaced by inter-module connections of density P_inter.
        

    Parameters
    ----------
    N : integer
        Total number of nodes.
    N_mod : integer, optional
        Number of modules. The default is 4.
    density : float, optional
        Mean density of the network. Must be lower than 1/N_mod. The default is 0.08.
    P_inter : float, optional
        Probability or density of inter-module connections. Must be between 0
        and density. The default is 0.01.
    rand_Nintra : float, optional.  0 <= rand <= 1
        Random factor of module sizes. If 0, all modules will be (mostly) the
        same size. If 1, module sizes will be random between 0.5 and 1.5 times the
        average size.
    sw_intra : float, optional. 0 <= sw_intra <=1
        Small-world parameter for intra-module connections. If sw_intra==1 (Default),
        modules are completely random. If 0, modules have a lattice structure.
        With values between 0.01 and 0.1, modules will have small-world architecture.
    directed : Boolean, optional
        if True, inter-module connections are established only to one side 
        of the diagonal. Default = False
    symmetrical_inter : Boolean, optional
        if True, CM[i,j]=CM[j,i] for connections outside modules. Note that 
        symmetrical_inter cannot be True if directed is True. Default = True.
    symmetrical_intra : Boolean, optional
        if True, CM[i,j]=CM[j,i] for connections inside modules. Default = True.
    diag : Boolean, optional.
        if False, self-connections are removed. Default = False.
    seed : integer or None
            random seed. default: None

    Raises
    ------
    ValueError
        If density is too high for the given number of modules, or P_inter is too high.

    Returns
    -------
    CM : NxN array of int
        Connectitivy matrix.

    """

    density_corr = density*N/(N-2)
    
    P_intra=density_corr * N_mod
    
    if P_intra > 1:
        raise ValueError(f"density too high for the given number of modules. Try a value lower than {1/N_mod:0.3g}")
    if P_inter > P_intra/N_mod:
        raise ValueError(f"P_inter ({P_inter}) must be lower than P_intra/N_mod ({P_intra:0.3g}/{N_mod}={P_intra/N_mod:0.4g})")
    if symmetrical_inter and directed:
        print("Warning: symmetrical_inter and directed cannot be True at the same time. Setting symmetrical_inter to False.")
        symmetrical_inter = False
        
    if sw_intra<0 or sw_intra>1:
        raise ValueError("sw_intra must be between 0 and 1")

    rng = np.random.default_rng(seed)
    avgN = N//N_mod
    if rand_Nintra==0:
        N_intra = np.ones(N_mod, dtype=int) * avgN
    else:
        Nmin, Nmax = int(avgN*(1-rand_Nintra/2)), int(avgN*(1+rand_Nintra/2))
        N_intra = rng.integers(Nmin, Nmax, size=N_mod) 
    
    while np.sum(N_intra) < N:
        if rng.uniform()>rand_Nintra:
            N_intra[np.argmin(N_intra)]+=1
        else:
            index=rng.integers(N_mod)
            while N_intra[index]>avgN*2:
                index=rng.integers(N_mod)
            N_intra[index]+=1
    
    while np.sum(N_intra) > N:
        if rng.uniform()>rand_Nintra:
            N_intra[np.argmax(N_intra)]-=1
        else:
            index=rng.integers(N_mod)
            while N_intra[index]<avgN/2:
                index=rng.integers(N_mod)
            N_intra[index]-=1
    
    end_indices = np.cumsum(N_intra)
    start_indices = end_indices - N_intra
    
    CMintra=np.zeros((N,N), dtype=int)
    
    for i,Ni in enumerate(N_intra):
        ii,jj = start_indices[i],end_indices[i]
        if sw_intra==1:
            CMintra[ii:jj,ii:jj]=rng.binomial(1,P_intra,size=(Ni,Ni))
        else:
            CMintra[ii:jj,ii:jj]=SW(Ni, density=P_intra, Prw=sw_intra)
    if symmetrical_intra:
        CMintra[np.tril_indices(N)]=0
        CMintra += CMintra.T
    if not diag:
        CMintra[np.diag_indices(N)]=0
    
    # print("density CMintra", calc_density(CMintra))
    
    CMinterU=rng.binomial(1,P_inter,(N,N))
    for i,Ni in enumerate(N_intra):
        ii,jj = start_indices[i],end_indices[i]
        CMinterU[ii:jj,ii:jj]=0
    if directed or symmetrical_inter:
        CMinterU[np.tril_indices(N)]=0        
    if symmetrical_inter:
        CMinter=CMinterU+CMinterU.T
        CMinter[CMinter>1]=1
    else:
        CMinter=CMinterU
    
    # print("density CMinter", calc_density(CMinter))
    
    for i in range(N):
        connections=np.where(CMinterU[i])[0]  #Connections we'll remove
        if symmetrical_inter:
            connections=connections[connections>i]
        for j in connections:
            if rng.uniform()>0.5 and np.sum(CMintra[i])>0:
                j2=rng.choice(np.where(CMintra[i]==1)[0])
                CMintra[i,j2]=0
                if symmetrical_intra:
                    CMintra[j2,i]=0
            elif np.sum(CMintra[:,j])>0:
                i2=rng.choice(np.where(CMintra[:,j]==1)[0])
                CMintra[i2,j]=0
                if symmetrical_intra:
                    CMintra[j,i2]=0
    
    CM=CMintra+CMinter
    # print("density CMintra corregido", calc_density(CMintra))
    # print("density CM", calc_density(CM))

    return CM

def HierModularNet(N, N_mod=16, density=0.05, rand_Nintra=0.5, N_inter=20,
                    dNinter=0.6, rand_inter=0, sw_intra = 1, directed = False, 
                    symmetrical_inter = False, inter_hubs=True, seed=None ):
    """
    Build a modular network with hierarchical inter-module connections

    Parameters
    ----------
    N : int
        Total number of nodes.
    N_mod : int, optional
        Number of modules. The default is 16.
    density : float, optional
        Initial density of the network. Intra-module density will be density * N_mod.
        0 < density < 1/N_mod. The default is 0.05.
    rand_Nintra : float, optional
        Random factor of module sizes. If 0, all modules will be (mostly) the
        same size. If 1, module sizes will be random between 0.5 and 1.5 times the
        average size. The default is 0.5.
    N_inter : int, optional
        Number of connection to establish between modules. The default is 20.
    dNinter : float, optional
        Factor by which the number of inter-module connection is multiplied as the 
        hierarchy increases. If 1, all hierarchies will contain the same amount
        of connections. dNinter > 1 means connections will increase. The default is 0.6.
    rand_inter : float, optional
        If greater than 0, a number of unspecific (random) inter-module 
        connections will be added. The default is 0.
    sw_intra : float, optional. 0 <= sw_intra <=1
        Small-world parameter for intra-module connections. If sw_intra==1 (Default),
        modules are completely random. If 0, modules have a lattice structure.
        With values between 0.01 and 0.1, modules will have small-world architecture.
    directed : Boolean, optional
        if True, inter-module connections are established only to one side 
        of the diagonal. Default = False
    symmetrical_inter : Boolean, optional
        if True, CM[i,j]=CM[j,i] for connections outside modules. Note that 
        symmetrical_inter cannot be True if directed is True. Default = True.
    inter_hubs : Boolean, optional
        If True, the nodes that make inter-module connections will be maintained
        between hierarchies. The default is True.
    seed : Integer or None
        random seed. default: None


    Raises
    ------
    ValueError
        DESCRIPTION.

    Returns
    -------
    CM : TYPE
        DESCRIPTION.

    """

    density_corr = density*N/(N-2)
        
    P_intra=density_corr * N_mod
        
    if P_intra > 1:
        raise ValueError(f"density too high for the given number of modules. Try a value lower than {1/N_mod:0.3g}")
    if N_mod%2 != 0:
        raise ValueError("Number of modules must be even")
    if sw_intra<0 or sw_intra>1:
        raise ValueError("sw_intra must be between 0 and 1")
    
    rng = np.random.default_rng(seed)
    avgN = N//N_mod
    if rand_Nintra==0:
        N_intra = np.ones(N_mod, dtype=int) * avgN
    else:
        Nmin, Nmax = int(avgN*(1-rand_Nintra/2)), int(avgN*(1+rand_Nintra/2))
        N_intra = rng.integers(Nmin, Nmax, size=N_mod)  # np.ones(N_mod, dtype=int) * (N//N_mod)
    
    while np.sum(N_intra) < N:
        if rng.uniform()>rand_Nintra:
            N_intra[np.argmin(N_intra)]+=1
        else:
            index=rng.integers(N_mod)
            while N_intra[index]>avgN*2:
                index=rng.integers(N_mod)
            N_intra[index]+=1
            
    while np.sum(N_intra) > N:
        if rng.uniform()>rand_Nintra:
            N_intra[np.argmax(N_intra)]-=1
        else:
            index=rng.integers(N_mod)
            while N_intra[index]<avgN/2:
                index=rng.integers(N_mod)
            N_intra[index]-=1
    
    end_indices = np.cumsum(N_intra)
    start_indices = end_indices - N_intra
    
    CMintra=np.zeros((N,N), dtype=int)
    
    for i,Ni in enumerate(N_intra):
        ii,jj = start_indices[i],end_indices[i]
        if sw_intra==1:
            CMintra[ii:jj,ii:jj]=rng.binomial(1,P_intra,size=(Ni,Ni))
        else:
            CMintra[ii:jj,ii:jj]=SW(Ni, density=P_intra, Prw=sw_intra)
    CMintra[np.diag_indices(N)]=0
    CMintra[np.tril_indices(N)]=0
    CMintra += CMintra.T
    
    CMinterU=np.zeros((N,N), dtype=np.int32)
    
    hierarchies = int(np.ceil(np.log2(N_mod)))
    
    for h in range(hierarchies):
        for n in range(0,N_mod,2**(h+1)):
            if (n + 2**h + 1) <= N_mod:
                iMod = n
                jMod = n + 2**h
                # print(h,n,iMod, jMod)
                ii1,ii2=start_indices[iMod], end_indices[iMod]
                jj1,jj2=start_indices[jMod], end_indices[jMod]
                
                if inter_hubs and h>0:
                    jPrev= n + 2**(h-1)
                    jj1P,jj2P=start_indices[jPrev], end_indices[jPrev]
                    Prev_i_conn = np.where(CMinterU[ii1:ii2,jj1P:jj2P])[0] + ii1
                    i1=rng.choice(Prev_i_conn,size=N_inter, replace=False)
                    
                    jPrev= n + 2**h + 2**(h-1)
                    h_aux=h
                    while jPrev >= N_mod:
                        h_aux -=1
                        jPrev = n + 2**h + 2**(h_aux-1)
                    jj1P,jj2P=start_indices[jPrev], end_indices[jPrev]
                    Prev_j_conn = np.where(CMinterU[jj1:jj2,jj1P:jj2P])[0] + jj1
                    i2=rng.choice(Prev_j_conn,size=N_inter, replace=False)
                                
                else:
                    i1=rng.choice(range(ii1,ii2),size=N_inter)
                    i2=rng.choice(range(jj1,jj2),size=N_inter)
                CMinterU[i1,i2]=1
                
                if not directed and not symmetrical_inter:
                    ii1,ii2=start_indices[jMod], end_indices[jMod]
                    jj1,jj2=start_indices[iMod], end_indices[iMod]
                    
                    if inter_hubs and h>0:
                        iPrev= n + 2**(h-1) + 2**h
                        h_aux=h
                        while iPrev >= N_mod:
                            h_aux -=1
                            iPrev = n + 2**h + 2**(h_aux-1)
                        ii1P,ii2P=start_indices[iPrev], end_indices[iPrev]
                        Prev_i_conn = np.where(CMinterU[ii1P:ii2P,ii1:ii2])[1] + ii1
                        i1=rng.choice(Prev_i_conn,size=N_inter, replace=False)
                        
                        iPrev= n + 2**(h-1)
                        ii1P,ii2P=start_indices[iPrev], end_indices[iPrev]
                        Prev_j_conn = np.where(CMinterU[ii1P:ii2P,jj1:jj2])[1] + jj1
                        i2=rng.choice(Prev_j_conn,size=N_inter, replace=False)
                    else:
                        i1=rng.choice(range(ii1,ii2),size=N_inter)
                        i2=rng.choice(range(jj1,jj2),size=N_inter)
                    
                    CMinterU[i1,i2]=1
                    
        N_inter = int(np.ceil(N_inter*dNinter))
    
    if symmetrical_inter:
        CMinterU+=CMinterU.T
        
    if rand_inter>0:
        CMinterU2=rng.binomial(1,rand_inter,(N,N))
        CMinterU2[np.diag_indices(N)]=0
        if directed or symmetrical_inter:
            CMinterU2[np.tril_indices(N)]=0
        for i,Ni in enumerate(N_intra):
            ii,jj = start_indices[i],end_indices[i]
            CMinterU2[ii:jj,ii:jj]=0
        if symmetrical_inter:
            CMinterU2+=CMinterU2.T
    else:
        CMinterU2=np.zeros((N,N), dtype=np.int16)            
    
    CM=CMintra+CMinterU+CMinterU2
    CM[CM>1]=1
    
    return CM
    
def density(M,diag=False):
    """
    Quick calculation of network connection density

    Parameters
    ----------
    M : 2D np.array
        Connectivity (adjacency) matrix.
    diag : Boolean, optional
        If True, then the diagonal entries are considered. This should be the case
        when the matrix describes the connection between *different* populations.
        If the matrix is the connection from one population with itself, this should
        be False. The default is False.

    Returns
    -------
    Float
        Fraction of connections present (being 1 when all possible connections are True).

    """

    m,n=np.shape(M)
    if m!=n:
        diag=True
    
    if diag: 
        return np.sum(M)/np.prod(M.shape)
    else:
        return np.sum(M)/(m*(n-1))
    


if __name__=="__main__":
    
    import matplotlib.pyplot as plt
    
    plt.figure(1,figsize=(9,5))
    plt.clf()
    
    CM1=SW(200,density=0.2,Prw=0.2)
    plt.subplot(231)
    plt.imshow(CM1,cmap='gray_r')
    plt.title("SW, 200x200 symm, dens=0.2, p=0.2",size='x-small')

    CM2=SW(100,75,density=0.2,Prw=0,directed=True)
    plt.subplot(232)
    plt.imshow(CM2,cmap='gray_r')
    plt.title("SW, 100x75 dir, dens=0.15, p=0",size='x-small')
    
    CM3=SW(100,125,density=0.2,Prw=0.1,symmetrical=False)
    plt.subplot(233)
    plt.imshow(CM3,cmap='gray_r')
    plt.title("SW, 100x125 asymm, dens=0.2, p=0.1",size='x-small')    
    
    CM4=distCM(200,200,density=0.2,rnd=0.1,symmetrical=True)
    plt.subplot(234)
    plt.imshow(CM4,cmap='gray_r')
    plt.title("distance, 200x200 symm, dens=0.2, rnd=0.1",size='x-small')
    
    CM5=distCM(75,100,density=0.2,rnd=0,directed=True)
    plt.subplot(235)
    plt.imshow(CM5,cmap='gray_r')
    plt.title("distance, 75x100 directed, dens=0.2, rnd=0",size='x-small')
    
    CM6=distCM(250,200,density=0.2,rnd=0.05,directed=False,symmetrical=False)
    plt.subplot(236)
    plt.imshow(CM6,cmap='gray_r')
    plt.title("distance, 250x200 asymm, dens=0.2, rnd=0.05",size='x-small')

    plt.tight_layout()
    
    #%%
    CM_mod1=ModularNet(260, N_mod=8, sw_intra= 0.05, P_inter=0.02)
    CM_mod2=ModularNet(260, N_mod=8, directed=True)
    CM_mod3=ModularNet(260, N_mod=8, symmetrical_inter=False)
    CM_mod4=HierModularNet(260, N_mod=8, sw_intra=0.05)
    
    plt.figure(2)
    plt.clf()
    for i,CM in enumerate((CM_mod1, CM_mod2, CM_mod3, CM_mod4)):
        plt.subplot(2,2,i+1)
        plt.imshow(CM, cmap='gray_r')
        print(f"density CMmod{i+1}: {density(CM)}")
    
    
    
