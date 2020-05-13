# -*- coding: utf-8 -*-
"""
demo2Dfilterinterp
Test random noise attenuation and interpolation of a 2D data set
via low rank approximation and reinsertion algorithm.

"""
import numpy as np
import matplotlib.pyplot as plt
import copy


def SSA(d,dt,k,flow,fhigh):
    """
    d(t,x) --> d(f,x) --> SSA --> dout(f,x) --> dout(t,x)
    """
    [nt,nx] = d.shape                         # Number of time samples and traces
    nf = 2*(1<<(nt-1).bit_length())           # Number of frequency samples for FFT
    doutFX = np.zeros((nf,nx),dtype=complex)  # Filtered data in FX
    dout = np.zeros((nf,nx))                  # Filtered data in tX (the output)
    # First and last samples of the DFT
    ilow  = int(np.floor(flow*dt*nf)+1)
    if ilow < 1:
        ilow = 1 
    
    ihigh = int(np.floor(fhigh*dt*nf)+1)
    if ihigh > np.floor(nf/2)+1:
        ihigh = np.floor(nf/2)+1

    # Transform data to Frequency-Space domain
    dtempFX = np.fft.fft(d,n=nf,axis=0)
    
    # Dimensions of Hankel Matrix
    Lcol = int(np.floor(nx/2)+1)
    Lrow = int(nx-Lcol+1)

    # For all frequencies in the signal
    for j in range(ilow-1,ihigh):
        
        # For each frequency slice, form a Level-1 Hankel matrix
        H = np.zeros((Lrow,Lcol),dtype=complex)
        H = L1Hankel(Lrow,Lcol,H,dtempFX[j,:])
        
        # Low rank approximation
        Hout = truncatedSVD(H,k)
        
        # Recover signal 
        doutFX[j,:] = aveL1(Hout,nx,Lcol,Lrow)    

    # Complex conjugate symmetry
    for i in range(int(nf/2+1),nf):
        doutFX[i,:] = np.conjugate(doutFX[nf-i,:])
            
    # Back to tX domain    
    dout = np.fft.ifft(doutFX,n=nf,axis=0).real
    dout = dout[:nt,:]
        
    return dout



def L1Hankel(Lrow,Lcol,H,dtempFX):    
    """
    Forms a complex-valued level-1 Hankel matrix from a 1D frequency slice.
    
    Example:
    --------
    >>> dtempFX = np.array([0,1,2,3])
    >>> dtempFX
    array([0, 1, 2, 3])
    
    >>> L1Hankel(Lrow=2,Lcol=3,H=np.zeros((2,3)),dtempFX)
    array([[0., 1., 2.],
           [1., 2., 3.]])
    """
    for lc in range(Lcol):
        H[:,lc]  = dtempFX[lc:lc+Lrow]
        
    return H


def truncatedSVD(A,K):
    """
    Performs SVD and computes the low rank approximation of
    matrix A for specified number of linear events k.
    
    A = UsV*
    Aout = U[:k]s[:k]V[:k]* 
    """
    U,s,V = np.linalg.svd(A,full_matrices=False)
    Aout = np.dot(U[:,:K],np.dot(np.diag(s[:K]),V[:K,:]))
    
    return Aout


def aveL1(Hout,nx,Lcol,Lrow):
    """
    Computes average of each anti-diagonal of level-1 Hankel matrix A, to recover
    the true signal.
    Returns a 1D array (frequency slice) of size nx.
    
    Example:
    --------
    >>> A = np.array([[0., 1., 2.],[1., 2., 3.]])
    >>> A
    array([[0., 1., 2.],
           [1., 2., 3.]])
    
    >>> aveL1(A,nx=4,Lcol=3,Lrow=2)
    array([0, 1, 2, 3])
    """
    count = np.zeros((nx,))                # Length of ith anti-diagonal
    tmp = np.zeros((nx,),dtype=complex)    # Sum of ith anti-diagonal
   
    for ic in range(Lcol):
     for ir in range(Lrow):
      count[ir+ic] = count[ir+ic]+1
      tmp[ir+ic]  = tmp[ir+ic] + Hout[ir,ic]

    return np.divide(tmp,count)


def quality(dout, d0):
    """
    Quality of of output data Q, measured in dB.
    
    d0    - Clean data
    dout  - Data + noise
    
    """
    Q = 10*np.log(np.linalg.norm(d0)/np.linalg.norm(dout-d0))
    
    return Q


def sampling(data, perc):
    """
    Creates a d.shape 2D array S, in which elements =1 if there exists
    an observation and =0 if there is a missing observation.
    
    Decimates original data randomly by a specified percentage.
    
    perc  - Amount of missing traces, in fraction
    d     - Original data (Complete).
    dout  - Data with missing observations.
    S     - Sampling operator.
    
    Example:
    --------
    >>> d = np.ones(3,4)
    >>> d
    array([[1, 1, 1, 1],
           [1, 1, 1, 1],
           [1, 1, 1, 1]]) 
    
    >>> A = sampling(d, 0.5)
    >>> A
    array([[0, 1, 0, 1],
           [0, 1, 0, 1],
           [0, 1, 0, 1]])  
    """
    
    S = np.ones((data.shape))   # Sampling operator, =0 if missing observation,else =1
    dsamp = copy.copy(data)     # Reference new object (make copy of data)
    
    # Remove traces randomly
    for i in range(data.shape[1]):
        if  np.random.uniform()<perc:
            S[:,i] = 0 
            dsamp[:,i] = 0
    

    
    return S, dsamp


def reinsert(k,flow,fhigh,a,niter,din,S):
    """
    This iterative method for recovering missing traces consists on finding the
    low rank approximation of each frequency slice for the whole data set and
    inserting the original traces in time domain.
    
    a = 1   : For interpolating noise-less data.
    0<a<1   : For interpolating data in presence of noise.
    a(iter) : Can also decrease a each iteration such that last iteration is
              pure denoising.
    
    """
    dout = copy.copy(din)                # Initial estimate of reconstructed data
    
    for i in range(niter):
      tmp = SSA(dout,dt,k,flow,fhigh) 
      dout =  a*din + np.multiply((1-a*S),tmp)
   
    return dout





"""
Load 3D data and extract a vertical slice: d(t,x,y) -> d(t,y).


d0   -  Clean signal d0(t,x,y).
d    -  Signal with random noise d(t,x,y).
dt   -  Time sampling, seconds.
dx   -  Sampling interval in y direction, y units.
dy   -  Sampling interval in y direction, y units.
t    -  Time-axis, seconds.
x    -  x-axis, x-units.
y    -  y-axis, y-units.
snr  -  Signal to noise ratio for 2D slice from 3D data.
"""   

npzfile = np.load('data.npz')         # Load synthetic data, SNR = 50%
print(npzfile.files)                  # Display all variables 
d0 = npzfile['d0']
d = npzfile['d']
dt = npzfile['dt']
x = npzfile['x']
y = npzfile['y']
t = npzfile['t']

# Chose a random profile along x-direction.
profile = int(np.random.randint(low=0, high=len(x), size=1))
d = d[:,profile,:]
d0 = d0[:,profile,:]


"""
2D denoising and interpolation examples begin here.

These are some examples of the quality of data recovery as a function of k.
"""


# Distribution of singular values
# =============================================================================

# Distribution for clean data
dFX = np.fft.fft(d0,axis=0)   
Lcol = int(np.floor(dFX.shape[1]/2)+1)
Lrow = int(dFX.shape[1]-Lcol+1)

#Amplitude of singular values as a function of frequency
sClean = np.zeros((dFX.shape[0],Lrow))  
for iw in range(dFX.shape[0]):
    # Frequency slice
    A = dFX[iw,:] 
    # Hankel matrix for frequency slice
    B = L1Hankel(Lrow,Lcol,np.zeros((Lrow,Lcol),dtype=complex),A)
    U,sigma,V = np.linalg.svd(B,full_matrices=False)
    if iw == 20:
        Hclean = copy.copy(B)
    sClean[iw,:] = sigma


# Distribution for corrupted data
perc = 0.5                       # Amount of missing traces, in fraction       
S,dsamp = sampling(d, perc)      # Decimate data and create sampling operator
dFX = np.fft.fft(dsamp,axis=0)  

#Amplitude of singular values as a function of frequency
sBad = np.zeros((dFX.shape[0],Lrow))  
for iw in range(dFX.shape[0]):
    # Frequency slice
    A = dFX[iw,:] 
    # Hankel matrix for frequency slice
    B = L1Hankel(Lrow,Lcol,np.zeros((Lrow,Lcol),dtype=complex),A)
    U,sigma,V = np.linalg.svd(B,full_matrices=False)
    if iw == 20:
        Hbad = copy.copy(B)
    sBad[iw,:] = sigma


plt.figure()
plt.imshow(sClean, cmap = 'hot')
plt.xlabel('Singular value Index')
plt.ylabel('Hz')
plt.colorbar()
plt.show()

plt.figure()
plt.imshow(sBad, cmap="hot")
plt.xlabel('Singular value Index')
plt.ylabel('Hz')
plt.colorbar()
plt.show()

plt.figure()
plt.imshow(Hclean.real,cmap='Greys')
plt.colorbar()
plt.show()

plt.figure()
plt.imshow(Hbad.real,cmap='Greys')
plt.colorbar()
plt.show()

# =============================================================================


# Filtering + Interpolation as a function of rank
# =============================================================================

perc = 0.5                       # Amount of missing traces, in fraction       
S,dsamp = sampling(d, perc)      # Decimate data and create sampling operator

# Input
fig,ax = plt.subplots()
ax.imshow(dsamp,cmap='Greys', interpolation='nearest')
ax.set_xticklabels(x)
plt.title('Input')
plt.xlabel('x')
plt.ylabel('time [s]')
ax.set_yticklabels(t)
plt.show()


rank = np.arange(1,8)         # Number of distinct dips in window of analysis
flow = 1
fhigh = 120
a = 0.4           # a=1 : Full replacement of observations, a=0 : pure denoising
niter = 10        # Number of iterations
Qin = quality(dsamp,d0)     # Quality of input data dB

Qout = np.zeros((rank.shape))    
for k in rank:
    
    dr = reinsert(k,flow,fhigh,a,niter,dsamp,S) 
    Qout[k-1] = quality(dr,d0)
    
    if k in np.array([3,4]):
        # Filtered data (output)
        fig,ax = plt.subplots()
        ax.imshow(dr,cmap='Greys', interpolation='nearest')
        ax.set_xticklabels(x)
        plt.title('Output k=%d' %k)
        plt.xlabel('x')
        plt.ylabel('time [s]')
        ax.set_yticklabels(t)
        plt.show()
        
        # Difference
        fig,ax = plt.subplots()
        ax.imshow(dr-d0,cmap='Greys', interpolation='nearest')
        ax.set_xticklabels(x)
        plt.title('Differnce')
        plt.xlabel('x')
        plt.ylabel('time [s]')
        ax.set_yticklabels(t) 
        plt.show()

    
# Quality
fig, ax = plt.subplots()
textstr = r'$Q_{Input}=%.2f dB$' % (Qin, )
ax.plot(rank,Qout,'ko-')
plt.xlabel('k')
plt.ylabel('Q(dB)')
ax.text(0.1, 0.2, textstr, transform=ax.transAxes, fontsize=14)
plt.show()

# =============================================================================




