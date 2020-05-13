# -*- coding: utf-8 -*-
"""
demo3DfilterInterp
Test random noise attenuation and interpolation of 3D data sets (synthetic and real)
via low rank approximation and reinsertion algorithm.

"""

import numpy as np
import matplotlib.pyplot as plt
import copy
import scipy.io


def MSSA3D(d,dt,k,flow,fhigh):
    """
    d(t,x1,x2) --> d(f,x1,x2) --> SSA --> dout(f,x1,x2) --> dout(t,x1,x2)
    """
    [nt,nx,ny] = d.shape                         # Number of time samples and traces
    nf = 2*(1<<(nt-1).bit_length())              # Number of frequency samples for FFT
    doutFX = np.zeros((nf,nx,ny),dtype=complex)  # Filtered data in FX
    dout = np.zeros((nf,nx,ny))                  # Filtered data in tX (the output)
    # First and last samples of the DFT
    ilow  = int(np.floor(flow*dt*nf)+1)
    if ilow < 1:
        ilow = 1 
    
    ihigh = int(np.floor(fhigh*dt*nf)+1)
    if ihigh > np.floor(nf/2)+1:
        ihigh = np.floor(nf/2)+1

    # Transform data to Frequency-Space domain
    dtempFX = np.fft.fft(d,n=nf,axis=0)

    # Dimension of Hankel Matrix in y
    Ncol = int(np.floor(ny/2)+1)
    Nrow = int(ny-Ncol+1)
    
    # Dimensions of Block Hankel Matrix composed of Hankel matrices
    Lcol = int(np.floor(nx/2)+1)
    Lrow = int(nx-Lcol+1)

    # For all frequencies in the signal
    for j in range(ilow-1,ihigh):
        
        # For each frequency slice, form a Level-2 Hankel matrix
        #H = np.zeros((Nrow*Lrow,Lrow*Lcol),dtype=complex)
        H = np.zeros((Nrow*Lrow,Ncol*Lcol),dtype=complex)
        H = L2Hankel(Lrow,Lcol,Nrow,Ncol,H,dtempFX[j,:,:])
        
        # Low rank approximation
        Hout = truncatedSVD(H,k)
        
        # Recover signal 
        doutFX[j,:,:] = aveL2(Hout,nx,ny,Lcol,Lrow,Ncol,Nrow)    

    # Complex conjugate symmetry
    for i in range(int(nf/2+1),nf):
        doutFX[i,:,:] = np.conjugate(doutFX[nf-i,:,:])
            
    # Back to tX domain  
    dout = np.fft.ifft(doutFX,n=nf,axis=0).real
    #dout = np.swapaxes(np.fft.ifft(doutFX,n=nf,axis=0).real,1,2)
    #dout = dout[:nt,:,:]
        
    return dout[:nt,:,:]



def L2Hankel(Lrow,Lcol,Nrow,Ncol,H,dtempFX):    
    """
    Forms a complex-valued level-2 Hankel matrix from a 2D frequency slice.
    
    Example:
    --------
    >>> dtempFX = np.array([[0,1,2],[0,1,2],[0,1,2]])
    >>> dtempFX
    array([[0, 1, 2],
           [0, 1, 2],
           [0, 1, 2]]])
    
    >>> L2Hankel(2,2,2,2,H=np.zeros((4,4)),dtempFX)
    array([[0., 1., 0., 1.],
           [1., 2., 1., 2.],
           [0., 1., 0., 1.],
           [1., 2., 1., 2.]])
    """
    
    for lc in range(Lcol):
        for lr in range(Lrow):
            tmp = dtempFX[lr+lc,:]
            for ic in range(Ncol):
                for ir in range(Nrow):
                    H[lr*Nrow+ir,lc*Ncol+ic] = tmp[ir+ic]
        
        
    return H


def truncatedSVD(A,K):
    """
    Performs SVD and computes the low rank approximation of
    matrix A for specified number of linear events k.
    
    A = UsV*
    Aout = U[:k]s[:k]V[:k]* 
    """
    A = np.nan_to_num(A)
    U,s,V = np.linalg.svd(A,full_matrices=False)
    Aout = np.dot(U[:,:K],np.dot(np.diag(s[:K]),V[:K,:]))
    
    return Aout


def aveL2(Hout,nx,ny,Lcol,Lrow,Ncol,Nrow):
    """
    Computes average of each anti-diagonal of each block the of level-2
    Hankel matrix A, to recover the signal.
    Returns a 2D array (frequency slice) of size [nx,ny].
    
    Example:
    --------
    >>> A = np.array([[0.,1.,0.,1.],[1., 2., 1., 2.],[0., 1., 0., 1.],[1., 2., 1., 2.]])
    >>> A
    array([[0., 1., 0., 1.],
           [1., 2., 1., 2.],
           [0., 1., 0., 1.],
           [1., 2., 1., 2.]])
    
    >>> aveL1(A,nx=3,ny=3,Lcol=2,Lrow=2,Ncol=2,Nrow=2)
    array([[0, 1, 2],
           [0, 1, 2],
           [0, 1, 2]]])
    """    
    count = np.zeros((ny,nx))                
    tmp = np.zeros((ny,nx),dtype=complex)
    
    for lc in range(Lcol):
        for lr in range(Lrow):
            for ic in range(Ncol):
                for ir in range(Nrow):
                    count[ir+ic,lc+lr] = count[ir+ic,lc+lr]+1
                    tmp[ir+ic,lc+lr] = tmp[ir+ic,lc+lr] + Hout[lr*Nrow+ir,lc*Ncol+ic]
    
    return np.swapaxes(np.divide(tmp,count),0,1)


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
    Creates a d.shape 3D array S, in which elements =1 if there exists
    an observation and =0 if there is a missing observation.
    
    Decimates original data randomly by a specified percentage.
    
    perc  - Amount of missing traces, in fraction
    d     - Original data (Complete).
    dout  - Data with missing observations.
    S     - Sampling operator.
    
    Example:
    --------
    >>> d = np.ones(3,4,1)
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
        for j in range(data.shape[2]):
            if  np.random.uniform()<perc:
                S[:,i,j] = 0 
                dsamp[:,i,j] = 0
        
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
      tmp = MSSA3D(dout,dt,k,flow,fhigh) 
      dout =  a*din + np.multiply((1-a*S),tmp)
   
    return dout





"""
Load 3D data and extract a vertical slice: d(t,x,y) -> d(t,y).
Load 2D data volume, each slice represents the same slice with different SNR.


d0   -  Clean signal d0(t,x,y).
d    -  Signal with random noise d(t,x,y).
dsnr -  Volume composed of 2D profile with different SNR values.
dt   -  Time sampling, seconds.
dx   -  Sampling interval in y direction, y units.
dy   -  Sampling interval in y direction, y units.
t    -  Time-axis, seconds.
x    -  x-axis, x-units.
y    -  y-axis, y-units.
snr  -  Signal to noise ratio for 2D slice from 3D data.
SNR  -  Signal to noise ratio array for volume compused of 2D slices.
"""    

npzfile = np.load('data.npz')         # Load synthetic data, SNR = 50%
print(npzfile.files)                  # Display all variables 
d0 = npzfile['d0']
d = npzfile['d']
dt = npzfile['dt']
x = npzfile['x']
y = npzfile['y']
t = npzfile['t']
real = scipy.io.loadmat('real_cube.mat') # Real field data
dreal = real['d']
dreal = dreal[200:400,80:120,23:83]      # Take a subset
#dreal = dreal/dreal.max(axis=0)          # Normalize traces  



"""
3D noise attenuation and interpolation examples begin here.

These are some synthetic and real data examples.
"""


# Denoising and interpolation synthetic (snr=0.5, 50% empty, k=optimal)
# =============================================================================

k = 4
flow = 1
fhigh = 120
a = 0.4                          # a=1 : Full replacement of observations, a=0 : pure denoising
niter = 10                       # Number of iterations
perc = 0.5                       # Amount of missing traces, in fraction       
S,dsamp = sampling(d, perc)      # Decimate data and create sampling operator

dr = reinsert(k,flow,fhigh,a,niter,dsamp,S) 


# Clean full data
fig,axs = plt.subplots(nrows=1, ncols=3, sharex=True)

ax = axs[0]
ax.imshow(d0[:,5,:],cmap='Greys', interpolation='nearest')
ax.set_yticklabels(t)
ax.set_title('Slice though x')
ax.set_xlabel('y')
ax.set_xlim(0,len(y))
ax.set_ylabel('time [s]')

ax = axs[1]
ax.imshow(d0[:,:,5],cmap='Greys', interpolation='nearest')
ax.set_yticklabels(t)
ax.set_title('Slice though y')
ax.set_xlim(0,len(x))
ax.set_xlabel('x')
ax.set_ylabel('time [s]')

ax = axs[2]
im = ax.imshow(d0[100,:,:],cmap='Greys', interpolation='nearest')
ax.set_title('Time slice ')
ax.set_xlabel('x')
ax.set_ylabel('y')

fig.suptitle('Full clean data')  
fig.subplots_adjust(right=0.8)
cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
fig.colorbar(im, cax=cbar_ax)
plt.show()


# Input
fig,axs = plt.subplots(nrows=1, ncols=3, sharex=True)

ax = axs[0]
ax.imshow(dsamp[:,5,:],cmap='Greys', interpolation='nearest')
ax.set_yticklabels(t)
ax.set_title('Slice though x')
ax.set_xlabel('y')
ax.set_xlim(0,len(y))
ax.set_ylabel('time [s]')

ax = axs[1]
ax.imshow(dsamp[:,:,5],cmap='Greys', interpolation='nearest')
ax.set_yticklabels(t)
ax.set_title('Slice though y')
ax.set_xlim(0,len(x))
ax.set_xlabel('x')
ax.set_ylabel('time [s]')

ax = axs[2]
im = ax.imshow(dsamp[100,:,:],cmap='Greys', interpolation='nearest')
ax.set_title('Time slice ')
ax.set_xlabel('x')
ax.set_ylabel('y')

fig.suptitle('Input')  
fig.subplots_adjust(right=0.8)
cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
fig.colorbar(im, cax=cbar_ax)
plt.show()


# Output
fig,axs = plt.subplots(nrows=1, ncols=3, sharex=True)

ax = axs[0]
ax.imshow(dr[:,5,:],cmap='Greys', interpolation='nearest')
ax.set_yticklabels(t)
ax.set_title('Slice though x')
ax.set_xlabel('y')
ax.set_xlim(0,len(y))
ax.set_ylabel('time [s]')

ax = axs[1]
ax.imshow(dr[:,:,5],cmap='Greys', interpolation='nearest')
ax.set_yticklabels(t)
ax.set_title('Slice though y')
ax.set_xlim(0,len(x))
ax.set_xlabel('x')
ax.set_ylabel('time [s]')

ax = axs[2]
im = ax.imshow(dr[100,:,:],cmap='Greys', interpolation='nearest')
ax.set_title('Time slice ')
ax.set_xlabel('x')
ax.set_ylabel('y')

fig.suptitle('Ouput')  
fig.subplots_adjust(right=0.8)
cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
fig.colorbar(im, cax=cbar_ax)
plt.show()


# Difference
fig,axs = plt.subplots(nrows=1, ncols=3, sharex=True)

ax = axs[0]
ax.imshow(dr[:,5,:]-d0[:,5,:],cmap='Greys', interpolation='nearest')
ax.set_yticklabels(t)
ax.set_title('Slice though x')
ax.set_xlabel('y')
ax.set_xlim(0,len(y))
ax.set_ylabel('time [s]')
#ax.colorbar()

ax = axs[1]
ax.imshow(dr[:,:,5]-d0[:,:,5],cmap='Greys', interpolation='nearest')
ax.set_yticklabels(t)
ax.set_title('Slice though y')
ax.set_xlim(0,len(x))
ax.set_xlabel('x')
ax.set_ylabel('time [s]')
#ax.colorbar()

ax = axs[2]
im = ax.imshow(dr[100,:,:]-d0[100,:,:],cmap='Greys', interpolation='nearest')
ax.set_title('Time slice ')
ax.set_xlabel('x')
ax.set_ylabel('y')
#ax.colorbar()

fig.suptitle('Difference')  
fig.subplots_adjust(right=0.8)
cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
fig.colorbar(im, cax=cbar_ax)
plt.show()

# =============================================================================

##### For some reason I am not able to get good results with real data
# Denoising and interpolation Real
# =============================================================================

k = 3           # Low k for harsher denoising                    
      
S,dsamp = sampling(dreal, perc)

dr = reinsert(k,flow,fhigh,a,niter,dsamp,S) 


# Decimated real data (Input)
fig,axs = plt.subplots(nrows=1, ncols=3, sharex=True)

ax = axs[0]
ax.imshow(dreal[:,20,:],cmap='Greys', interpolation='nearest')
ax.set_title('Slice though x')
ax.set_xlabel('x')
ax.set_ylabel('time')

ax = axs[1]
ax.imshow(dreal[:,:,30],cmap='Greys', interpolation='nearest')
ax.set_title('Slice though y')
ax.set_xlabel('x')
ax.set_ylabel('time')

ax = axs[2]
ax.imshow(dreal[39,:,:],cmap='Greys', interpolation='nearest')

ax.set_title('Time slice ')
ax.set_xlabel('x')
ax.set_ylabel('y')

fig.suptitle('Real data')  

plt.show()


# Output
fig,axs = plt.subplots(nrows=1, ncols=3, sharex=True)

ax = axs[0]
ax.imshow(dr[:,20,:],cmap='Greys', interpolation='nearest')
ax.set_title('Slice though x')
ax.set_xlabel('x')
ax.set_ylabel('time')

ax = axs[1]
ax.imshow(dr[:,:,30],cmap='Greys', interpolation='nearest')
ax.set_title('Slice though y')
ax.set_xlabel('x')
ax.set_ylabel('time')

ax = axs[2]
ax.imshow(dr[39,:,:],cmap='Greys', interpolation='nearest')
ax.set_title('Time slice ')
ax.set_xlabel('x')
ax.set_ylabel('y')

fig.suptitle('Output k=%d' %k)  

plt.show()


# =============================================================================











