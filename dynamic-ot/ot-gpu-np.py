import numpy as np
import cupy as cp
import mrcfile
from math import pi
from numpy.fft import fft2, ifft2,fftn,ifftn
from scipy.ndimage import convolve
from skimage.transform import warp
import matplotlib.pyplot as plt
from scipy.fft import dctn,idctn
from kernel_cubic import cubic
import imageio
from PIL import Image
P = 240
Q = 240
R = 240
T = 20
d=[P,Q,T]
epsilon = 1e-8

class staggered():
    """
    Vector field on staggered grid
    (M1,M2,M3) is the 3D coordinate of momentum M
    F is the density function(space&time), with boundary in time domain being fixed
    axis 0,1,2(P,Q,R) represent space domain, axis 3(T) represents time domain
    """
    def __init__(self):
        self.M1 = np.zeros([P+1, Q, R, T])
        self.M2 = np.zeros([P, Q+1, R, T])
        self.M3 = np.zeros([P, Q, R+1, T])
        self.F  = np.zeros([P, Q, R, T+1])
        self.F[:, :, :,  0] = i1
        self.F[:, :, :, -1] = i2

def interp(U):
    """
        Interpolation operator from staggered grid to centered grid
        Input: an object on staggered grid
        Output: an object on centered grid
        Note: Object on centered grid is of size 4*P*Q*R*T
        dimension 0,1,2 of axis 0 represents momentum M1,M2,M3
        dimension 3 of axis 0 represents density function F
        axis 1,2,3(P,Q,R) represent space domain, axis 4(T) represents time domain
    """
    V  = np.zeros([4, P, Q, R, T])
    M1 = U.M1.copy()
    M2 = U.M2.copy()
    M3 = U.M3.copy()
    F  = U.F.copy()
    V[0, :, :, :, :] = (M1[:-1, :, :, :] + M1[1:, :, :, :])/2
    V[1, :, :, :, :] = (M2[:, :-1, :, :] + M2[:, 1:, :, :])/2
    V[2, :, :, :, :] = (M3[:, :, :-1, :] + M3[:, :, 1:, :])/2
    V[3, :, :, :, :] = ( F[:, :, :, :-1] +  F[:, :, :, 1:])/2
    return V

def interp_ad(V):
    """
        Interpolation operator from centered grid to staggered grid
        Input: an object on centered grid
        Output: an object on staggered grid
        Note: Object on centered grid is of size 4*P*Q*R*T
        dimension 0,1,2 of axis 0 represents momentum M1,M2,M3
        dimension 3 of axis 0 represents density function F
        axis 1,2,3(P,Q,R) represent space domain, axis 4(T) represents time domain
    """
    U = staggered()
    U.M1[1:-1, :, :, :] = (V[0, 1:, :, :, :] + V[0, :-1, :, :, :])/2
    U.M2[:, 1:-1, :, :] = (V[1, :, 1:, :, :] + V[1, :, :-1, :, :])/2
    U.M3[:, :, 1:-1, :] = (V[2, :, :, 1:, :] + V[2, :, :, :-1, :])/2
    U.F [:, :, :, 1:-1] = (V[3, :, :, :, 1:] + V[3, :, :, :, :-1])/2
    U.M1[ 0, :, :, :] = 0
    U.M1[-1, :, :, :] = 0
    U.M2[:,  0, :, :] = 0
    U.M2[:, -1, :, :] = 0
    U.M3[:, :,  0, :] = 0
    U.M3[:, :, -1, :] = 0
    U.F [:, :, :,  0] = 0
    U.F [:, :, :, -1] = 0
    return U

def ProxJ(V0, gamma):
    '''
    Compurte Prox_{\gamma J}(M1,M2,M3,F) on centered grid
    Input: centered grid object V0, a constant gammma
    Output: centered grid object a
    '''
    a  = np.zeros([4, P, Q, R, T])
    f0 = V0[3, :, :, :, :]
    m1 = V0[0, :, :, :, :]
    m2 = V0[1, :, :, :, :]
    m3 = V0[2, :, :, :, :]
    gamma=gamma/2
    b = cp.array(4 * gamma - f0)
    c = cp.array(4 * gamma * gamma - 4 * gamma * f0)
    d = cp.array(-4 * gamma * gamma * f0 - gamma * (m1 ** 2 + m2 ** 2 + m3 ** 2))
    a[3, :, :, :, :] = cp.asnumpy(cubic(b, c, d))
    a[0, :, :, :, :] = a[3, :, :, :, :] * m1/(a[3, :, :, :, :] + 2 * gamma)
    a[1, :, :, :, :] = a[3, :, :, :, :] * m2/(a[3, :, :, :, :] + 2 * gamma)
    a[2, :, :, :, :] = a[3, :, :, :, :] * m3/(a[3, :, :, :, :] + 2 * gamma)
    return a

def poly_root(b,c,d):
    '''
     the largest real root of cubic equation with coefficient(1, b, c, d)
     Here b,c,d is numpy matrix of same size( P * Q * R * T), which means we solve P*Q*R*T equations
     at the same time
     Input: b,c,d (numpy array of size[P,Q,R,T])
     Output: x (numpy array of size[P,Q,R,T])
    '''
    u = (9 * b * c - 27 * d - 2 * b ** 3)/54
    u = u.astype('complex')
    delta = 3 * (4 * c ** 3 - b ** 2 * c ** 2 - 18 * b * c * d + 27 * d ** 2 + 4 * b ** 3 * d)
    v = np.zeros_like(delta, dtype = 'complex')
    v[delta > 0] = delta[delta > 0] ** 0.5/18
    v[delta < 0] = 1j * ((-delta[delta < 0]) ** 0.5)/18
    m = np.zeros_like(u, dtype = 'complex')
    xx = abs(u + v) - abs(u - v)
    m[xx >= 0] = (u + v)[xx >= 0] ** (1/3)
    m[xx  < 0] = (u - v)[xx  < 0] ** (1/3)
    n = np.zeros_like(u, dtype = 'complex')
    n[m != 0] = (b[m != 0] ** 2 - 3 * c[m != 0])/(9 * m[m != 0])
    w = -0.5 + 0.5j * np.sqrt(3)
    root = np.zeros([3, P, Q, R, T], dtype = 'complex')
    root[0] = m + n - b/3
    root[1] = w * m + w * w * n - b/3
    root[2] = w * w * m + w * n - b/3
    realMask = abs(root.imag) < 1e-8
    root[~realMask] = 0
    return np.amax(root.real, axis = 0)

def ProxFS(y, sigma):
    '''
    Compute dual proximal of function J，use Moreau's identity
    Prox_{sigma F^*}(w) = w - sigma * Prox_{F/sigma}(w/sigma)
    Input: centered grid object y, constant sigma
    Output: centered grid object
    '''
    return y - sigma * ProxJ(y/sigma, 1/sigma)

def div(u):
    '''
    Compute divergence operator on staggered grid
    Input: staggered grid object u
    Output: v (P*Q*R*T), scalar field on centered grid
    '''
    v = P * (u.M1[1:, :, :, :] - u.M1[:-1, :, :, :]) + Q * (u.M2[:, 1:, :, :] - u.M2[:, :-1, :, :]) + \
        R * (u.M3[:, :, 1:, :] - u.M3[:, :, :-1, :]) + T * (u.F [:, :, :, 1:] - u.F [:, :, :, :-1])
    return v

def poisson_Neumann(f):
    '''
    Solving poisson equation: Au = f on centered gird
    Input:  f(size P*Q*R*T), scalar field on centered grid
    Output: u(size P*Q*R*T), scalar field on centered grid
    '''
    x =(2 * np.cos(np.pi * np.arange(0, P) / P) - 2) * P ** 2
    y =(2 * np.cos(np.pi * np.arange(0, Q) / Q) - 2) * Q ** 2
    z =(2 * np.cos(np.pi * np.arange(0, R) / R) - 2) * R ** 2
    t =(2 * np.cos(np.pi * np.arange(0, T) / T) - 2) * T ** 2
    denom2 = x.reshape(P, 1, 1, 1) + y.reshape(1, Q, 1, 1) + z.reshape(1, 1, R, 1) + t.reshape(1, 1, 1, T)
    denom2[abs(denom2) < 1e-8] = 1
    fhat = dctn(f, norm = 'ortho')
    uhat = -(fhat)/denom2
    res = idctn(uhat, norm = 'ortho')
    return res

def ProxG(u):
    '''
    Perform projection onto set {U|AU=f}
    Input: staggered grid object u
    Output: staggered grid object v
    '''
    p = poisson_Neumann(-div(u))
    v = staggered()
    v.M1 = u.M1.copy()
    v.M2 = u.M2.copy()
    v.M3 = u.M3.copy()
    v.F  = u.F.copy()
    v.M1[1:-1, :, :, :] = v.M1[1:-1, :, :, :] - (p[1:, :, :, :] - p[:-1, :, :, :]) * P
    v.M2[:, 1:-1, :, :] = v.M2[:, 1:-1, :, :] - (p[:, 1:, :, :] - p[:, :-1, :, :]) * Q
    v.M3[:, :, 1:-1, :] = v.M3[:, :, 1:-1, :] - (p[:, :, 1:, :] - p[:, :, :-1, :]) * R
    v.F [:, :, :, 1:-1] = v.F [:, :, :, 1:-1] - (p[:, :, :, 1:] - p[:, :, :, :-1]) * T
    return v

def perform_primal_dual(x,niter,theta=1):
    '''
    perform primal dual algorithm
    Input: staggered grid object x, total iteration times niter, a parameter theta
    Output: staggered grid object x, its interpolation y
    '''
    sigma = 100
    tau = 0.99/sigma
    x1 = staggered()
    x1.M1 = x.M1.copy()
    x1.M2 = x.M2.copy()
    x1.F = x.F.copy()
    y = interp(x)
    for i in range(niter):
        xold = staggered()
        xold.M1 = x.M1.copy()
        xold.M2 = x.M2.copy()
        xold.M3 = x.M3.copy()
        xold.F  = x.F.copy()
        y0 = interp(x1)
        y = ProxFS(y + sigma * y0, sigma)
        z = interp_ad(y)
        z.M1 = x.M1 - tau * z.M1
        z.M2 = x.M2 - tau * z.M2
        z.M3 = x.M3 - tau * z.M3
        z.F  = x.F  - tau * z.F
        x = ProxG(z)
        x1.M1 = x.M1 + theta * (x.M1 - xold.M1)
        x1.M2 = x.M2 + theta * (x.M2 - xold.M2)
        x1.M3 = x.M3 + theta * (x.M3 - xold.M3)
        x1.F  = x.F  + theta * (x.F  - xold.F)
        aa = interp(x)
        E = np.sum((aa[0, :, :, :, :] ** 2 + aa[1, :, :, :, :] ** 2 + aa[2, :, :, :, :]**2)\
                   /np.maximum(aa[3, :, :, :, :] ** 2, 1e-8)).item()
        print(f'iteration {i:3d}, energy {E:4.8f} ')
    return (x, y)

if __name__ == "__main__":
    '''
    data prepartion,i1,i2
    '''
    with mrcfile.open('/home/test/fanxiao/data/emd_21546.map', permissive=True, mode='r') as mrc:
        i1 = np.array(mrc.data, dtype = np.float64)
    i1[i1 < 0.05] = 0
    i2 = np.zeros_like(i1, dtype = np.float64)
    i2[:, 40:, :] = i1[:, :-40, :]
    '''
    Normalize i1,i2
    '''
    i1 = i1/i1.sum()
    i2 = i2/i2.sum()
    U0 = staggered()
    '''
    initial density F, we use linear interpolation 
    '''
    F_init = np.zeros([P, Q, R, T + 1])
    for t in range(T + 1):
        F_init[:, :, :, t] = i2 * t / T + i1 * (T - t) / T
    U0.F = F_init
    '''
    perform primal-dual algorithm
    '''
    U, V = perform_primal_dual(U0, 1)
    '''
    postprocessing data
    '''
    F = np.zeros([T+1, P, Q, R])
    for t in range(T + 1):
        F[t, :, :, :] = U.F[:, :, :, t]
        max = np.max(F[t, :, :, :])
        min = np.min(F[t, :, :, :])
        F[t, :, :, :] = min + (F[t, :, :, :] - min) / (max - min)
    np.save('/home/test/fanxiao/fgg', F)






