import numpy as np
import cupy as cp
import mrcfile
from math import pi
from numpy.fft import fft2, ifft2,fftn,ifftn
from scipy.ndimage import convolve
from skimage.transform import warp
import matplotlib.pyplot as plt
from scipy.fft import dctn,idctn
import imageio
from PIL import Image
P = 240
Q = 240
R = 240
T = 8
d=[P,Q,T]
epsilon = 1e-8
class staggered():
    """
    Vector field on staggered grid
    (M1,M2,M3) is the 3D coordinate of momentum M
    F is the density function(space&time), with boundary in time domain being fixed
    """
    def __init__(self):
        self.M1 = cp.zeros([P+1, Q, R, T])
        self.M2 = cp.zeros([P, Q+1, R, T])
        self.M3 = cp.zeros([P, Q, R+1, T])
        self.F  = cp.zeros([P, Q, R, T+1])
        self.F[:, :, :,  0] = i0
        self.F[:, :, :, -1] = i1

def interp(U):
    """
        Interpolation operator from staggered grid to centered grid
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

def ProxJ(V0, gamma, epsilon):
    a  = cp.zeros([4, P, Q, R, T])
    f0 = V0[3, :, :, :, :]
    m1 = V0[0, :, :, :, :]
    m2 = V0[1, :, :, :, :]
    m3 = V0[2, :, :, :, :]
    gamma=gamma/2
    a[3, :, :, :, :] = poly_root(4 * gamma - f0, 4 * gamma * gamma - 4 * gamma * f0,\
                                    -4 * gamma * gamma * f0 - gamma * (m1 ** 2 + m2 ** 2 + m3 ** 2))
    a[0, :, :, :, :] = a[3, :, :, :, :] * m1/(a[3, :, :, :, :] + 2 * gamma)
    a[1, :, :, :, :] = a[3, :, :, :, :] * m2/(a[3, :, :, :, :] + 2 * gamma)
    a[2, :, :, :, :] = a[3, :, :, :, :] * m3/(a[3, :, :, :, :] + 2 * gamma)
    return a

def poly_root(b,c,d):
    '''
     the largest real root of cubic equation with coefficient(1, b, c, d)
    '''
    u = (9 * b * c - 27 * d - 2 * b ** 3)/54
    u = u.astype('complex')
    delta = 3 * (4 * c ** 3 - b ** 2 * c ** 2 - 18 * b * c * d + 27 * d ** 2 + 4 * b ** 3 * d)
    v = cp.zeros_like(delta, dtype = 'complex')
    v[delta > 0] = delta[delta > 0] ** 0.5/18
    v[delta < 0] = 1j * ((-delta[delta < 0]) ** 0.5)/18
    m = cp.zeros_like(u, dtype = 'complex')
    xx = abs(u + v) - abs(u - v)
    m[xx >= 0] = (u + v)[xx >= 0] ** (1/3)
    m[xx  < 0] = (u - v)[xx  < 0] ** (1/3)
    n = cp.zeros_like(u, dtype = 'complex')
    n[m != 0] = (b[m != 0] ** 2 - 3 * c[m != 0])/(9 * m[m != 0])
    w = -0.5 + 0.5j * np.sqrt(3)
    root = cp.zeros([3, P, Q, R, T], dtype = 'complex')
    root[0] = m + n - b/3
    root[1] = w * m + w * w * n - b/3
    root[2] = w * w * m + w * n - b/3
    realMask = abs(root.imag) < 1e-8
    root[~realMask] = 0
    return cp.amax(root.real, axis = 0)

def ProxFS(y, sigma):
    return y - sigma * ProxJ(y/sigma, 1/sigma, epsilon)

def div(u):
    v = P * (u.M1[1:, :, :, :] - u.M1[:-1, :, :, :]) + Q * (u.M2[:, 1:, :, :] - u.M2[:, :-1, :, :]) + \
        R * (u.M3[:, :, 1:, :] - u.M3[:, :, :-1, :]) + T * (u.F [:, :, :, 1:] - u.F [:, :, :, :-1])
    return v

def poisson_Neumann(f):
    x =(2 * cp.cos(np.pi * cp.arange(0, P) / P) - 2) * P ** 2
    y =(2 * cp.cos(np.pi * cp.arange(0, Q) / Q) - 2) * Q ** 2
    z =(2 * cp.cos(np.pi * cp.arange(0, R) / R) - 2) * R ** 2
    t =(2 * cp.cos(np.pi * cp.arange(0, T) / T) - 2) * T ** 2
    denom2 = x.reshape(P, 1, 1, 1) + y.reshape(1, Q, 1, 1) + z.reshape(1, 1, R, 1) + t.reshape(1, 1, 1, T)
    denom2[abs(denom2) < 1e-8] = 1
    fhat=dct4(f)
    uhat=-(fhat)/denom2
    res=idct4(uhat)
    return res

def dct4(xx):
    x = xx.copy()
    xind = cp.empty_like(x,np.float64)
    xind[ :int(P/2), :, :, :] = x[:: 2, :, :, :]
    xind[int(P/2):P, :, :, :] = x[::-2, :, :, :]
    w = 2 * cp.exp(((-1j * np.pi) / (2 * P)) * cp.arange(P)) / cp.sqrt(2 * P)
    w[0] = w[0] / cp.sqrt(2)
    w = w.reshape(P, 1, 1, 1)
    x = cp.real(w * cp.fft.fft(xind, axis = 0))
    xind[:,  :int(Q/2), :, :] = x[:, :: 2, :, :]
    xind[:, int(Q/2):Q, :, :] = x[:, ::-2, :, :]
    w = 2 * cp.exp(((-1j * np.pi) / (2 * Q)) * cp.arange(Q)) / cp.sqrt(2 * Q)
    w[0] = w[0] / cp.sqrt(2)
    w = w.reshape(1, Q, 1, 1)
    x = cp.real(w * cp.fft.fft(xind, axis = 1))
    xind[:, :,  :int(R/2), :] = x[:, :, :: 2, :]
    xind[:, :, int(R/2):R, :] = x[:, :, ::-2, :]
    w = 2 * cp.exp(((-1j * np.pi) / (2 * R)) * cp.arange(R)) / cp.sqrt(2 * R)
    w[0] = w[0] / cp.sqrt(2)
    w = w.reshape(1, 1, R, 1)
    x = cp.real(w * cp.fft.fft(xind, axis = 2))
    xind[:, :, :,  :int(T/2)] = x[:, :, :, :: 2]
    xind[:, :, :, int(T/2):T] = x[:, :, :, ::-2]
    w = 2 * cp.exp(((-1j * np.pi) / (2 * T)) * cp.arange(T)) / cp.sqrt(2 * T)
    w[0] = w[0] / cp.sqrt(2)
    w = w.reshape(1, 1, 1, T)
    x = cp.real(w * cp.fft.fft(xind, axis = 3))
    return x

def idct4(xx):
    x = xx.copy()
    w = 2 * cp.exp(((-1j * np.pi) / (2 * P)) * cp.arange(P)) / cp.sqrt(2 * P)
    w[0] = w[0] / cp.sqrt(2)
    w = w.reshape(P, 1, 1, 1)
    x = cp.real(cp.fft.fft(w * x, axis = 0))
    xind = x.copy()
    x[:: 2, :, :, :] = xind[ :int(P/2), :, :, :]
    x[::-2, :, :, :] = xind[int(P/2):P, :, :, :]
    w = 2 * cp.exp(((-1j * np.pi) / (2 * Q)) * cp.arange(Q)) / cp.sqrt(2 * Q)
    w[0] = w[0] / cp.sqrt(2)
    w = w.reshape(1, Q, 1, 1)
    x = cp.real(cp.fft.fft(w * x, axis = 1))
    xind = x.copy()
    x[:, :: 2, :, :] = xind[:,  :int(Q/2), :, :]
    x[:, ::-2, :, :] = xind[:, int(Q/2):Q, :, :]
    w = 2 * cp.exp(((-1j * np.pi) / (2 * R)) * cp.arange(R)) / cp.sqrt(2 * R)
    w[0] = w[0] / np.sqrt(2)
    w = w.reshape(1, 1, R, 1)
    x = cp.real(cp.fft.fft(w * x, axis = 2))
    xind = x.copy()
    x[:, :, :: 2, :] = xind[:, :,  :int(R/2), :]
    x[:, :, ::-2, :] = xind[:, :, int(R/2):R, :]
    w = 2 * cp.exp(((-1j * np.pi) / (2 * T)) * cp.arange(T)) / cp.sqrt(2 * T)
    w[0] = w[0] / cp.sqrt(2)
    w = w.reshape(1, 1, 1, T)
    x = cp.real(cp.fft.fft(w * x, axis = 3))
    xind = x.copy()
    x[:, :, :, :: 2] = xind[:, :, :,  :int(T/2)]
    x[:, :, :, ::-2] = xind[:, :, :, int(T/2):T]
    return x

def ProxG(u):
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
        E = cp.sum((aa[0, :, :, :, :] ** 2 + aa[1, :, :, :, :] ** 2 + aa[2, :, :, :, :]**2)\
                   /cp.maximum(aa[3, :, :, :, :] ** 2, 1e-8))
        print(f'iteration {i:3d}, energy {E:4.8f} ')
    return (x, y)

if __name__ == "__main__":
    with mrcfile.open('/home/test/fanxiao/data/emd_21546.map', permissive=True, mode='r') as mrc:
        i1 = cp.array(mrc.data, dtype = np.float64)
    i1[i1 < 0.05] = 0
    i2 = cp.zeros_like(i1, dtype = np.float64)
    i2[:, 40:, :] = i1[:, :-40, :]
    i1 = i1/i1.sum()
    i2 = i2/i2.sum()
    U0 = staggered()
    F_init = np.zeros([P, Q, R, T + 1])
    for t in range(T + 1):
        F_init[:, :, :, t] = i2 * t / T + i1 * (T - t) / T
    U0.F = F_init
    U, V = perform_primal_dual(U0, 10)
    F = cp.zeros([T+1, P, Q, R])
    for t in range(T + 1):
        F[t, :, :, :] = U.F[:, :, :, t]
        max = cp.max(F[t, :, :, :])
        min = cp.min(F[t, :, :, :])
        F[t, :, :, :] = min + (F[t, :, :, :] - min) / (max - min)
    J = cp.asnumpy(F)
    np.save('/home/test/fanxiao/fgg', J)






