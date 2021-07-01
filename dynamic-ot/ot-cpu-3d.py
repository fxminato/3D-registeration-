import numpy as np
import mrcfile
from math import pi
from numpy.fft import fft2, ifft2,fftn,ifftn
from scipy.ndimage import convolve
from skimage.transform import warp
import matplotlib.pyplot as plt
from scipy.fft import dctn,idctn
import imageio
from PIL import Image
from skimage.transform import downscale_local_mean
P = 60
Q = 60
R = 60
T = 20
epsilon = 1e-8
class staggered():
    """
    Vector field on staggered grid
    (M1,M2,M3) is the 3D coordinate of momentum M
    F is the density function(space&time), with boundary in time domain being fixed
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

def ProxJ(V0, gamma):

    a  = np.zeros([4, P, Q, R, T])
    f0 = V0[3, :, :, :, :]
    m1 = V0[0, :, :, :, :]
    m2 = V0[1, :, :, :, :]
    m3 = V0[2, :, :, :, :]
    gamma = gamma/2
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
    return y - sigma * ProxJ(y/sigma, 1/sigma)

def div(u):
    v = P * (u.M1[1:, :, :, :] - u.M1[:-1, :, :, :]) + Q * (u.M2[:, 1:, :, :] - u.M2[:, :-1, :, :]) + \
        R * (u.M3[:, :, 1:, :] - u.M3[:, :, :-1, :]) + T * (u.F [:, :, :, 1:] - u.F [:, :, :, :-1])
    return v

def poisson_Neumann(f):
    x =(2 * np.cos(np.pi * np.arange(0, P) / P) - 2) * P ** 2
    y =(2 * np.cos(np.pi * np.arange(0, Q) / Q) - 2) * Q ** 2
    z =(2 * np.cos(np.pi * np.arange(0, R) / R) - 2) * R ** 2
    t =(2 * np.cos(np.pi * np.arange(0, T) / T) - 2) * T ** 2
    denom2 = x.reshape(P, 1, 1, 1) + y.reshape(1, Q, 1, 1) + z.reshape(1, 1, R, 1) + t.reshape(1, 1, 1, T)
    denom2[abs(denom2) < 1e-8] = 1
    fhat = dct4(f)
    uhat = -(fhat)/denom2
    res = idct4(uhat)
    return res

def dct4(xx):
    x = xx.copy()
    xind = np.empty_like(x,np.float64)
    xind[ :int(P/2), :, :, :] = x[:: 2, :, :, :]
    xind[int(P/2):P, :, :, :] = x[::-2, :, :, :]
    w = 2 * np.exp(((-1j * np.pi) / (2 * P)) * np.arange(P)) / np.sqrt(2 * P)
    w[0] = w[0] / np.sqrt(2)
    w = w.reshape(P, 1, 1, 1)
    x = np.real(w * np.fft.fft(xind, axis = 0))
    xind[:,  :int(Q/2), :, :] = x[:, :: 2, :, :]
    xind[:, int(Q/2):Q, :, :] = x[:, ::-2, :, :]
    w = 2 * np.exp(((-1j * np.pi) / (2 * Q)) * np.arange(Q)) / np.sqrt(2 * Q)
    w[0] = w[0] / np.sqrt(2)
    w = w.reshape(1, Q, 1, 1)
    x = np.real(w * np.fft.fft(xind, axis = 1))
    xind[:, :,  :int(R/2), :] = x[:, :, :: 2, :]
    xind[:, :, int(R/2):R, :] = x[:, :, ::-2, :]
    w = 2 * np.exp(((-1j * np.pi) / (2 * R)) * np.arange(R)) / np.sqrt(2 * R)
    w[0] = w[0] / np.sqrt(2)
    w = w.reshape(1, 1, R, 1)
    x = np.real(w * np.fft.fft(xind, axis = 2))
    xind[:, :, :,  :int(T/2)] = x[:, :, :, :: 2]
    xind[:, :, :, int(T/2):T] = x[:, :, :, ::-2]
    w = 2 * np.exp(((-1j * np.pi) / (2 * T)) * np.arange(T)) / np.sqrt(2 * T)
    w[0] = w[0] / np.sqrt(2)
    w = w.reshape(1, 1, 1, T)
    x = np.real(w * np.fft.fft(xind, axis = 3))
    return x

def idct4(xx):
    x = xx.copy()
    w = 2 * np.exp(((-1j * np.pi) / (2 * P)) * np.arange(P)) / np.sqrt(2 * P)
    w[0] = w[0] / np.sqrt(2)
    w = w.reshape(P, 1, 1, 1)
    x = np.real(np.fft.fft(w * x, axis = 0))
    xind = x.copy()
    x[:: 2, :, :, :] = xind[ :int(P/2), :, :, :]
    x[::-2, :, :, :] = xind[int(P/2):P, :, :, :]
    w = 2 * np.exp(((-1j * np.pi) / (2 * Q)) * np.arange(Q)) / np.sqrt(2 * Q)
    w[0] = w[0] / np.sqrt(2)
    w = w.reshape(1, Q, 1, 1)
    x = np.real(np.fft.fft(w * x, axis = 1))
    xind = x.copy()
    x[:, :: 2, :, :] = xind[:,  :int(Q/2), :, :]
    x[:, ::-2, :, :] = xind[:, int(Q/2):Q, :, :]
    w = 2 * np.exp(((-1j * np.pi) / (2 * R)) * np.arange(R)) / np.sqrt(2 * R)
    w[0] = w[0] / np.sqrt(2)
    w = w.reshape(1, 1, R, 1)
    x = np.real(np.fft.fft(w * x, axis = 2))
    xind = x.copy()
    x[:, :, :: 2, :] = xind[:, :,  :int(R/2), :]
    x[:, :, ::-2, :] = xind[:, :, int(R/2):R, :]
    w = 2 * np.exp(((-1j * np.pi) / (2 * T)) * np.arange(T)) / np.sqrt(2 * T)
    w[0] = w[0] / np.sqrt(2)
    w = w.reshape(1, 1, 1, T)
    x = np.real(np.fft.fft(w * x, axis = 3))
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
        E = np.sum((aa[0, :, :, :, :] ** 2 + aa[1, :, :, :, :] ** 2 + aa[2, :, :, :, :]**2)\
                   /np.maximum(aa[3, :, :, :, :] ** 2, 1e-8)).item()
        print(f'iteration {i:3d}, energy {E:4.8f} ')
    return (x, y)

if __name__ == "__main__":
    with mrcfile.open('/Users/minato/Desktop/data/emd_21546.map', permissive=True, mode='r') as mrc:
        i1 = np.array(mrc.data, dtype = np.float64)
    i1[i1 < 0.05] = 0
    i2 = np.zeros_like(i1, dtype = np.float64)
    i2[:, 40:, :] = i1[:, :-40, :]
    i1 = downscale_local_mean(i1, (2, 2, 2))
    i1 = downscale_local_mean(i1, (2, 2, 2))
    i2 = downscale_local_mean(i2, (2, 2, 2))
    i2 = downscale_local_mean(i2, (2, 2, 2))
    i1 = i1/i1.sum()
    i2 = i2/i2.sum()
    #i1 = np.zeros([60, 60, 60])
    #i1[20:40, 20:40, 20:40] = 1
    #a = i1.sum()
    #i1 = i1/a
    #i2 = np.zeros_like(i1, dtype = np.float64)
    #2[:, 10:, :] = i1[:, :-10, :]
    U0 = staggered()
    F_init = np.zeros([P, Q, R, T + 1])
    for t in range(T + 1):
        F_init[:, :, :, t] = i2 * t / T + i1 * (T - t) / T
    U0.F = F_init
    U, V = perform_primal_dual(U0, 2000)
    U.F = U.F.astype(np.float32)
    F = np.zeros([T+1, P, Q, R])
    for t in range(T + 1):
        F[t, :, :, :] = U.F[:, :, :, t]
        max = np.max(F[t, :, :, :])
        min = np.min(F[t, :, :, :])
        F[t, :, :, :] = min + (F[t, :, :, :] - min) / (max - min)
    F = F.astype(np.float32)
    print(np.linalg.norm(i1-U.F[:,:,:,0]))
    print(np.linalg.norm(i2-U.F[:,:,:,-1]))
    mrcfile.new('/Users/minato/Desktop/data/fx0.mrc', U.F[:, :, :, 0], overwrite = True)
    mrcfile.new('/Users/minato/Desktop/data/fx1.mrc', U.F[:, :, :, 1], overwrite = True)
    mrcfile.new('/Users/minato/Desktop/data/fx2.mrc', U.F[:, :, :, 2], overwrite = True)
    mrcfile.new('/Users/minato/Desktop/data/fx3.mrc', U.F[:, :, :, 3], overwrite = True)
    mrcfile.new('/Users/minato/Desktop/data/fx4.mrc', U.F[:, :, :, 4], overwrite = True)
    mrcfile.new('/Users/minato/Desktop/data/fx5.mrc', U.F[:, :, :, 5], overwrite = True)
    mrcfile.new('/Users/minato/Desktop/data/fx6.mrc', U.F[:, :, :, 6], overwrite = True)
    mrcfile.new('/Users/minato/Desktop/data/fx7.mrc', U.F[:, :, :, 7], overwrite = True)
    mrcfile.new('/Users/minato/Desktop/data/fx8.mrc', U.F[:, :, :, 8], overwrite = True)
    mrcfile.new('/Users/minato/Desktop/data/fx9.mrc', U.F[:, :, :, 9], overwrite = True)
    mrcfile.new('/Users/minato/Desktop/data/fx90.mrc', U.F[:, :, :, 10], overwrite=True)
    mrcfile.new('/Users/minato/Desktop/data/fx91.mrc', U.F[:, :, :, 11], overwrite=True)
    mrcfile.new('/Users/minato/Desktop/data/fx92.mrc', U.F[:, :, :, 12], overwrite=True)
    mrcfile.new('/Users/minato/Desktop/data/fx93.mrc', U.F[:, :, :, 13], overwrite=True)
    mrcfile.new('/Users/minato/Desktop/data/fx94.mrc', U.F[:, :, :, 14], overwrite=True)
    mrcfile.new('/Users/minato/Desktop/data/fx95.mrc', U.F[:, :, :, 15], overwrite=True)
    mrcfile.new('/Users/minato/Desktop/data/fx96.mrc', U.F[:, :, :, 16], overwrite=True)
    mrcfile.new('/Users/minato/Desktop/data/fx97.mrc', U.F[:, :, :, 17], overwrite=True)
    mrcfile.new('/Users/minato/Desktop/data/fx98.mrc', U.F[:, :, :, 18], overwrite=True)
    mrcfile.new('/Users/minato/Desktop/data/fx99.mrc', U.F[:, :, :, 19], overwrite=True)
    mrcfile.new('/Users/minato/Desktop/data/fx990.mrc', U.F[:, :, :, 20], overwrite=True)
