import numpy as np
from scipy.fft import fft,dctn,idctn
def dct4(xx):
    x = xx.copy()
    P = x.shape[0]
    Q = x.shape[1]
    R = x.shape[2]
    T = x.shape[3]
    xind = np.empty_like(x,np.float64)
    xind[ :int(P/2), :, :, :] = x[:: 2, :, :, :]
    xind[int(P/2):P, :, :, :] = x[::-2, :, :, :]
    w = 2 * np.exp(((-1j * np.pi) / (2 * P)) * np.arange(P)) / np.sqrt(2 * P)
    w[0] = w[0] / np.sqrt(2)
    w = w.reshape(P, 1, 1, 1)
    x = np.real(w * fft(xind, axis = 0))
    xind[:,  :int(Q/2), :, :] = x[:, :: 2, :, :]
    xind[:, int(Q/2):Q, :, :] = x[:, ::-2, :, :]
    w = 2 * np.exp(((-1j * np.pi) / (2 * Q)) * np.arange(Q)) / np.sqrt(2 * Q)
    w[0] = w[0] / np.sqrt(2)
    w = w.reshape(1, Q, 1, 1)
    x = np.real(w * fft(xind, axis = 1))
    xind[:, :,  :int(R/2), :] = x[:, :, :: 2, :]
    xind[:, :, int(R/2):R, :] = x[:, :, ::-2, :]
    w = 2 * np.exp(((-1j * np.pi) / (2 * R)) * np.arange(R)) / np.sqrt(2 * R)
    w[0] = w[0] / np.sqrt(2)
    w = w.reshape(1, 1, R, 1)
    x = np.real(w * fft(xind, axis = 2))
    xind[:, :, :,  :int(T/2)] = x[:, :, :, :: 2]
    xind[:, :, :, int(T/2):T] = x[:, :, :, ::-2]
    w = 2 * np.exp(((-1j * np.pi) / (2 * T)) * np.arange(T)) / np.sqrt(2 * T)
    w[0] = w[0] / np.sqrt(2)
    w = w.reshape(1, 1, 1, T)
    x = np.real(w * fft(xind, axis = 3))
    return x
def idct4(xx):
    x = xx.copy()
    P = x.shape[0]
    Q = x.shape[1]
    R = x.shape[2]
    T = x.shape[3]
    w = 2 * np.exp(((-1j * np.pi) / (2 * P)) * np.arange(P)) / np.sqrt(2 * P)
    w[0] = w[0] / np.sqrt(2)
    w = w.reshape(P, 1, 1, 1)
    x = np.real(fft(w*x, axis = 0))
    xind = x.copy()
    x[:: 2, :, :, :] = xind[ :int(P/2), :, :, :]
    x[::-2, :, :, :] = xind[int(P/2):P, :, :, :]
    w = 2 * np.exp(((-1j * np.pi) / (2 * Q)) * np.arange(Q)) / np.sqrt(2 * Q)
    w[0] = w[0] / np.sqrt(2)
    w = w.reshape(1, Q, 1, 1)
    x = np.real(fft(w*x, axis = 1))
    xind = x.copy()
    x[:, :: 2, :, :] = xind[:,  :int(Q/2), :, :]
    x[:, ::-2, :, :] = xind[:, int(Q/2):Q, :, :]
    w = 2 * np.exp(((-1j * np.pi) / (2 * R)) * np.arange(R)) / np.sqrt(2 * R)
    w[0] = w[0] / np.sqrt(2)
    w = w.reshape(1, 1, R, 1)
    x = np.real(fft(w*x, axis = 2))
    xind = x.copy()
    x[:, :, :: 2, :] = xind[:, :,  :int(R/2), :]
    x[:, :, ::-2, :] = xind[:, :, int(R/2):R, :]
    w = 2 * np.exp(((-1j * np.pi) / (2 * T)) * np.arange(T)) / np.sqrt(2 * T)
    w[0] = w[0] / np.sqrt(2)
    w = w.reshape(1, 1, 1, T)
    x = np.real(fft(w*x, axis = 3))
    xind = x.copy()
    x[:, :, :, :: 2] = xind[:, :, :,  :int(T/2)]
    x[:, :, :, ::-2] = xind[:, :, :, int(T/2):T]
    return x
a = np.arange(64).reshape(2,2,4,4)
b1 = dctn(a, norm = 'ortho')
c1 = dct4(a)
print(np.linalg.norm(b1-c1))
b2 = idctn(a, norm= 'ortho')
c2 = idct4(a)
print(np.linalg.norm(b2-c2))