import numpy as np
import scipy as sp

from math import pi, sqrt
from scipy.fft import fft

__all__ = ['dctn', 'idctn']

def dctn(x):
    d = x.ndim
    y = np.empty_like(x, dtype = np.float64)
    for i in range(d):
        n = x.shape[i]
        assert n % 2 == 0

        s_full = slice(None)
        s_first_half = slice(None, n // 2)
        s_second_half = slice(n // 2, None)
        s_even = slice(None, None, 2)
        s_odd = slice(None, None, -2)
        y[tuple(s_first_half  if j == i else s_full for j in range(d))] = x[tuple(s_even if j == i else s_full for j in range(d))]
        y[tuple(s_second_half if j == i else s_full for j in range(d))] = x[tuple(s_odd  if j == i else s_full for j in range(d))]

        w = sqrt(2 / n) * np.exp((-1j * pi / (2 * n)) * np.arange(n))
        w[0] /= sqrt(2)
        w = w.reshape(tuple(n if j == i else 1 for j in range(d)))

        x = np.real(w * fft(y, axis = i))
    return x

def idctn(x):
    d = x.ndim
    x = x.copy()
    for i in range(d):
        n = x.shape[i]
        assert n % 2 == 0

        w = sqrt(2 / n) * np.exp((-1j * pi / (2 * n)) * np.arange(n))
        w[0] /= sqrt(2)
        w = w.reshape(tuple(n if j == i else 1 for j in range(d)))

        y = np.real(fft(w * x, axis = i))

        s_full = slice(None)
        s_first_half = slice(None, n // 2)
        s_second_half = slice(n // 2, None)
        s_even = slice(None, None, 2)
        s_odd = slice(None, None, -2)
        x[tuple(s_even if j == i else s_full for j in range(d))] = y[tuple(s_first_half  if j == i else s_full for j in range(d))]
        x[tuple(s_odd  if j == i else s_full for j in range(d))] = y[tuple(s_second_half if j == i else s_full for j in range(d))]
    return x

if __name__ == '__main__':
    for t in range(10):
        a = np.random.randn(4, 4)
        if not np.allclose(dctn(a), sp.fft.dctn(a, norm = 'ortho')):
            print('Wrong!')
            break
        if not np.allclose(idctn(a), sp.fft.idctn(a, norm = 'ortho')):
            print('Wrong!')
            break
    else:
        print('Right!')
