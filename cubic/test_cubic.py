import numpy as np
import cupy as cp
from kernel_cubic import cubic

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

b = cp.random.randn(64)
c = cp.random.randn(64)
d = cp.random.randn(64)
#a1 = poly_root(b,c,d)
a2 = cubic(b, c, d)
print(cp.allclose(a2, (9 * b * c - 27 * d - 2 * b * b * b) / 54))
