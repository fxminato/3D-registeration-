import numpy as np
import cupy as cp
import time
dx = cp.random.randn(3,240,240,240)
dy = cp.random.randn(3,240,240,240)
dz = cp.random.randn(3,240,240,240)
s = time.time()
a = abs(dx[0] * dy[1] * dz[2]+dx[1] * dy[2]*dz[0]+dx[2]*dy[0]*dz[1]-\
dx[2] * dy[1] * dz[0]-dx[1] * dy[0]*dz[2]-dx[0]*dy[2]*dz[1])
t = time.time()
print(t-s)
ker_determinant = cp.ElementwiseKernel(
        'float64 x0, float64 x1, float64 x2, float64 y0, float64 y1, float64 y2, float64 z0, float64 z1, float64 z2',
        'float64 z',
        'z = x0 * y1 * z2 + x1 * y2 * z0 + x2 * y0 * z1 - x2 * y1 * z0 - x1 * y0 * z2 - x0 * y2 * z1',
        'ker_determinant'
)
s = time.time()
c = abs(ker_determinant(dx[0],dx[1],dx[2],dy[0],dy[1],dy[2],dz[0],dz[1],dz[2]))
t = time.time()
print(t-s)
