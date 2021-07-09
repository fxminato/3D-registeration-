import numpy as np
import cupy as cp
from kernel_wrap import *
from Volume import *
from skimage.transform import warp
img0 = np.random.randn(100,100,100)
#img0 = np.maximum(img0,0)
phi0 = np.random.randn(3,100,100,100)
#phi0 = np.maximum(phi0,0)
#img0 = np.arange(8).reshape([2,2,2])
#img0 = img0.astype(np.float64)
#phi0 = np.zeros([3,2,2,2],dtype= np.float64)
#phi0[0,:,:,:] = 1
#phi0[1,:,:,:] = 2
result0 = cp.array(warp(img0, phi0, mode= 'edge'))
img = cp.array(img0)
phi = cp.array(phi0)
m0 = img
m1 = phi[0, :, :, :]
m2 = phi[1, :, :, :]
m3 = phi[2, :, :, :]
result1 = wrap(m0, m1, m2, m3)
#print(result0)
#print(result1)
print(cp.linalg.norm(result0 - result1))
imaggg =  img.astype(cp.float64, order = 'C')