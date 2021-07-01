import numpy as np
from math import pi
from numpy.fft import fft2, ifft2,fftn,ifftn
from scipy.ndimage import convolve
from skimage.transform import warp
import matplotlib.pyplot as plt
from scipy.fft import dctn,idctn
from skimage.io import imread, imsave
import imageio
from PIL import Image
sigma1=20
sigma2=100
i0=np.zeros([64,64])
i1=np.zeros([64,64])
for i in range(64):
    for j in range(64):
        i0[i,j]=np.exp(-(i-31.5)**2/(2*sigma1)-(j-31.5)**2/(2*sigma2))/(2*np.pi*sigma1*sigma2)
        i1[i,j]=np.exp(-(j - 31.5) ** 2 / (2 * sigma1) - (i - 31.5) ** 2 / (2 * sigma2)) / (
                    2 * np.pi * sigma1 * sigma2)
max0=np.max(i0)
min0=np.min(i0)
max1=np.max(i1)
min1=np.min(i1)
i0=min0+(i0-min0)/(max0-min0)
i1=min1+(i1-min1)/(max1-min1)
i0=(i0*255).astype('uint8')
i1=(i1*255).astype('uint8')
imsave('/Users/minato/desktop/i0.png',i0)
imsave('/Users/minato/desktop/i1.png',i1)