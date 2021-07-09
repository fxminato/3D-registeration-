import os
import cupy as cp

BLOCKSIZE = 1024
kernel_path = os.path.dirname(__file__)
module = cp.RawModule(code = open(f'{kernel_path}/my_laplacian.cu').read(), \
                      backend = 'nvrtc', \
                      options = ('--std=c++11', f'-I {kernel_path}'), \
                      name_expressions = ['laplacian'])
ker_laplacian = module.get_function('laplacian')

def laplacian(x):
    n =  x.shape[0]
    y = cp.empty_like(x)
    ker_laplacian(((n * n * n - 1) // BLOCKSIZE + 1, ), \
                  (BLOCKSIZE, ), \
                  (y, x, n))
    return y
