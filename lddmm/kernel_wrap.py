import os
import cupy as cp
BLOCKSIZE = 1024
kernel_path = os.path.dirname(__file__)
module = cp.RawModule(code = open(f'{kernel_path}/3d_wrap.cu').read(), \
                      backend = 'nvrtc', \
                      options = ('--std=c++11', f'-I {kernel_path}'), \
                      name_expressions = ['wrap'])
ker_wrap = module.get_function('wrap')
def wrap(x, f1, f2, f3):
    y = cp.empty_like(x, dtype=cp.float64)
    n = x.shape[0]
    ker_wrap(((n * n * n - 1) // BLOCKSIZE + 1, ),
             (BLOCKSIZE, ),
             (y, x, f1, f2, f3, n))
    return y
