import os
import cupy as cp

BLOCKSIZE = 1024
module = cp.RawModule(path = os.path.dirname(__file__) + '/kernels.cubin', backend = 'nvcc')
ker_cubic = module.get_function('cubic')

def cubic(b, c, d):
    x = cp.empty_like(b)

    ker_cubic(((x.size- 1) // BLOCKSIZE + 1, ), \
                  (BLOCKSIZE, ), \
                  (b, c, d, x.size, x))

    return x
