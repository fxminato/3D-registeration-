import os
import cupy as cp
BLOCKSIZE = 1024
kernel_path = os.path.dirname(__file__)
module = cp.RawModule(code = open(f'{kernel_path}/my_gradient.cu').read(), \
                      backend = 'nvrtc', \
                      options = ('--std=c++11', f'-I {kernel_path}'), \
                      name_expressions = ['gradientx','gradienty','gradientz'])
ker_gradientx = module.get_function('gradientx')
ker_gradienty = module.get_function('gradienty')
ker_gradientz = module.get_function('gradientz')
def gradientx(x):
    n =  x.shape[0]
    y = cp.empty_like(x)
    ker_gradientx(((n * n * n - 1) // BLOCKSIZE + 1, ), \
                  (BLOCKSIZE, ), \
                  (y , x, n))
    return y

def gradienty(x):
    n =  x.shape[0]
    y = cp.empty_like(x)
    ker_gradienty(((n * n * n - 1) // BLOCKSIZE + 1, ), \
                  (BLOCKSIZE, ), \
                  (y , x, n))
    return y

def gradientz(x):
    n =  x.shape[0]
    y = cp.empty_like(x)
    ker_gradientz(((n * n * n - 1) // BLOCKSIZE + 1, ), \
                  (BLOCKSIZE, ), \
                  (y , x, n))
    return y