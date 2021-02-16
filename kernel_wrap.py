import os
import cupy as cp

BLOCKSIZE = 1024
kernel_path = os.path.dirname(__file__)
module = cp.RawModule(code=open(f'{kernel_path}/cuda/3d_wrap.cu').read(), backend='nvrtc',
                      options=('--std=c++11', f'-I {kernel_path}/cuda'),
                      name_expressions=['wrap'])
ker_wrap = module.get_function('wrap')
def wrap(x, f1, f2, f3):
    y=Volume(size=(x.n_slc(),x.n_row(),x.n_col()))
    ker_wrap(((x.n_col() * x.n_row() * x.n_slc() - 1) // BLOCKSIZE + 1,),(BLOCKSIZE, ),
                (y.data_RL, x.data_RL, x.n_col(), x.n_row(), x.n_slc(), f1.data_RL,f2.data_RL,f3.data_RL))
    return y




