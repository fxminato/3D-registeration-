import numpy as np
import cupy as cp
import time
s = time.time()
a = np.ones([20,240,240,240])
t = time.time()
print(f'np-create {t-s:4.8f}')
s = time.time()
b = cp.ones([20,240,240,240])
t = time.time()
print(f'cp-create {t-s:4.8f}')
s = time.time()
aa = cp.array(a)
t = time.time()
print(f'np-to-cp {t-s:4.8f}')
s = time.time()
bb = cp.asnumpy(b)
t = time.time()
print(f'cp-to np {t-s:4.8f}')
s = time.time()
c = a + bb
t = time.time()
print(f'np plus np {t-s:4.8f}')
s = time.time()
cc = b + aa
t = time.time()
print(f'cp plus cp {t-s:4.8f}')
s = time.time()
d = a * 2
t = time.time()
print(f'np times {t-s:4.8f}')
s = time.time()
e = b * 2
t = time.time()
print(f'cp times {t-s:4.8f}')
