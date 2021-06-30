#ifndef GLOBAL_CUBIC_CU
#define GLOBAL_CUBIC_CU

#include <thrust/complex.h>

extern "C" __global__ void cubic(const double* B, const double* C, const double* D, long n, double* x)
{
    long i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < n) {
        double b = B[i], c = C[i], d = D[i];
        thrust::complex<double> u, v, m, n, w, r0, r1, r2, delta;
        u = (9 * b * c - 27 * d - 2 * b * b * b) / 54;
        x[i] = u.real();
    }

 /*
 u = (9 * b * c - 27 * d - 2 * pow(b,3)) / 54;
 delta = 3 * (4 * pow(c, 3) - pow(b, 2) * pow(c, 2)  - 18 * b * c * d + 27 * pow(d, 2) + 4 * pow(b, 3) * d);
 v = sqrt(delta) / 18.0;
 if(abs(u+v) >= abs(u-v)
 {
     m = pow(u+v, 1.0/3.0);
 }
 else
 {
     m = pow(u-v, 1.0/3.0);
 }
 if (abs(m) != 0)
 {
 n = (pow(b, 2) - 3 * c)/(m * 9.0);
 }
 else n = 0;
 w.real(-0.5);
 w.imag(0.5 * sqrt(3));
 root0 = m + n - b/3;
 root1 = w * m + w * w * n - b/3;
 root2 = w * w * m + w * n - b/3;
 if(root0.imag() > 1e-8)
 root0.real(0);
 if(root1.imag() > 1e-8)
 root1.real(0);
 if(root2.imag() > 1e-8)
 root2.real(0);
 A[i] = max(max(root0.real(),root1.real()),root2.real());*/
}

#endif