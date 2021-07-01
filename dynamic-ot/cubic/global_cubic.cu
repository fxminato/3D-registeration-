#ifndef GLOBAL_CUBIC_CU
#define GLOBAL_CUBIC_CU

#include <thrust/complex.h>
using namespace thrust;

extern "C" __global__ void cubic(const double* B,
                                 const double* C,
                                 const double* D,
                                 long n,
                                 double* x)
{
    long i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < n) {
        double b = B[i], c = C[i], d = D[i];
        complex<double> delta, u, v, m, n, w, r0, r1, r2;
        u = (9 * b * c - 27 * d - 2 * b * b * b) / 54;
        delta = 3 * (4 * pow(c, 3) - pow(b, 2) * pow(c, 2)  - 18 * b * c * d + 27 * pow(d, 2) + 4 * pow(b, 3) * d);
        v = sqrt(delta) / 18;
        m = abs(u + v) >= abs(u - v) ? pow(u + v, 1. / 3) : pow(u - v, 1. / 3);
        n = abs(m) > 1e-8 ? (pow(b, 2) - 3 * c) / (m * 9) : 0;
        w.real(-0.5); w.imag(0.5 * sqrt(3.0));
        r0 = m + n - b / 3;
        r1 = w * m + w * w * n - b / 3;
        r2 = w * w * m + w * n - b / 3;
        if (abs(r0.imag()) > 1e-8) r0.real(0);
        if (abs(r1.imag()) > 1e-8) r1.real(0);
        if (abs(r2.imag()) > 1e-8) r2.real(0);
        x[i] = fmax(fmax(r0.real(), r1.real()), r2.real());
    }
}
#endif