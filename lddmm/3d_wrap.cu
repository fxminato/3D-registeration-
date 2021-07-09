#ifndef WRAP_CU
#define WRAP_CU
extern "C" __global__ void wrap(double* y,const double* x,
                                     const double* f1,
                                     const double* f2,
                                     const double* f3,
                                     const long n)
{
    long tid = blockDim.x * blockIdx.x + threadIdx.x;
    if(tid< n * n * n )
    {   y[tid] = 0;
        double f11;
        double f22;
        double f33;
        f11 = f1[tid];
        f22 = f2[tid];
        f33 = f3[tid];
        f11 = (f11 > 0) ? f11 : 0;
        f11 = (f11 < n - 1) ? f11 : (n - 1);
        f22 = (f22 > 0) ? f22 : 0;
        f22 = (f22 < n - 1) ? f22 : (n - 1);
        f33 = (f33 > 0) ? f33 : 0;
        f33 = (f33 < n - 1) ? f33 : (n - 1);
        long fx1 = floor(f11);
        long fy1 = floor(f22);
        long fz1 = floor(f33);
        long fx2 = ceil(f11);
        long fy2 = ceil(f22);
        long fz2 = ceil(f33);
        double dx1 = f11 - fx1;
        double dx2 = 1 - dx1;
        double dy1 = f22 - fy1;
        double dy2 = 1 - dy1;
        double dz1 = f33 - fz1;
        double dz2 = 1 - dz1;
        y[tid] += (x[fx1 * n * n + fy1 * n + fz1] * dx2 * dy2 * dz2);
        y[tid] += (x[fx1 * n * n + fy1 * n + fz2] * dx2 * dy2 * dz1);
        y[tid] += (x[fx1 * n * n + fy2 * n + fz1] * dx2 * dy1 * dz2);
        y[tid] += (x[fx1 * n * n + fy2 * n + fz2] * dx2 * dy1 * dz1);
        y[tid] += (x[fx2 * n * n + fy1 * n + fz1] * dx1 * dy2 * dz2);
        y[tid] += (x[fx2 * n * n + fy1 * n + fz2] * dx1 * dy2 * dz1);
        y[tid] += (x[fx2 * n * n + fy2 * n + fz1] * dx1 * dy1 * dz2);
        y[tid] += (x[fx2 * n * n + fy2 * n + fz2] * dx1 * dy1 * dz1);


    }
}

#endif