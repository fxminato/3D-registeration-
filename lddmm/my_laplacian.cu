#ifndef MY_LAPLACIAN_CU
#define MY_LAPLACIAN_CU

extern "C" __global__ void laplacian(double* y,
                                     const double* x,
                                     const long n)
{
    long tid = blockDim.x * blockIdx.x + threadIdx.x;

    if (tid < n * n * n)
    {
        long i, j, k;
        i = tid / (n * n);
        j = (tid - i * n * n) / n;
        k = (tid - i * n * n) % n;
        y[tid] += x[tid] - x[((i + 1) % n) * n * n + j * n + k];
        y[tid] += x[tid] - x[((i + n - 1) % n) * n * n + j * n + k];
        y[tid] += x[tid] - x[i * n * n + ((j + 1) % n) * n + k];
        y[tid] += x[tid] - x[i * n * n + ((j + n - 1) % n) * n + k];
        y[tid] += x[tid] - x[i * n * n + j * n + ((k + 1) % n)];
        y[tid] += x[tid] - x[i * n * n + j * n + ((k + n - 1) % n)];
    }
}

#endif
