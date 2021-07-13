#ifndef WRAP_CU
#define WRAP_CU

__device__ __forceinline__ bool is_legal(long i, long j, long k, long n) {
    return 0 <= i && i < n && 0 <= j && j < n && 0 <= k && k < n;
}

__device__ __forceinline__ long logic_to_physical(long i, long j, long k, long n) {
    return (i * n + j) * n + k;
}

extern "C" __global__ void wrap(double* y,
                                const double* x,
                                const double* f1,
                                const double* f2,
                                const double* f3,
                                const long n)
{
    long tid = blockDim.x * blockIdx.x + threadIdx.x;
    if (tid < n * n * n) {
        long i = f1[tid];
        long j = f2[tid];
        long k = f3[tid];
        double di = f1[tid] - i;
        double dj = f2[tid] - j;
        double dk = f3[tid] - k;
        y[tid] = 0;
        if (is_legal(i    , j    , k    , n))
            y[tid] += x[logic_to_physical(i    , j    , k    , n)] * (1 - di) * (1 - dj) * (1 - dk);
        if (is_legal(i    , j    , k + 1, n))
            y[tid] += x[logic_to_physical(i    , j    , k + 1, n)] * (1 - di) * (1 - dj) *      dk ;
        if (is_legal(i    , j + 1, k    , n))
            y[tid] += x[logic_to_physical(i    , j + 1, k    , n)] * (1 - di) *      dj  * (1 - dk);
        if (is_legal(i    , j + 1, k + 1, n))
            y[tid] += x[logic_to_physical(i    , j + 1, k + 1, n)] * (1 - di) *      dj  *      dk ;
        if (is_legal(i + 1, j    , k    , n))
            y[tid] += x[logic_to_physical(i + 1, j    , k    , n)] *      di  * (1 - dj) * (1 - dk);
        if (is_legal(i + 1, j    , k + 1, n))
            y[tid] += x[logic_to_physical(i + 1, j    , k + 1, n)] *      di  * (1 - dj) *      dk ;
        if (is_legal(i + 1, j + 1, k    , n))
            y[tid] += x[logic_to_physical(i + 1, j + 1, k    , n)] *      di  *      dj  * (1 - dk);
        if (is_legal(i + 1, j + 1, k + 1, n))
            y[tid] += x[logic_to_physical(i + 1, j + 1, k + 1, n)] *      di  *      dj  *      dk ;
    }
}

#endif