#ifndef 3D_WRAP_CU
#define 3D_WRAP_CU
#include "device_index.cu"
extern "C" __global__ void wrap(double* y,const double* x,
                                    const long n_col,
                                     const long n_row,
                                     const long n_slc,
                                     const double* f1,
                                     const double* f2,
                                     const double* f3)
{
    long tid=blockDim.x*blockIdx.x+threadIdx.x;
    if(tid<n_col*n_row*n_slc)
    {   y[tid]=0;
        long i,j,k;
        dev_volume_rvs_index(&i,&j,&k,tid,n_col,n_row)
        if((f1[tid]>=0) && (f1[tid]<=n_col-1)&&(f2[tid]>=0)&&(f2[tid]<=n_row-1)&&(f3[tid]>=0)&&(f3[tid]<=n_slc-1))
        {long fx1=floor(f1[tid]);
         long fx2=ceil(f1[tid]);
         long fy1=floor(f2[tid]);
         long fy2=ceil(f2[tid]);
         long fz1=floor(f3[tid]);
         long fz2=ceil(f3[tid]);
         double dx1=f1[tid]-fx1;
         double dx2=fx2-f1[tid];
         double dy1=f2[tid]-fy1;
         double dy2=fy2-f2[tid];
         double dz1=f3[tid]-fz1;
         double dz2=fz2-f3[tid];
         y[tid]+=(x[dev_volume_index(fx1,fy1,fz1,n_col,n_row)]*dx2*dy2*dz2);
         y[tid]+=(x[dev_volume_index(fx1,fy1,fz2,n_col,n_row)]*dx2*dy2*dz1);
         y[tid]+=(x[dev_volume_index(fx1,fy2,fz1,n_col,n_row)]*dx2*dy1*dz2);
         y[tid]+=(x[dev_volume_index(fx1,fy2,fz2,n_col,n_row)]*dx2*dy1*dz1);
         y[tid]+=(x[dev_volume_index(fx2,fy1,fz1,n_col,n_row)]*dx1*dy2*dz2);
         y[tid]+=(x[dev_volume_index(fx2,fy1,fz2,n_col,n_row)]*dx1*dy2*dz1);
         y[tid]+=(x[dev_volume_index(fx2,fy2,fz1,n_col,n_row)]*dx1*dy1*dz2);
         y[tid]+=(x[dev_volume_index(fx2,fy2,fz2,n_col,n_row)]*dx1*dy1*dz1);
        }
        
    }
}
