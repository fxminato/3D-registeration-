nvcc --gpu-architecture=sm_70 -O2 -ptx global_cubic.cu -o global_cubic.ptx
nvcc --gpu-architecture=sm_70 --device-link --cubin global_cubic.ptx -o kernels.cubin
python3 test_cubic.py
