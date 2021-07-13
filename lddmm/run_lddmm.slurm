#! /bin/bash
#SBATCH -J fxxxxxx
#SBATCH -p TOMO-V100
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --output=%j.out
#SBATCH --error=%j.err
#SBATCH --exclude=gpu05,gpu06
###SBATCH --profile=all

source /Share/THUDATA/Softwares/mpich3.3.sh

srun -o output.txt docker run --gpus all --rm -v /Share/THUDATA/UserHome/thunder_user/:/home 192.103.2.201:5000/picos_mpi4y:cupy82-cuda92_python37_pytorch17_matplotlib33_scipy15_mrcfile12_skimage017 sh -c "cd /home/fanxiao/run_test/run_2/ && python3 lddmm.py"