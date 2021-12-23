#!/bin/bash 
#SBATCH -C gpu 
#SBATCH -A ntrain3_g
#SBATCH --ntasks-per-node 4
#SBATCH --cpus-per-task 32
#SBATCH --gpus-per-task 1
#SBATCH --time=0:10:00
#SBATCH --image=romerojosh/containers:sc21_tutorial
#SBATCH -J pm-crop64
#SBATCH -o %x-%j.out
#-SBATCH --reservation perlmutter_day3

DATADIR=/pscratch/sd/j/joshr/nbody2hydro/datacopies
LOGDIR=${SCRATCH}/ml-pm-training-2022/logs
mkdir -p ${LOGDIR}
args="${@}"

hostname

export NCCL_NET_GDR_LEVEL=PHB

set -x
srun -u shifter -V ${DATADIR}:/data -V ${LOGDIR}:/logs \
    bash -c "
    source export_DDP_vars.sh
    python train.py ${args}
    "
