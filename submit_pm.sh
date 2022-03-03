#!/bin/bash 
#SBATCH -C gpu 
#SBATCH --ntasks-per-node 4
#SBATCH --cpus-per-task 32
#SBATCH --gpus-per-node 4
#SBATCH --time=0:10:00
#SBATCH --image=romerojosh/containers:sc21_tutorial
#SBATCH -J pm-crop64
#SBATCH -o %x-%j.out

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
