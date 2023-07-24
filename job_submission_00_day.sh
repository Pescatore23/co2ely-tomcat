#!/bin/bash -l
#SBATCH --job-name=$1recos_to_h5
#SBATCH --time=6:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=200G
#SBATCH -p day


# Activate conda env
export PYTHONPATH=''
#eval "$(/store/empa/em13/fischer/lib/miniconda3/bin/conda shell.bash hook)"
#conda activate membrane_fingering
eval "$(/das/home/fische_r/miniconda3/bin/conda shell.bash hook)"
conda activate base

# debugging flags (optional)
# export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK


# Execute command in the container, ipython for debugging to avoid defaul python, change back to python eventually
# srun python -u train_random.py $ARGS
srun python -u ~/lib/co2ely-tomcat/processing/00_cropping.py -s $1
