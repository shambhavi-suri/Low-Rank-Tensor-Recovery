#!/bin/bash
#SBATCH --job-name=large_3d_test    # create a short name for your job
#SBATCH --nodes=1                # node count
#SBATCH --ntasks=1               # total number of tasks across all nodes
#SBATCH --cpus-per-task=1        # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --mem-per-cpu=45G        # memory per cpu-core (4G is default)
#SBATCH --time=04:00:00          # total run time limit (HH:MM:SS)
#SBATCH --mail-type=begin,end,fail     # send email if job fails or ends
#SBATCH --mail-user=ss3472@princeton.edu
#SBATCH --array=0-10

#version
WORKDIR="files/3d_tensors_run_3_4_8_12_150itr"

mkdir -p "$WORKDIR" && cd "$WORKDIR" || exit -1
module purge
module load anaconda3/2024.10
conda activate tensor-env
python /scratch/network/ss3472/python_test/convergence_code/convergence_large_3d.py
