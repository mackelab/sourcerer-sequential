#!/bin/bash
  
# Sample Slurm job script for Galvani 

#SBATCH -J sequential              # Job name
#SBATCH --ntasks=1                 # Number of tasks
#SBATCH --cpus-per-task=8          # Number of CPU cores per task
#SBATCH --nodes=1                  # Ensure that all cores are on the same machine with nodes=1
#SBATCH --partition=a100-galvani   # Which partition will run your job
#SBATCH --time=1-12:15             # Allowed runtime in D-HH:MM
#SBATCH --gres=gpu:1               # (optional) Requesting type and number of GPUs
#SBATCH --mem=50G                  # Total memory pool for all cores (see also --mem-per-cpu); exceeding this number will cause your job to fail.
#SBATCH --output=/mnt/qb/work/macke/mwe102/sourcerer-sequential/results_sourcerer/myjob-%j.out      # File to which STDOUT will be written - make sure this is not on $HOME
#SBATCH --error=/mnt/qb/work/macke/mwe102/sourcerer-sequential/results_sourcerer/myjob-%j.err       # File to which STDERR will be written - make sure this is not on $HOME

# Diagnostic and Analysis Phase - please leave these in.
scontrol show job $SLURM_JOB_ID
pwd
nvidia-smi # only if you requested gpus
# ls $WORK # not necessary just here to illustrate that $WORK is available here

# Setup Phase
# add possibly other setup code here, e.g.
# - copy singularity images or datasets to local on-compute-node storage like /scratch_local
# - loads virtual envs, like with anaconda
# - set environment variables
# - determine commandline arguments for `srun` calls
source /usr/lib/python3.6/site-packages/conda/shell/etc/profile.d/conda.sh
conda activate /mnt/qb/work/macke/mwe102/.conda/sbi

# this script is run from inside the scripts/ directory, go back to main directory
cd ..

# $1-seed $2-budget
echo $1
echo $2

# Compute Phase
srun python notebooks/hh_sequential_naive_script.py base.tag=run_$2_10_$1 base.folder=seq_hh_$2_10_$1 surrogate=hh_train_surrogate base.seed=$1 source.fin_lambda=0.25 sequential.total_simulation_budget=$2 sequential.number_of_iterations=10
srun python notebooks/hh_sequential_naive_script.py base.tag=run_$2_8_$1 base.folder=seq_hh_$2_8_$1 surrogate=hh_train_surrogate base.seed=$1 source.fin_lambda=0.25 sequential.total_simulation_budget=$2 sequential.number_of_iterations=8
srun python notebooks/hh_sequential_naive_script.py base.tag=run_$2_6_$1 base.folder=seq_hh_$2_6_$1 surrogate=hh_train_surrogate base.seed=$1 source.fin_lambda=0.25 sequential.total_simulation_budget=$2 sequential.number_of_iterations=6
srun python notebooks/hh_sequential_naive_script.py base.tag=run_$2_5_$1 base.folder=seq_hh_$2_5_$1 surrogate=hh_train_surrogate base.seed=$1 source.fin_lambda=0.25 sequential.total_simulation_budget=$2 sequential.number_of_iterations=5
srun python notebooks/hh_sequential_naive_script.py base.tag=run_$2_4_$1 base.folder=seq_hh_$2_4_$1 surrogate=hh_train_surrogate base.seed=$1 source.fin_lambda=0.25 sequential.total_simulation_budget=$2 sequential.number_of_iterations=4
srun python notebooks/hh_sequential_naive_script.py base.tag=run_$2_3_$1 base.folder=seq_hh_$2_3_$1 surrogate=hh_train_surrogate base.seed=$1 source.fin_lambda=0.25 sequential.total_simulation_budget=$2 sequential.number_of_iterations=3
srun python notebooks/hh_sequential_naive_script.py base.tag=run_$2_2_$1 base.folder=seq_hh_$2_2_$1 surrogate=hh_train_surrogate base.seed=$1 source.fin_lambda=0.25 sequential.total_simulation_budget=$2 sequential.number_of_iterations=2
srun python notebooks/hh_sequential_naive_script.py base.tag=run_$2_1_$1 base.folder=seq_hh_$2_1_$1 surrogate=hh_train_surrogate base.seed=$1 source.fin_lambda=0.25 sequential.total_simulation_budget=$2 sequential.number_of_iterations=1


