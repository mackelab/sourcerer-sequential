#!/bin/bash
  
# Sample Slurm job script for Galvani 

#SBATCH -J sequential              # Job name
#SBATCH --ntasks=1                 # Number of tasks
#SBATCH --cpus-per-task=8          # Number of CPU cores per task
#SBATCH --nodes=1                  # Ensure that all cores are on the same machine with nodes=1
#SBATCH --partition=2080-galvani   # Which partition will run your job
#SBATCH --time=1-10:05             # Allowed runtime in D-HH:MM
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

cd ..

# $1-simulator $2-seed
echo $1
echo $2

# Compute Phase
srun python notebooks/benchmark_sequential_naive_script.py base.tag=run_1000_1_$2 base.folder=seq_$1_1000_1_$2 surrogate=train_surrogate base.seed=$2 simulator=$1 sequential.total_simulation_budget=1000 sequential.number_of_iterations=1
srun python notebooks/benchmark_sequential_naive_script.py base.tag=run_1000_2_$2 base.folder=seq_$1_1000_2_$2 surrogate=train_surrogate base.seed=$2 simulator=$1 sequential.total_simulation_budget=1000 sequential.number_of_iterations=2
srun python notebooks/benchmark_sequential_naive_script.py base.tag=run_1000_3_$2 base.folder=seq_$1_1000_3_$2 surrogate=train_surrogate base.seed=$2 simulator=$1 sequential.total_simulation_budget=1000 sequential.number_of_iterations=3
srun python notebooks/benchmark_sequential_naive_script.py base.tag=run_1000_4_$2 base.folder=seq_$1_1000_4_$2 surrogate=train_surrogate base.seed=$2 simulator=$1 sequential.total_simulation_budget=1000 sequential.number_of_iterations=4
srun python notebooks/benchmark_sequential_naive_script.py base.tag=run_1000_5_$2 base.folder=seq_$1_1000_5_$2 surrogate=train_surrogate base.seed=$2 simulator=$1 sequential.total_simulation_budget=1000 sequential.number_of_iterations=5
srun python notebooks/benchmark_sequential_naive_script.py base.tag=run_1000_6_$2 base.folder=seq_$1_1000_6_$2 surrogate=train_surrogate base.seed=$2 simulator=$1 sequential.total_simulation_budget=1000 sequential.number_of_iterations=6
srun python notebooks/benchmark_sequential_naive_script.py base.tag=run_1000_8_$2 base.folder=seq_$1_1000_8_$2 surrogate=train_surrogate base.seed=$2 simulator=$1 sequential.total_simulation_budget=1000 sequential.number_of_iterations=8
srun python notebooks/benchmark_sequential_naive_script.py base.tag=run_1000_10_$2 base.folder=seq_$1_1000_10_$2 surrogate=train_surrogate base.seed=$2 simulator=$1 sequential.total_simulation_budget=1000 sequential.number_of_iterations=10
srun python notebooks/benchmark_sequential_naive_script.py base.tag=run_2000_1_$2 base.folder=seq_$1_2000_1_$2 surrogate=train_surrogate base.seed=$2 simulator=$1 sequential.total_simulation_budget=2000 sequential.number_of_iterations=1
srun python notebooks/benchmark_sequential_naive_script.py base.tag=run_2000_2_$2 base.folder=seq_$1_2000_2_$2 surrogate=train_surrogate base.seed=$2 simulator=$1 sequential.total_simulation_budget=2000 sequential.number_of_iterations=2
srun python notebooks/benchmark_sequential_naive_script.py base.tag=run_2000_3_$2 base.folder=seq_$1_2000_3_$2 surrogate=train_surrogate base.seed=$2 simulator=$1 sequential.total_simulation_budget=2000 sequential.number_of_iterations=3
srun python notebooks/benchmark_sequential_naive_script.py base.tag=run_2000_4_$2 base.folder=seq_$1_2000_4_$2 surrogate=train_surrogate base.seed=$2 simulator=$1 sequential.total_simulation_budget=2000 sequential.number_of_iterations=4
srun python notebooks/benchmark_sequential_naive_script.py base.tag=run_2000_5_$2 base.folder=seq_$1_2000_5_$2 surrogate=train_surrogate base.seed=$2 simulator=$1 sequential.total_simulation_budget=2000 sequential.number_of_iterations=5
srun python notebooks/benchmark_sequential_naive_script.py base.tag=run_2000_6_$2 base.folder=seq_$1_2000_6_$2 surrogate=train_surrogate base.seed=$2 simulator=$1 sequential.total_simulation_budget=2000 sequential.number_of_iterations=6
srun python notebooks/benchmark_sequential_naive_script.py base.tag=run_2000_8_$2 base.folder=seq_$1_2000_8_$2 surrogate=train_surrogate base.seed=$2 simulator=$1 sequential.total_simulation_budget=2000 sequential.number_of_iterations=8
srun python notebooks/benchmark_sequential_naive_script.py base.tag=run_2000_10_$2 base.folder=seq_$1_2000_10_$2 surrogate=train_surrogate base.seed=$2 simulator=$1 sequential.total_simulation_budget=2000 sequential.number_of_iterations=10
srun python notebooks/benchmark_sequential_naive_script.py base.tag=run_3000_1_$2 base.folder=seq_$1_3000_1_$2 surrogate=train_surrogate base.seed=$2 simulator=$1 sequential.total_simulation_budget=3000 sequential.number_of_iterations=1
srun python notebooks/benchmark_sequential_naive_script.py base.tag=run_3000_2_$2 base.folder=seq_$1_3000_2_$2 surrogate=train_surrogate base.seed=$2 simulator=$1 sequential.total_simulation_budget=3000 sequential.number_of_iterations=2
srun python notebooks/benchmark_sequential_naive_script.py base.tag=run_3000_3_$2 base.folder=seq_$1_3000_3_$2 surrogate=train_surrogate base.seed=$2 simulator=$1 sequential.total_simulation_budget=3000 sequential.number_of_iterations=3
srun python notebooks/benchmark_sequential_naive_script.py base.tag=run_3000_4_$2 base.folder=seq_$1_3000_4_$2 surrogate=train_surrogate base.seed=$2 simulator=$1 sequential.total_simulation_budget=3000 sequential.number_of_iterations=4
srun python notebooks/benchmark_sequential_naive_script.py base.tag=run_3000_5_$2 base.folder=seq_$1_3000_5_$2 surrogate=train_surrogate base.seed=$2 simulator=$1 sequential.total_simulation_budget=3000 sequential.number_of_iterations=5
srun python notebooks/benchmark_sequential_naive_script.py base.tag=run_3000_6_$2 base.folder=seq_$1_3000_6_$2 surrogate=train_surrogate base.seed=$2 simulator=$1 sequential.total_simulation_budget=3000 sequential.number_of_iterations=6
srun python notebooks/benchmark_sequential_naive_script.py base.tag=run_3000_8_$2 base.folder=seq_$1_3000_8_$2 surrogate=train_surrogate base.seed=$2 simulator=$1 sequential.total_simulation_budget=3000 sequential.number_of_iterations=8
srun python notebooks/benchmark_sequential_naive_script.py base.tag=run_3000_10_$2 base.folder=seq_$1_3000_10_$2 surrogate=train_surrogate base.seed=$2 simulator=$1 sequential.total_simulation_budget=3000 sequential.number_of_iterations=10
srun python notebooks/benchmark_sequential_naive_script.py base.tag=run_4000_1_$2 base.folder=seq_$1_4000_1_$2 surrogate=train_surrogate base.seed=$2 simulator=$1 sequential.total_simulation_budget=4000 sequential.number_of_iterations=1
srun python notebooks/benchmark_sequential_naive_script.py base.tag=run_4000_2_$2 base.folder=seq_$1_4000_2_$2 surrogate=train_surrogate base.seed=$2 simulator=$1 sequential.total_simulation_budget=4000 sequential.number_of_iterations=2
srun python notebooks/benchmark_sequential_naive_script.py base.tag=run_4000_3_$2 base.folder=seq_$1_4000_3_$2 surrogate=train_surrogate base.seed=$2 simulator=$1 sequential.total_simulation_budget=4000 sequential.number_of_iterations=3
srun python notebooks/benchmark_sequential_naive_script.py base.tag=run_4000_4_$2 base.folder=seq_$1_4000_4_$2 surrogate=train_surrogate base.seed=$2 simulator=$1 sequential.total_simulation_budget=4000 sequential.number_of_iterations=4
srun python notebooks/benchmark_sequential_naive_script.py base.tag=run_4000_5_$2 base.folder=seq_$1_4000_5_$2 surrogate=train_surrogate base.seed=$2 simulator=$1 sequential.total_simulation_budget=4000 sequential.number_of_iterations=5
srun python notebooks/benchmark_sequential_naive_script.py base.tag=run_4000_6_$2 base.folder=seq_$1_4000_6_$2 surrogate=train_surrogate base.seed=$2 simulator=$1 sequential.total_simulation_budget=4000 sequential.number_of_iterations=6
srun python notebooks/benchmark_sequential_naive_script.py base.tag=run_4000_8_$2 base.folder=seq_$1_4000_8_$2 surrogate=train_surrogate base.seed=$2 simulator=$1 sequential.total_simulation_budget=4000 sequential.number_of_iterations=8
srun python notebooks/benchmark_sequential_naive_script.py base.tag=run_4000_10_$2 base.folder=seq_$1_4000_10_$2 surrogate=train_surrogate base.seed=$2 simulator=$1 sequential.total_simulation_budget=4000 sequential.number_of_iterations=10
srun python notebooks/benchmark_sequential_naive_script.py base.tag=run_5000_1_$2 base.folder=seq_$1_5000_1_$2 surrogate=train_surrogate base.seed=$2 simulator=$1 sequential.total_simulation_budget=5000 sequential.number_of_iterations=1
srun python notebooks/benchmark_sequential_naive_script.py base.tag=run_5000_2_$2 base.folder=seq_$1_5000_2_$2 surrogate=train_surrogate base.seed=$2 simulator=$1 sequential.total_simulation_budget=5000 sequential.number_of_iterations=2
srun python notebooks/benchmark_sequential_naive_script.py base.tag=run_5000_3_$2 base.folder=seq_$1_5000_3_$2 surrogate=train_surrogate base.seed=$2 simulator=$1 sequential.total_simulation_budget=5000 sequential.number_of_iterations=3
srun python notebooks/benchmark_sequential_naive_script.py base.tag=run_5000_4_$2 base.folder=seq_$1_5000_4_$2 surrogate=train_surrogate base.seed=$2 simulator=$1 sequential.total_simulation_budget=5000 sequential.number_of_iterations=4
srun python notebooks/benchmark_sequential_naive_script.py base.tag=run_5000_5_$2 base.folder=seq_$1_5000_5_$2 surrogate=train_surrogate base.seed=$2 simulator=$1 sequential.total_simulation_budget=5000 sequential.number_of_iterations=5
srun python notebooks/benchmark_sequential_naive_script.py base.tag=run_5000_6_$2 base.folder=seq_$1_5000_6_$2 surrogate=train_surrogate base.seed=$2 simulator=$1 sequential.total_simulation_budget=5000 sequential.number_of_iterations=6
srun python notebooks/benchmark_sequential_naive_script.py base.tag=run_5000_8_$2 base.folder=seq_$1_5000_8_$2 surrogate=train_surrogate base.seed=$2 simulator=$1 sequential.total_simulation_budget=5000 sequential.number_of_iterations=8
srun python notebooks/benchmark_sequential_naive_script.py base.tag=run_5000_10_$2 base.folder=seq_$1_5000_10_$2 surrogate=train_surrogate base.seed=$2 simulator=$1 sequential.total_simulation_budget=5000 sequential.number_of_iterations=10
srun python notebooks/benchmark_sequential_naive_script.py base.tag=run_6000_1_$2 base.folder=seq_$1_6000_1_$2 surrogate=train_surrogate base.seed=$2 simulator=$1 sequential.total_simulation_budget=6000 sequential.number_of_iterations=1
srun python notebooks/benchmark_sequential_naive_script.py base.tag=run_6000_2_$2 base.folder=seq_$1_6000_2_$2 surrogate=train_surrogate base.seed=$2 simulator=$1 sequential.total_simulation_budget=6000 sequential.number_of_iterations=2
srun python notebooks/benchmark_sequential_naive_script.py base.tag=run_6000_3_$2 base.folder=seq_$1_6000_3_$2 surrogate=train_surrogate base.seed=$2 simulator=$1 sequential.total_simulation_budget=6000 sequential.number_of_iterations=3
srun python notebooks/benchmark_sequential_naive_script.py base.tag=run_6000_4_$2 base.folder=seq_$1_6000_4_$2 surrogate=train_surrogate base.seed=$2 simulator=$1 sequential.total_simulation_budget=6000 sequential.number_of_iterations=4
srun python notebooks/benchmark_sequential_naive_script.py base.tag=run_6000_5_$2 base.folder=seq_$1_6000_5_$2 surrogate=train_surrogate base.seed=$2 simulator=$1 sequential.total_simulation_budget=6000 sequential.number_of_iterations=5
srun python notebooks/benchmark_sequential_naive_script.py base.tag=run_6000_6_$2 base.folder=seq_$1_6000_6_$2 surrogate=train_surrogate base.seed=$2 simulator=$1 sequential.total_simulation_budget=6000 sequential.number_of_iterations=6
srun python notebooks/benchmark_sequential_naive_script.py base.tag=run_6000_8_$2 base.folder=seq_$1_6000_8_$2 surrogate=train_surrogate base.seed=$2 simulator=$1 sequential.total_simulation_budget=6000 sequential.number_of_iterations=8
srun python notebooks/benchmark_sequential_naive_script.py base.tag=run_6000_10_$2 base.folder=seq_$1_6000_10_$2 surrogate=train_surrogate base.seed=$2 simulator=$1 sequential.total_simulation_budget=6000 sequential.number_of_iterations=10
srun python notebooks/benchmark_sequential_naive_script.py base.tag=run_8000_1_$2 base.folder=seq_$1_8000_1_$2 surrogate=train_surrogate base.seed=$2 simulator=$1 sequential.total_simulation_budget=8000 sequential.number_of_iterations=1
srun python notebooks/benchmark_sequential_naive_script.py base.tag=run_8000_2_$2 base.folder=seq_$1_8000_2_$2 surrogate=train_surrogate base.seed=$2 simulator=$1 sequential.total_simulation_budget=8000 sequential.number_of_iterations=2
srun python notebooks/benchmark_sequential_naive_script.py base.tag=run_8000_3_$2 base.folder=seq_$1_8000_3_$2 surrogate=train_surrogate base.seed=$2 simulator=$1 sequential.total_simulation_budget=8000 sequential.number_of_iterations=3
srun python notebooks/benchmark_sequential_naive_script.py base.tag=run_8000_4_$2 base.folder=seq_$1_8000_4_$2 surrogate=train_surrogate base.seed=$2 simulator=$1 sequential.total_simulation_budget=8000 sequential.number_of_iterations=4
srun python notebooks/benchmark_sequential_naive_script.py base.tag=run_8000_5_$2 base.folder=seq_$1_8000_5_$2 surrogate=train_surrogate base.seed=$2 simulator=$1 sequential.total_simulation_budget=8000 sequential.number_of_iterations=5
srun python notebooks/benchmark_sequential_naive_script.py base.tag=run_8000_6_$2 base.folder=seq_$1_8000_6_$2 surrogate=train_surrogate base.seed=$2 simulator=$1 sequential.total_simulation_budget=8000 sequential.number_of_iterations=6
srun python notebooks/benchmark_sequential_naive_script.py base.tag=run_8000_8_$2 base.folder=seq_$1_8000_8_$2 surrogate=train_surrogate base.seed=$2 simulator=$1 sequential.total_simulation_budget=8000 sequential.number_of_iterations=8
srun python notebooks/benchmark_sequential_naive_script.py base.tag=run_8000_10_$2 base.folder=seq_$1_8000_10_$2 surrogate=train_surrogate base.seed=$2 simulator=$1 sequential.total_simulation_budget=8000 sequential.number_of_iterations=10
srun python notebooks/benchmark_sequential_naive_script.py base.tag=run_12000_1_$2 base.folder=seq_$1_12000_1_$2 surrogate=train_surrogate base.seed=$2 simulator=$1 sequential.total_simulation_budget=12000 sequential.number_of_iterations=1
srun python notebooks/benchmark_sequential_naive_script.py base.tag=run_12000_2_$2 base.folder=seq_$1_12000_2_$2 surrogate=train_surrogate base.seed=$2 simulator=$1 sequential.total_simulation_budget=12000 sequential.number_of_iterations=2
srun python notebooks/benchmark_sequential_naive_script.py base.tag=run_12000_3_$2 base.folder=seq_$1_12000_3_$2 surrogate=train_surrogate base.seed=$2 simulator=$1 sequential.total_simulation_budget=12000 sequential.number_of_iterations=3
srun python notebooks/benchmark_sequential_naive_script.py base.tag=run_12000_4_$2 base.folder=seq_$1_12000_4_$2 surrogate=train_surrogate base.seed=$2 simulator=$1 sequential.total_simulation_budget=12000 sequential.number_of_iterations=4
srun python notebooks/benchmark_sequential_naive_script.py base.tag=run_12000_5_$2 base.folder=seq_$1_12000_5_$2 surrogate=train_surrogate base.seed=$2 simulator=$1 sequential.total_simulation_budget=12000 sequential.number_of_iterations=5
srun python notebooks/benchmark_sequential_naive_script.py base.tag=run_12000_6_$2 base.folder=seq_$1_12000_6_$2 surrogate=train_surrogate base.seed=$2 simulator=$1 sequential.total_simulation_budget=12000 sequential.number_of_iterations=6
srun python notebooks/benchmark_sequential_naive_script.py base.tag=run_12000_8_$2 base.folder=seq_$1_12000_8_$2 surrogate=train_surrogate base.seed=$2 simulator=$1 sequential.total_simulation_budget=12000 sequential.number_of_iterations=8
srun python notebooks/benchmark_sequential_naive_script.py base.tag=run_12000_10_$2 base.folder=seq_$1_12000_10_$2 surrogate=train_surrogate base.seed=$2 simulator=$1 sequential.total_simulation_budget=12000 sequential.number_of_iterations=10
srun python notebooks/benchmark_sequential_naive_script.py base.tag=run_15000_1_$2 base.folder=seq_$1_15000_1_$2 surrogate=train_surrogate base.seed=$2 simulator=$1 sequential.total_simulation_budget=15000 sequential.number_of_iterations=1
srun python notebooks/benchmark_sequential_naive_script.py base.tag=run_15000_2_$2 base.folder=seq_$1_15000_2_$2 surrogate=train_surrogate base.seed=$2 simulator=$1 sequential.total_simulation_budget=15000 sequential.number_of_iterations=2
srun python notebooks/benchmark_sequential_naive_script.py base.tag=run_15000_3_$2 base.folder=seq_$1_15000_3_$2 surrogate=train_surrogate base.seed=$2 simulator=$1 sequential.total_simulation_budget=15000 sequential.number_of_iterations=3
srun python notebooks/benchmark_sequential_naive_script.py base.tag=run_15000_4_$2 base.folder=seq_$1_15000_4_$2 surrogate=train_surrogate base.seed=$2 simulator=$1 sequential.total_simulation_budget=15000 sequential.number_of_iterations=4
srun python notebooks/benchmark_sequential_naive_script.py base.tag=run_15000_5_$2 base.folder=seq_$1_15000_5_$2 surrogate=train_surrogate base.seed=$2 simulator=$1 sequential.total_simulation_budget=15000 sequential.number_of_iterations=5
srun python notebooks/benchmark_sequential_naive_script.py base.tag=run_15000_6_$2 base.folder=seq_$1_15000_6_$2 surrogate=train_surrogate base.seed=$2 simulator=$1 sequential.total_simulation_budget=15000 sequential.number_of_iterations=6
srun python notebooks/benchmark_sequential_naive_script.py base.tag=run_15000_8_$2 base.folder=seq_$1_15000_8_$2 surrogate=train_surrogate base.seed=$2 simulator=$1 sequential.total_simulation_budget=15000 sequential.number_of_iterations=8
srun python notebooks/benchmark_sequential_naive_script.py base.tag=run_15000_10_$2 base.folder=seq_$1_15000_10_$2 surrogate=train_surrogate base.seed=$2 simulator=$1 sequential.total_simulation_budget=15000 sequential.number_of_iterations=10

