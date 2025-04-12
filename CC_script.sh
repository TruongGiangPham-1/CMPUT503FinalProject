#!/bin/bash
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=16G
#SBATCH --time=60:00:00
#SBATCH --account=rrg-mtaylor3
#SBATCH --output=/home/truonggi/scratch/slurm_out/%A.out
#SBATCH --mail-user=truonggi@ualberta.ca
#SBATCH --mail-type=ALL


export results=$SLURM_TMPDIR/results
export data=$SLURM_TMPDIR/data

module load python/3.10
module load cuda
module load gcc opencv/4.9.0
source /home/truonggi/scratch/CMPUT503FinalProject/env/bin/activate

echo $1  # run label

python3 ppo.py --total_timesteps 1000000 --num_steps 2048  --run_label $1 --video
