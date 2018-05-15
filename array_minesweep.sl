#!/bin/bash
#SBATCH --array=1-200
#SBATCH -J Machine_Learning
#SBATCH -t 10:00:00
#SBATCH -o out/MACHINE_Learning_array%A%a.out
#SBATCH -e err/MACHINE_Learning_array%A%a.err

echo 'The seed for setting parameters is: ' $SLURM_ARRAY_TASK_ID
python3 array_minesweep.py $SLURM_ARRAY_TASK_ID models/Model_${SLURM_ARRAY_TASK_ID} 
