#!/bin/bash
#SBATCH --job-name=dataset_test
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --mail-user="s3305139@vuw.leidenuniv.nl"
#SBATCH --mail-type="ALL"
#SBATCH --mem-per-cpu=10GB
#SBATCH --time=02:00:00
#SBATCH --partition=cpu-short
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10


echo "[$SHELL] #### Starting Python test"
echo "[$SHELL] ## This is $SLURM_JOB_USER and this job has the ID $SLURM_JOB_ID"
# get the current working directory
export CWD=$(pwd)
echo "[$SHELL] ## current working directory: "$CWD

# Run the file
echo "[$SHELL] ## Run script"

python3 DrugEx/dataset.py


echo "[$SHELL] ## Script finished"

echo "[$SHELL] #### Finished Python test. Have a nice day"
