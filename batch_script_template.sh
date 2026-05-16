#!/bin/bash
#SBATCH --job-name=fairness_<job_identifier>                     # Job name
#SBATCH --output=batch_logs/<job_identifier>_output-%j.log       # Output log file
#SBATCH --error=batch_logs/<job_identifier>_error-%j.log         # Error log file
#SBATCH --time=<max_time>                                        # (-t) Max run time in hh:mm:ss format
#SBATCH --nodes=1                                                # Limit resources to a number of nodes (fairnessBench isn't optimized to spread across nodes)
#SBATCH --mem=<amount>G                                          # Total virtual memory required in GB
#SBATCH --partition=<option1>,<option2>,...                      # (-p) Choose partition (gpu,cpu,etc.)
#SBATCH --gpus=<num_of_devices>				                     # Number of GPUs IF partition was gpu. Must equal number of concurrent tasks
#SBATCH --constraint=vram<8/16/24/32/48/80>                      # Specifically request that each GPU has amount of memory (instead of gres)
#SBATCH --cpus-per-gpu=3			                             # IF partition was gpu
#SBATCH --cpus-per-task=3			                             # IF partition was cpu

# For GPU jobs, to confirm type of GPUs used for job
echo
nvidia-smi
echo


source .venv/bin/activate
echo "virtual env activated"

export HF_HOME=/datasets/ai/<local_model_path> # For Unity local LLMs
echo "HF_HOME set to <model>"

# Path where final_exp_logs and workspace will be stored
export LOG_PATH=<path> 

cd fairnessBench/
echo "In "$(pwd)

echo run_experiments.sh
echo

bash run_experiments.sh



#SBATCH --array=<range>%<max_concurrent_jobs>                    # If running an array of batch jobs
#SBATCH --qos=short                                              # Used for jobs that are anticipated to be done in less that 2 hours
