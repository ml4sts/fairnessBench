#/bin/bash

# Stop 2: 
# This script prepares the task called, creates necessary log folders and runs the task with runner
# For every task this script will be run at least 3 times; model, retrival, agent/s
# This scrip calls on the runner.py

# grab preliminary info
exp_path=$1
task=$2
n_device=$3
shift 3

# Devices are the number of GPUs to run separate runs on
declare -a devices=()

# Get X numbers
for (( i=0; i<$n_device; i++ ))
do
  devices+=($1)
  shift 
done



extra_args="${@}"
folder=$exp_path
python=$(which python)

echo "exp_path: $exp_path"
echo "task: $task"
echo "n_devices: $n_device"
echo "devcies: ${devices[@]}"
echo "extra_args: $extra_args"

echo "Logs will be saved to $folder"

# Create a run ID for each device 
for i in "${devices[@]}"
do 
  # time in current Unix timestamp
  ts=$(date +%s)

  # Check for log folder with a time-named folder in it or create one
  if [ -d "/project/pi_brownsarahm_uri_edu/ayman_uri/fairnessBench/new_$folder/$ts" ]; then
      echo "Folder /project/pi_brownsarahm_uri_edu/ayman_uri/fairnessBench/new_$folder/$ts already exists. removing it"
      rm -rf /project/pi_brownsarahm_uri_edu/ayman_uri/fairnessBench/new_$folder/$ts
  fi
  mkdir -p "/project/pi_brownsarahm_uri_edu/ayman_uri/fairnessBench/new_$folder/$ts"

  # Call the prepare task script
  python -u -m fairnessBench.prepare_task $task $python
  
  # Printing command for debugging purposes and executing task with runner.py
  echo "python -u -m fairnessBench.runner --python $python --task $task --device $i --log-dir /project/pi_brownsarahm_uri_edu/ayman_uri/fairnessBench/new_$folder/$ts  --work-dir /scratch3/workspace/ayman_sandouk_uri_edu-fairness/fairnessBench/workspaces/$folder/$ts ${extra_args}" > /project/pi_brownsarahm_uri_edu/ayman_uri/fairnessBench/new_$folder/$ts/log 2>&1 &
  
  eval "python -u -m fairnessBench.runner --python $python --task $task --device $i --log-dir /project/pi_brownsarahm_uri_edu/ayman_uri/fairnessBench/new_$folder/$ts  --work-dir /scratch3/workspace/ayman_sandouk_uri_edu-fairness/fairnessBench/workspaces/$folder/$ts ${extra_args}" > /project/pi_brownsarahm_uri_edu/ayman_uri/fairnessBench/new_$folder/$ts/log 2>&1 &

  # 2 seconds between runs
  sleep 2
done

# Shouldn't run more than n devices at a time (all GPUs are occupied)
wait

