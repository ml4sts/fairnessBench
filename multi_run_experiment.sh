#/bin/bash

### AS ###
# Stop 2: 
# This script prepares the task called, creates necessary log folders and runs the task with runner
# For every task this script will be run at least 3 times; model, retrival, agent/s
# This scrip calls on the runner.py

# grab preliminary info
exp_path=$1
task=$2
n_device=$3
shift 3

# The values here are normally {0..7}
declare -a devices=()

# Get X numbers
for (( i=0; i<$n_device; i++ ))
do
  devices+=($1)
  shift 
done



extra_args="${@}"
# Don't know what this name change is doing 
folder=$exp_path
python=$(which python)

echo "exp_path: $exp_path"
echo "task: $task"
echo "n_devices: $n_device"
echo "devcies: ${devices[@]}"
echo "extra_args: $extra_args"

echo "Logs will be saved to $folder"

# Looks like simply the number if times this will happen
for i in "${devices[@]}"
do 
  # time in current Unix timestamp
  ts=$(date +%s)

  # Check for log folder with a time-named folder in it or create one
  if [ -d $folder/$ts ]; then
      echo "Folder $folder/$ts already exists. removing it"
      rm -rf $folder/$ts
  fi
  mkdir -p $folder/$ts

  # Call the prepare task script
  python -u -m fairnessBench.prepare_task $task $python
  
  # Printing command for debugging purposes and executing task with runner.py
  echo "python -u -m fairnessBench.runner --python $python --task $task --device $i --log-dir $folder/$ts  --work-dir workspaces/$folder/$ts ${extra_args} > $folder/$ts/log 2>&1"
  eval "python -u -m fairnessBench.runner --python $python --task $task --device $i --log-dir $folder/$ts  --work-dir workspaces/$folder/$ts ${extra_args}" > $folder/$ts/log 2>&1 &

  # Are we sleeping to allow the LLM to finish its work?
  sleep 2
done
# Why??
wait

