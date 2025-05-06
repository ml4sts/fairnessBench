#/bin/bash

# Final step: Evaluation
# This script evaluates each model on the run of each task
# For each model and task we get 1 JSON file

# all_tasks="adult adultrecon germancredit creditdefault sanity_check"
all_tasks="adult_di_best-sex"

log_dir=final_exp_logs

# models="claude2.1 gpt-4-0125-preview gemini-pro"
models="claude" 
eval_model="llama"

for model in $models
do
    for task in $all_tasks
    do
        echo "python -m fairnessBench.eval --log-folder $log_dir/$model/$task --task $task --output-file ${model}_${task}.json"
        python -m fairnessBench.eval --log-folder $log_dir/$model/$task --task $task --output-file ${model}_${task}.json --eval_model $eval_model 
        # add --eval-intermediate to evaluate intermediate steps 
    done
done