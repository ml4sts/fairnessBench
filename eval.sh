#/bin/bash

# all_tasks="cifar10 imdb"
all_tasks="adult"
log_dir=final_exp_logs
# models="claude2.1 gpt-4-0125-preview gemini-pro"
models="huggingface/codellama/CodeLlama-7b-hf" # AS: This name was determined by the LLM.py module. It is required to follow this name. But don't know how to use the local

for model in $models
do
    for task in $all_tasks
    do
        echo "python -m fairnessBench.eval --log-folder $log_dir/$model/$task --task $task --output-file ${model}_${task}.json"
        python -m fairnessBench.eval --log-folder $log_dir/$model/$task --task $task --output-file ${model}_${task}.json 
        # add --eval-intermediate to evaluate intermediate steps 
    done
done