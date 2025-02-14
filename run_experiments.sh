#/bin/bash

### AS ###
# Stop 1: 
# This script is a starting point to run example tasks
# This scrip calls on the multi_run_experiments.sh
# First 4 args are the name of log_dir, task name, '8' and {0..7}

# example tasks to run
# all_tasks="cifar10 imdb"
all_tasks="adult"
log_dir=final_exp_logs
# AS: 
# models="claude-2.1 gpt-4-0125-preview gemini-pro"
models="huggingface/codellama/CodeLlama-7b-hf" # AS: This name was determined by the LLM.py module. It is required to follow this name. But don't know how to use the local


# Run listed tasks 
for model in $models
do
    for task in $all_tasks
    do  
        bash multi_run_experiment.sh $log_dir/$model/$task $task 8 {0..7} --llm-name $model --edit-script-llm-name $model --fast-llm-name $model
    done
done

# other agent variants

# What does --retrival do??
for task in $all_tasks
do 
    bash multi_run_experiment.sh $log_dir/retrieval/$task $task 8 {0..7}  --retrieval 
done


## Agents do something with the log files (What was being set up in the previous two steps)

# for task in $all_tasks
# do 
#     bash multi_run_experiment.sh $log_dir/autogpt/$task $task 8 {0..7} --agent-type AutoGPTAgent 
# done

for task in $all_tasks
do 
    bash multi_run_experiment.sh $log_dir/react/$task $task 8 {0..7} --agent-type ReasoningActionAgent 
done 


for task in $all_tasks
do 
    bash multi_run_experiment.sh $log_dir/langchain/$task $task 8 {0..7}  --agent-type LangChainAgent 
done
