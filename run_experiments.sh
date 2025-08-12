#/bin/bash

# Stop 1: 
# This script is a starting point to run a list of tasks
# This scrip calls on the multi_run_experiments.sh
# First 4 args are the name of log_dir, task name and the number of devices used in parallel

# NOTE: Running this script as is is unrealistic. The (amount of tasks) x (number of models) will end up taking too much time

all_tasks="
adult_di_best-sex
adult_di_best-race
adult_di_balance-sex
adult_di_balance-race
adult_di_target10-sex
adult_di_target10-race
adult_di_implicit-sex
adult_di_implicit-race
adult_spd_best-sex
adult_spd_best-race
adult_spd_balance-sex
adult_spd_balance-race
adult_spd_target10-sex
adult_spd_target10-race
adult_spd_implicit-sex
adult_spd_implicit-race
adult_eod_best-sex
adult_eod_best-race
adult_eod_balance-sex
adult_eod_balance-race
adult_eod_target10-sex
adult_eod_target10-race
adult_eod_implicit-sex
adult_eod_implicit-race
adult_erd_best-sex
adult_erd_best-race
adult_erd_balance-sex
adult_erd_balance-race
adult_erd_target10-sex
adult_erd_target10-race
adult_erd_implicit-sex
adult_erd_implicit-race
adult_err_best-sex
adult_err_best-race
adult_err_balance-sex
adult_err_balance-race
adult_err_target10-sex
adult_err_target10-race
adult_err_implicit-sex
adult_err_implicit-race
adult_ford_best-sex
adult_ford_best-race
adult_ford_balance-sex
adult_ford_balance-race
adult_ford_target10-sex
adult_ford_target10-race
adult_ford_implicit-sex
adult_ford_implicit-race
german_di_best-sex
german_di_balance-sex
german_di_target10-sex
german_di_implicit-sex
german_eod_best-sex
german_eod_balance-sex
german_eod_target10-sex
german_eod_implicit-sex
creditdefault_di_best-gender
creditdefault_di_balance-gender
creditdefault_di_target10-gender
creditdefault_di_implicit-gender
creditdefault_eod_best-gender
creditdefault_eod_balance-gender
creditdefault_eod_target10-gender
creditdefault_eod_implicit-gender
adrecon_allmetric_targetselection-gender
adrecon_allmetric_targetselection-race
sampadult_di_best-race
sampadult_di_balance-race
sampadult_di_target10-race
sampadult_di_implicit-race
sampadult_di_best-sex
sampadult_di_balance-sex
sampadult_di_target10-sex
sampadult_di_implicit-sex
sampadult_eod_best-race
sampadult_eod_balance-race
sampadult_eod_target10-race
sampadult_eod_implicit-race
sampadult_eod_best-sex
sampadult_eod_balance-sex
sampadult_eod_target10-sex
sampadult_eod_implicit-sex
randoadult_di_best-race
randoadult_di_balance-race
randoadult_di_target10-race
randoadult_di_implicit-race
randoadult_eod_best-race
randoadult_eod_balance-race
randoadult_eod_target10-race
randoadult_eod_implicit-race
randoadult_di_best-sex
randoadult_di_balance-sex
randoadult_di_target10-sex
randoadult_di_implicit-sex
randoadult_eod_best-sex
randoadult_eod_balance-sex
randoadult_eod_target10-sex
randoadult_eod_implicit-sex
"

log_dir=final_exp_logs

models="gpt-4o-mini gpt-4o llama qwen granite claude-3-7-sonnet-20250219 claude-3-5-haiku-20241022 claude-3-opus-20240229"

# Run listed tasks with default agent (ResearchAgent)
for model in $models
do

    edit_script_model=$models
    fast_llm=$models
    for task in $all_tasks
    do  
        bash multi_run_experiment.sh $log_dir/$model/$task $task 4 {0..3} --llm-name $model --edit-script-llm-name $model --fast-llm-name $model
    done
done
echo

# Other agent variants are not used or well tested

# Retrieval is a varient of the ResearchAgent that just adds trace history to the prompt
# for task in $all_tasks
# do 
#     bash multi_run_experiment.sh $log_dir/retrieval/$task $task 1 0  --retrieval 
# done
# echo


# for task in $all_tasks
# do 
#     bash multi_run_experiment.sh $log_dir/react/$task $task 1 0 --agent-type ReasoningActionAgent 
# done 
# echo


# for task in $all_tasks
# do 
#     bash multi_run_experiment.sh $log_dir/langchain/$task $task 1 0  --agent-type LangChainAgent 
# done
