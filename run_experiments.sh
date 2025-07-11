#/bin/bash

### AS ###
# Stop 1: 
# This script is a starting point to run example tasks
# This scrip calls on the multi_run_experiments.sh
# First 4 args are the name of log_dir, task name, '8' and {0..7}

if [ $1 ]; then
    bash clean.sh
fi


# Adult tasks to run
# "adult_di_best-sex adult_di_best-race adult_di_target10-sex adult_di_target10-race adult_di_balance-sex adult_di_balance-race adult_spd_best-sex adult_spd_best-race adult_spd_target10-sex adult_spd_target10-race adult_spd_balance-sex adult_spd_balance-race adult_eod_best-sex adult_eod_best-race adult_eod_target10-sex adult_eod_target10-race adult_eod_balance-sex adult_eod_balance-race adult_erd_best-sex adult_erd_best-race adult_erd_target10-sex adult_erd_target10-race adult_erd_balance-sex adult_erd_balance-race adult_err_best-sex adult_err_best-race adult_err_target10-sex adult_err_target10-race adult_err_balance-sex adult_err_balance-race adult_ford_best-sex adult_ford_best-race adult_ford_target10-sex adult_ford_target10-race adult_ford_balance-sex adult_ford_balance-race "

# Adultrecon tasks
# "adrecon_allmetric_targetselection-race adrecon_allmetric_targetselection-gender" 

# Germancredit tasls
# "german_di_best-sex german_di_balance-sex german_eod_best-sex german_eod_balance-sex" 

# Creditdefault
# "creditdefault_di_best-gender creditdefault_di_balance-sex creditdefault_eod_best-sex creditdefault_eod_balance-sex" 

# Randoadult tasks
# all_tasks="randoadult_di_best-race randoadult_di_balance-race randoadult_eod_best-race randoadult_eod_balance-race randoadult_di_best-sex randoadult_di_balance-sex randoadult_eod_best-sex randoadult_eod_balance-sex"

# Sampadult tasks
# "sampadult_di_best-race sampadult_di_balance-race sampadult_di_best-sex sampadult_di_balance-sex sampadult_eod_best-race sampadult_eod_balance-race sampadult_eod_best-sex sampadult_eod_balance-sex"


log_dir=final_exp_logs

# models="claude-2.1 gpt-4-0125-preview gemini-pro"
# models="gpt-4o-mini gpt-4o llama qwen granite claude-3-7-sonnet-20250219 claude-3-5-haiku-20241022 claude-3-opus-20240229"
models="qwen"

edit_script_model="gpt-4o"
fast_llm="gpt-4o"


# Run listed tasks 
for model in $models
do
    for task in $all_tasks
    do  
        bash multi_run_experiment.sh $log_dir/$model/$task $task 1 0 --llm-name $model --edit-script-llm-name $model --fast-llm-name $model
    done
done
echo

# other agent variants

# Retrieval is a varient of the ResearchAgent that just adds trace history to the prompt
# for task in $all_tasks
# do 
#     bash multi_run_experiment.sh $log_dir/retrieval/$task $task 1 0  --retrieval 
# done
# echo


# AS: Agents do something with the log files (What was being set up in the previous two steps)

# AS: Don't have gpt credits
# for task in $all_tasks
# do 
#     bash multi_run_experiment.sh $log_dir/autogpt/$task $task 8 {0..7} --agent-type AutoGPTAgent 
# done

# for task in $all_tasks
# do 
#     bash multi_run_experiment.sh $log_dir/react/$task $task 1 0 --agent-type ReasoningActionAgent 
# done 
# echo


# for task in $all_tasks
# do 
#     bash multi_run_experiment.sh $log_dir/langchain/$task $task 1 0  --agent-type LangChainAgent 
# done
