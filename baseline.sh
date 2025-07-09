#/bin/bash

# This run will use the base agent just to run the raw train.py of tasks before any agent modifying them to have prediction metrics to compare to


# Adult tasks
# Adultrecon tasks
# Germancredit tasks
# Creditdefault
# Randoadult tasks
# Sampadult tasks
all_tasks="adult_di_best-sex adult_di_best-race adult_di_target10-sex adult_di_target10-race adult_di_balance-sex adult_di_balance-race adult_spd_best-sex adult_spd_best-race adult_spd_target10-sex adult_spd_target10-race adult_spd_balance-sex adult_spd_balance-race adult_eod_best-sex adult_eod_best-race adult_eod_target10-sex adult_eod_target10-race adult_eod_balance-sex adult_eod_balance-race adult_erd_best-sex adult_erd_best-race adult_erd_target10-sex adult_erd_target10-race adult_erd_balance-sex adult_erd_balance-race adult_err_best-sex adult_err_best-race adult_err_target10-sex adult_err_target10-race adult_err_balance-sex adult_err_balance-race adult_ford_best-sex adult_ford_best-race adult_ford_target10-sex adult_ford_target10-race adult_ford_balance-sex adult_ford_balance-race adrecon_allmetric_targetselection-race adrecon_allmetric_targetselection-gender german_di_best-sex german_di_balance-sex german_eod_best-sex german_eod_balance-sex creditdefault_di_best-gender creditdefault_di_balance-gender creditdefault_eod_best-gender creditdefault_eod_balance-gender creditdefault_di_balance-sex creditdefault_eod_best-sex creditdefault_eod_balance-sex randoadult_di_best-race randoadult_di_balance-race randoadult_eod_best-race randoadult_eod_balance-race randoadult_di_best-sex randoadult_di_balance-sex randoadult_eod_best-sex randoadult_eod_balance-sex sampadult_di_best-race sampadult_di_balance-race sampadult_di_best-sex sampadult_di_balance-sex sampadult_eod_best-race sampadult_eod_balance-race sampadult_eod_best-sex sampadult_eod_balance-sex" # AS: 

for task in $all_tasks
do    
    # 3 runs sanity check
    bash multi_run_experiment.sh final_exp_logs/sanity_check/$task $task 1 0 --agent-type Agent
    # bash multi_run_experiment.sh final_exp_logs/sanity_check/$task $task 1 0 --agent-type Agent
    # bash multi_run_experiment.sh final_exp_logs/sanity_check/$task $task 1 0 --agent-type Agent
done

