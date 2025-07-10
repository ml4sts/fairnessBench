# Fairness Bench

This is a benchmark to evaluate AI capabilities to do fair data driven decision-making.

The benchmark consists of several tasks.

A fairnessBench task is defined as follows:
For a dataset and a very simple training script that uses logistic regression model. 
How well can an LLM agent improve the training script to achieve high fairness metrics.

## Fairness Metrics:



## Different LLM models used for agent

## Different LLM models used for eval

## What is baseline

## What does eval do?

## Reading eval results


Roles:
- task-specific: environment files for the task, the train.py, etc
- benchmarking infrastructure: code needed to overall run benchmark, scoring etc (`eval-<type>.py`)
- agent: agent tools, agent prompts, etc


| file | description | role|
|--- |---| ---|


## Instructions for running the benchmark:
Pick a task/ list of tasks to run from tasks.json.
Run using run_experiment.sh

### run_experiment.sh

- log_dir
  - The directory name for the llm to create logs
- models
  - The models you want to use for the tasks
  - claude-2.1, gpt-4-0125-preview, gemini-pro, gpt-4o-mini, gpt-4o, llama, qwen, granite, claude-3-7-sonnet-20250219, claude-3-5-haiku-20241022, claude-3-opus-20240229
- edit_script_model

- fast_llm
### eval.sh

- log_dir
  - directory that the llm placed the experiment logs
- json_folder
  - 
- all_tasks
  - list of tasks to be evaluated
- models
  - Models being used to evaluate results
- eval_model
