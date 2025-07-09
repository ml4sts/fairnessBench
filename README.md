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