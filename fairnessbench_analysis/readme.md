# fairnessbench analysis

This folder contains all the code and data for analyzing the fairnessbench results.
The main analysis script is explode_results.py, which loads the raw results data and creates clean CSV files ready for analysis. 

# A. Setup:

**Local path configuration**
1. Create `paths.py` at the repo root.
2. Create the csv_files and graphs directories, then set CSV_FILES and GRAPHS in paths.py to the absolute paths on your machine.

### Required variables in `paths.py`
- **PROJECT_ROOT** — Directory that contains all *raw results*. 
- **CSV_FILES** — Directory that contains the *clean CSV files* produced by `explode_results.py`.
- **GRAPHS** — Directory where analysis scripts will save generated *figures/plots*.
- **FILES** — Directory that stores *CSV files from different evaluation models* (used by `cv_scores_evalmodels.py`).

# B. Run Analysis:
**Run main file**
```python
python explode_results.py
```
This will create the following files in the csv_files/ directory:
- Result_Final_code_clean*.csv: File contains raw scores and final scores from the llm evaluation on the training scripts(code).
- Result_Final_log_clean*: File contains raw scores and final scores from the llm evaluation on the reasoning process of the agent(log).
- Final_step_performance*.csv: File contains performance metric(e.g. accuracy,disparate impact etc.) scores of the models on each task.
These files are then used for futher analysis 
**Analysis**
In FairnessBench we run several analyses on our results. Each `.py` file performs a different analysis and generates plots that are stored in the `graphs/` directory.
To run an analysis, change the input CSV filename in the script to the file required for that analysis.
**Example:** To analyze different types of fairness for the Adult dataset, run `adult_fairness.py`. Before running it, update the script’s input CSV to the new file generated in the `csv_files/` directory.

```python
python ....py
```
***Key files:***
`- adult_di_code_llmeval.py: Analyzes disparate impact (DI) for the Adult dataset using the LLM code evaluation.`\
`- adult_fairness.py: Analyze the fairness metrics used in the benchmark for the Adult dataset.`\
`- balancing_fairness.py: Analyzes the fairness metrics used in the benchmark for the Balance research problem.`\
`- code_log_llm_eval.py: Generates plots showing how rubric-section scores vary across tasks and datasets for both code and log evaluations.`\
`- correlation_flake8_code.py: Analyzes the correlation between LLM code-evaluation scores and Flake8 (linter-based) scores.`\
`- cv_scores_evalmodels.py: Analyzes the consistency of different evaluation models based on cross-validation (CV) scores.`\
`- di_across_datasets.py: Analyzes disparate impact (DI) and accuracy across datasets and research problems.`\
`- performance_flake8_code.py: Generates a scatter plot of Flake8 performance vs. LLM code-evaluation scores across models, datasets, and research problems.`\
`- target_selection.py: Analyze the models performance and fairness metrics for the targey selection tasks.`\
`- target10_sucess_rate.py: Generates a plot showing how agent performance differs from the baseline on the Target10 research problem, by dataset and model.`\


