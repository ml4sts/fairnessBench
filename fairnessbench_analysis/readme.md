# fairnessbench analysis

This folder contains all the code and data for analyzing the fairnessbench results.
The main analysis script is explode_results.py, which loads the raw results data and creates clean CSV files ready for analysis. 

# A. Setup:

**Local path configuration**
1. Create `paths.py` at the repo root.
2. Create the csv_files and graphs directories, then set CSV_FILES and GRAPHS in paths.py to the absolute paths on your machine.
3. `paths.py` is in `.gitignore`.
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

These files are then used for futher analysis. 

**Analysis**
In FairnessBench we run several analyses on our results. Each `.py` file performs a different analysis and generates plots that are stored in the `graphs/` directory.
To run an analysis, change the input CSV filename in the script to the file required for that analysis.
**Example:** To analyze different types of fairness for the Adult dataset, run `adult_fairness.py`. Before running it, update the script’s input CSV to the new file generated in the `csv_files/` directory.

```python
python ....py
```
**Key files:**
- adult_fairness.py: Analyze the fairness metrics used in the benchmark for the Adult dataset.
- balancing_fairness.py: Analyzes the fairness metrics used in the benchmark for the Balance research problem.
- di_across_datasets.py: Analyzes fairness (disparate impact (DI) and equal opportunity diff (EOB))and accuracy across datasets and research problems.
- comparing_flake8_bal_be_impli.py: Generates a bar plot of Flake8 performance for 3 research problems ( balance, best, implicit) for different models and datasets.
- target_selection.py: Analyze the models performance and fairness metrics for the targey selection tasks.
- target10_sucess_rate.py: Generates a plot showing how agent performance differs from the baseline on the Target10 research problem, by dataset and model.
- acc_di_tradeoff_heatmap.py: Generates a heatmap that shows the tradeoff between accuracy and fairness (disparate impact) for 3 research problems ( balance, best, implicit) using the Pareto frontier.
- acc_fairness_overlap_heatmap.py: Analyses the variation in datasets (randoadult,sampadult, nondescriptive, health) to the adult dataset. 
- run_count.py: Analyzes the number of completed and successful runs for each model and dataset.
- sensitivity_analysis.py: code analyzing the LLM agent's performance and fairness for different versions of the same prompt.
- dollarstreet_analysis.py: codebase for analyzing the performance across income levels.



