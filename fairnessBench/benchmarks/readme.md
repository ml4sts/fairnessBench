
# Benchmarks
This folder contains benchmark configurations for experiments.

## Naming Scheme

Each benchmark follows a structured naming convention to make experiments easy to interpret and reproduce.
We organize the benchmark as:
One folder per (dataset × fairness metric x demographic attribute)

**What’s different across folders?**
Each folder represents a distinct evaluation task defined by:
A specific dataset (e.g., adult, german, etc)
A fairness metric (e.g., statistical parity, equal opportunity)
A protected attribute (e.g., race, gender)

So even if two folders use the same dataset, they differ in:
- the fairness metric being evaluated, or
- the protected group under consideration

**Each folder = one concrete benchmark task.**

We aim to cover a diverse set of benchmark tasks spanning multiple datasets, fairness definitions, and evaluation settings.
**Format:**
```
<dataset>-<fair_metric>-<protected_attribute>
```

**Components:**
* `<dataset>`: Source dataset (e.g., `adult`, `dollarstreet`, `non_descriptive_ad`)
* `<fair_metric>`: Group fairness metric used in the research problem (e.g., `di`, `eod`, `err`)
* `<protected_attribute>`: Protected attribute (e.g., `sex`, `race`, `patt1`)

**Example:**
```
adult-di-sex
```

## Datasets

| Dataset Name   | Description   |
| -------------- | ---------------------------------------------------------------------------------------------------|
| Adult       | demographic, career, and tax features to predict income $>=$\$50k            |
| Randoadult  | oversampled  Adult to balance demographics w/matched feature  correlations   |
| Sampadult | synthetic dataset for income prediction, generated with balanced binary classes. 2 are drawn from a standard normal distribution with mean 0 and variance 1. 2 are conditionally dependent on income and are generated as target plus gaussian noise with mean 0 and variance 0.1 The protected attributes include sex, race, marital\_status, and age, mathed to Adult.   |
| Non-descriptive Adult | Adult data with nondescriptive feature names   |
| Health Adult | Adult data modified to have healthcare feature names, with scaling and categorical label mapping.   |
| Health new | ......... |
| German | credit risk assessments labeled as good or bad.   |
| Credit Default | financial and demographic features to predict default status(binary)   |
| Adult reconstruction | Adult dataset reconstructed from Census Data, with original income in dollars instead of a binarizing   |
| Dollarstreet | images of household objects with labels for object type (60 classes), country and household income level   |


## Fair Metrics
We use the following group fairness metrics to capture disparities, assess differences in true positive rates, to quantify misclassification disparities and to examine disparities in false negatives across groups:

### Independence 
Measures whether the prediction and demographic group are independent.
* **Disparate Impact**
* **Statistical Parity Difference**
* **Error Rate Difference**
* **Error Rate Ratio**

### Separation
separation measures if the prediction & demographics are independent conditioned on the ground truth.
* **Equal Opportunity Difference** 

### Sufficiency
Sufficiency measures if the ground truth is independent of the demographic variables, conditioned on the prediction
* **False Omission Rate Difference**


## Reproducibility
For Reproducibility, identify the relevant task folder(s) mentioned in the paper.
Each folder contains:
env\ : containing environment files which the agent has access to
- data_description.txt
- train.py baseline script
scripts\
- prepare.py file, dataset and preprocessing steps 
-  eval.py valuation script
-  read_only_files.txt, 

Compare outputs with reported metrics.

## Running the Benchmark on Your Model / Agent
The benchmark is model-agnostic, any model/agent that produces predictions can be evaluated.

## Extending the Benchmark (Adding New Tasks)
If you want to add new evaluations or tasks, follow this structure:
1. Create a new folder
2. Use the naming convention:
<dataset>_<fairness_metric>_<protected_attribute>
3. Define the task
Include:
Dataset - loading/pre-processing script (prepare.py)
Protected attribute specification
Prediction task definition
Research problem specified in task.json file


