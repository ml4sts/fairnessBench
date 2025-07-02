from fairnessBench.LLM import complete_text
import re
import json

# Read code from file
# -------------------

def read_file(file_path: str) -> str:
    with open(file_path, 'r') as file:
        return file.read()

# Function to read the prompts from a file
def read_prompts(file_path: str):
        with open(file_path, 'r') as file:
            # 1. Data representation
            # 2. Data collection and preprocessing
            # 3. Model training ( Transparency and explainability)
            # 4. Evaluation and testing ( Disaggregated Evaluation)
            # 5. Evaluation and testing ( Bias Auditing)
            prompts = [prompt.strip() for prompt in file.read().split('---')]  # Split by the delimiter (---)
        return prompts 

def llm_evaluation_by_section(log: str, system_prompt: str, rubric: str, eval_model: str) -> dict:
    """
    Evaluate code by sending one rubric section at a time to the LLM and collecting both scores and justifications.
    """
    rubric_sections = [
        "Model Overview",
        "Stakeholder Identification and Fairness Definition",
        "Data Collection and Processing",
        "Bias Detection and Mitigation",
        "Fairness Metric Selection",
        "Model Selection and Training",
        "Evaluation and Testing"
    ]
    
    results = {
        "scores": {},
        "justifications": {}
    }
    
    for section in rubric_sections:
        prompt = (
            f"""SYSTEM:
{system_prompt}

RUBRIC:
{rubric}

USER LOG:
{log}

Respond ONLY with:
{{
  "{section}": {{
    "score": <int>,
    "justification": <str>
  }}
}}
"""
        )
        response_text = complete_text(prompt, None, eval_model)
        try:
            # Extract JSON from response
            start_idx = response_text.find('{')
            if start_idx >= 0:
                # Count braces to find matching closing brace
                # doing this because i encouted error Extra data: line 7 column 1 (char 341)
                brace_count = 0
                end_idx = -1
                # i am iterating over each character in the response text
                for i in range(start_idx, len(response_text)):
                    # if i find a brace, i am incrementing the brace count
                    if response_text[i] == '{':
                        brace_count += 1
                    # if i find a closing brace, i am decrementing the brace count
                    elif response_text[i] == '}':
                        brace_count -= 1
                        # if the brace count is 0, i am setting the end index to i+1
                        if brace_count == 0:
                            end_idx = i + 1
                            break
                
                json_str = response_text[start_idx:end_idx] if end_idx > start_idx else ""
            else:
                json_str = ""
            
            if json_str:
                response = json.loads(json_str)
                # here  i am extracting the  score and justification for each section from the llm's response 
                if section in response:
                    section_data = response[section]
                    results["scores"][section] = section_data.get("score", 0)
                    results["justifications"][section] = section_data.get("justification", "")
                else:
                    results["scores"][section] = 0
                    results["justifications"][section] = "No justification provided"
            else:
                results["scores"][section] = 0
                results["justifications"][section] = "Error parsing response"
                
        except Exception:
            # Fallback to regex extraction if JSON parsing fails
            # this is for if the llm is not returning a json, i am just using regex to extract the score and justification
            score_match = re.search(r'"score":\s*(\d+)', response_text)
            justification_match = re.search(r'"justification":\s*"([^"\\]*(?:\\.[^"\\]*)*)"', response_text, re.DOTALL)
            results["scores"][section] = int(score_match.group(1)) if score_match else 0
            results["justifications"][section] = justification_match.group(1) if justification_match else ""
    
    return results
    
# Compute weighted subtotals and overall percentage
def compute_llm_score(raw_scores: dict) -> dict:  
    """
    Calculate weighted subtotals and overall percentage from raw rubric scores.

    Args:
        raw_scores (dict): Mapping of criterion names to integer scores (1–4).

    Returns:
        dict: {
            'subtotals': {criterion: subtotal_percentage, ...},
            'total_percentage': overall_percentage
        }
    """
    weights = {
        "Model Overview" : 0.10,
        "Stakeholder Identification and Fairness Definition":0.15,
        "Data Collection and Processing": 0.30,
        "Bias Detection and Mitigation": 0.15,
        "Fairness Metric Selection": 0.10,
        "Model Selection and Training": 0.10,
        "Evaluation and Testing": 0.10
    }
     # here i am collecting each criterion's weighted percentage in this dict.
    subtotals = {}
    # i am iterating over each criterion (eg model overview) and the scores (eg 1-4 from the llm)
    for criterion, score in raw_scores.items():
        # just initializing weights for each criteria, putting a zero so that if by mistake a weight or criterion is missing, it doesn't cause an error.
        weight = weights.get(criterion, 0)
        # so i am taking the score divided by 4 bc 4 is the highest score the llm will attribute to the questions, doing this bc i want the total score to stay in range of 0-100
        subtotal = (score / 4) * weight * 100
        #maybe i will try 1 decimal place later 
        subtotals[criterion] = round(subtotal, 2)

    total_percentage = round(sum(subtotals.values()), 2)
    # i am returning the subtotals (eg, 11%, 12%) like the total for each question in the rubric 
    return {
        'subtotals': subtotals,
        'total_llm_score': total_percentage
    }

def llm_eval_log(file_path="/work/pi_brownsarahm_uri_edu/Ritta_uri/fairnessBench/fairnessBench/main_log", eval_model="granite"):
    # Read the code from the file
    log = read_file(file_path)

    # Read the prompts from a file (for evaluation)
    system_prompt_path = 'system_prompt_log.txt'
    rubric_path = 'log_rubric.txt'
    
    # Read the system_prompts and rubric
    system_prompt = read_prompts(system_prompt_path)[0]  # Assuming the first prompt is what i want
    rubric = read_prompts(rubric_path)[0]  # Assuming the first rubric is what i want

    # Get raw scores and justifications from LLM, section by section
    evaluation_results = llm_evaluation_by_section(log, system_prompt, rubric, eval_model)
    raw_scores = evaluation_results["scores"]
    justifications = evaluation_results["justifications"]
    # Compute weighted scores
    final_scores = compute_llm_score(raw_scores)
    # Combine and return
    return {
        "raw_scores": raw_scores,
        "justifications": justifications,
        **final_scores
    }

def repeat_llm_eval_log(n=5, file_path="/work/pi_brownsarahm_uri_edu/Ritta_uri/fairnessBench/fairnessBench/main_log", eval_model="granite"):
    """
    Run `llm_eval` multiple times and return a list of results.
    """
    results = []
    for i in range(n):
        print(f"Running evaluation {i + 1}...")
        result = llm_eval_log(file_path=file_path, eval_model=eval_model)
        results.append(result)
    return results
