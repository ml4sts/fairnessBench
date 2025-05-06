from fairnessBench.LLM import complete_text
import re
import json
 
def read_code(file_path: str):
        with open(file_path, 'r') as file:
             code = file.read()
        return code

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

def llm_evaluation(code: str, system_prompt: str, rubric: str, eval_model: str)-> dict:
        prompt = (
        f"""SYSTEM:
{system_prompt}

RUBRIC:
{rubric}

USER CODE:
```python
{code}
```

Respond ONLY with:
{{
  "scores": {{
    "Model Overview": <int>,
    "Stakeholder Identification and Fairness Definition": <int>,
    "Data Collection and Processing": <int>,
    "Bias Detection and Mitigation": <int>,
    "Fairness Metric Selection": <int>,
    "Model Selection and Training": <int>,
    "Evaluation and Testing": <int>
  }}
}}
"""
    )
        response_text=complete_text(prompt, "test.txt", eval_model)
        try:
            # Try to find and extract just the JSON part
                start_idx = response_text.find('{')
                end_idx = response_text.rfind('}') + 1
        
                if start_idx >= 0 and end_idx > start_idx:
                        cleaned_response = response_text[start_idx:end_idx]
                        #print(f"Cleaned JSON: {cleaned_response}")
                        response = json.loads(cleaned_response)
                        #print(f"Parsed response: {response}")
                        return response
                else:
                        print(f"Could not find valid JSON in response: {response_text}")
                        return {"scores": []}
        except Exception as e:
                print(f"Unexpected error: {e}")
            # Fallback: zero scores for all criteria
                return {"scores": {k: 0 for k in [
                    "Model Overview",
                    "Stakeholder Identification and Fairness Definition",
                    "Data Collection and Processing",
                    "Bias Detection and Mitigation",
                    "Fairness Metric Selection",
                    "Model Selection and Training",
                    "Evaluation and Testing"
                ]}}

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
        "Model Overview": 0.10, # I turned them to fractions or decimal to better calculate this is 10%
        "Stakeholder Identification and Fairness Definition": 0.15,
        "Data Collection and Processing": 0.30,
        "Bias Detection and Mitigation": 0.15,
        "Fairness Metric Selection": 0.10,
        "Model Selection and Training": 0.10,
        "Evaluation and Testing": 0.10
    }
     # here i am  collecting each criterion’s weighted percentage in this dict.
    subtotals = {}
    # i am iterating over each criterion( eg model overview) and the  scores( eg 1-4 from the llm)
    for criterion, score in raw_scores.items():
        # just intitializing weights for each criteria , puting a zero so that if by mistake a weight or criterion is missing, it doesn't cause an error.
        weight = weights.get(criterion, 0)
        # so i ma taking the score divided by 4 bc 4 is the heighest score the llm will attribute to the questions, doing this bc i want the total score to stay in range of 0-100
        subtotal = (score / 4) * weight * 100
        #maybe i will try 1 decimal place later 
        subtotals[criterion] = round(subtotal, 2)

    total_percentage = round(sum(subtotals.values()), 2)
    # i am returning the subtotals (eg, 11%, 12% ) like the total for each question in the rubric 
    return {
        'subtotals': subtotals,
        'total_llm_score': total_percentage
    }

def llm_eval(file_path= "/work/pi_brownsarahm_uri_edu/Ritta_uri/fairnessBench/train.py",eval_model="llama"):
    # Hardcoded path to the file containing the code
    file_path = '/work/pi_brownsarahm_uri_edu/Ritta_uri/fairnessBench/train.py'

    # Read the code from the file
    code = read_code(file_path)

    # Read the prompts from a file (for evaluation)
    system_prompt_path = '/work/pi_brownsarahm_uri_edu/Ritta_uri/fairnessBench/system_prompt.txt'  # Path to the file containing multiple prompts
    rubric_path="/work/pi_brownsarahm_uri_edu/Ritta_uri/fairnessBench/rubric.txt"
    # Read the syetem_prompts and rubric
    system_prompt = read_prompts(system_prompt_path)
    rubric= read_prompts(rubric_path)

    # Get raw scores from LLM
    response = llm_evaluation(code, system_prompt, rubric, eval_model)
    raw_scores = response.get("scores", {})

    # Compute weighted scores
    final_scores = compute_llm_score(raw_scores)

    # Combine and return
    return {
        "raw_scores": raw_scores,
        **final_scores
    }
output = llm_eval(
        file_path="/work/pi_brownsarahm_uri_edu/Ritta_uri/fairnessBench/train.py",
        eval_model="llama"
)
print(output)
