from fairnessBench.LLM import complete_text
import json
 
def read_code(file_path: str):
        with open(file_path, 'r') as file:
             code = file.read()
        return code

def read_prompts(file_path: str):
        with open(file_path, 'r') as file:
            # 1. Data representation
            # 2. Data collection and preprocessing
            # 3. Model training ( Transparency and explainability)
            # 4. Evaluation and testing ( Disaggregated Evaluation)
            # 5. Evaluation and testing ( Bias Auditing)
            prompts = [prompt.strip() for prompt in file.read().split('---')]  # Split by the delimiter (---)
        return prompts 
def compute_llm_score(raw_scores: list[float], section: str) -> dict:
        """
        raw_scores: list of floats (each 0–10)
        section: one of "section1", "section2", "section3", "section4", "section5"

        Returns: 
                {
                "score": int,           # average raw score (0–10), rounded
                "total_score": "X/Y"    # Y is the max for this section lets say 15
                }
        """
        config = {
                "section1":{"checks":3, "scale":15},
                "section2":{"checks":3, "scale":15},
                "section3":{"checks":5, "scale":20},
                "section4":{"checks":5, "scale":10},
                "section5":{"checks":5, "scale":10}
        }
        if section not in config:
                raise ValueError(f"Invalid section: {section}")
        if not raw_scores:
               raise ValueError("raw_scores is empty.")
        
        avg_score = sum(raw_scores) / len(raw_scores)
        scaled_score = avg_score / 10
        rounded_score = round(avg_score)
        
        # Calculate the total score as a fraction
        max_score = config[section]["scale"]
        total_score = f"{round(scaled_score * max_score, 1)}/{max_score}"
        
        return {
                "score": rounded_score,
                "total_score": total_score
        }

def llm_evaluation(code: str, prompt_template: str, eval_model: str) -> dict:
        prompt = prompt_template.replace("[Insert Code Here]", code) + """
Respond ONLY with:
{
  "raw_scores": [<number>, ...]
}
"""
        response_text = complete_text(prompt, "test.txt", eval_model)
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
                        return {"raw_scores": []}
        except json.JSONDecodeError as e:
                #print(f"Parsing error: {e}\nResponse was:\n{response_text}")
                return {"raw_scores": []}
        except Exception as e:
                print(f"Unexpected error: {e}")
                return {"raw_scores": []}
        

        

def llm_eval(file_path= "/work/pi_brownsarahm_uri_edu/Ritta_uri/fairnessBench/train.py",eval_model="llama"  ):  # AS: Remove path when done testing
        # Hardcoded path to the file containing the code
        # file_path = '/work/pi_brownsarahm_uri_edu/ayman_uri/fairness/fairnessBench'
        
        # Read the code from the file
        code = read_code(file_path)
        
        # Read the prompts from a file (for evaluation)
        prompt_file = '/work/pi_brownsarahm_uri_edu/Ritta_uri/fairnessBench/llm_eval_prompt.txt'  # Path to the file containing multiple prompts
        
        
        
        # Read the prompts
        prompts = read_prompts(prompt_file)
        
        results = []
        for idx, prompt in enumerate(prompts, 1):
                # 1) get raw scores list from the LLM
                resp = llm_evaluation(code, prompt, eval_model)
                if "raw_scores" not in resp or not resp["raw_scores"]:
                        print(f"Warning: Empty or missing raw_scores for section{idx}")
                        raw = [0]  # Default value to avoid breaking computation
                else:
                     raw = resp["raw_scores"]


                # 2) compute the final score for this section
                section = f"section{idx}"
                final = compute_llm_score(raw, section)

                results.append({
                "section": section,
                "raw_scores": raw,
                **final
                })
        return results
        # Loop through all the prompts and evaluate the code with each prompt

output = llm_eval(
        file_path="/work/pi_brownsarahm_uri_edu/Ritta_uri/fairnessBench/train.py",
        eval_model="llama"
)
for r in output:
        print(r)
