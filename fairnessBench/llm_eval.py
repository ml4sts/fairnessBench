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
        
        avg_score = sum(raw_scores) / (len(raw_scores)) * 10
        rounded_score = round(avg_score)
        
        # Calculate the total score as a fraction
        max_score = config[section]["scale"]
        total_score = round(avg_score * max_score,1)
        
        return {
                "score": rounded_score,
                "total_score": total_score
        }

def llm_evaluation(code: str, prompt_template: str, eval_model="qwen") -> dict:
        prompt = prompt_template.replace("[Insert Code Here]", code) + """
Respond ONLY with:
{
  "raw_scores": [<number>, ...]
}
"""
        # print(prompt)
        response_text = complete_text(prompt, "test.txt", eval_model)
        return json.loads(response_text)
        #response = complete_text(prompt, "test.txt", "llama")
        #return response
        # response = openai.chat.completions.create(
        #     model="gpt-4.1",  
        #     messages=[{
        #        "role": "system", 
        #        "content": "You are an assistant that evaluates code based on fairness rubric."
        #     }, {
        #        "role": "user", 
        #        "content": prompt
        #     }],
        #     max_tokens=500,  
        #     temperature=0.7,  
        # )
    
        # return response.choices[0].message.content



def llm_eval(file_path = " ../final_exp_logs/llama/adult/1745027075/env_log/traces/step_final_files/updated_train.py",eval_model: str = "qwen"):
        # Hardcoded path to the file containing the code
        # file_path = '/work/pi_brownsarahm_uri_edu/ayman_uri/fairness/fairnessBench'
        
        # Read the code from the file
        code = read_code(file_path)
        
        # Read the prompts from a file (for evaluation)
        prompt_file = 'llm_eval_prompt.txt'  # Path to the file containing multiple prompts
        
        
        
        # Read the prompts
        prompts = read_prompts(prompt_file)
        
        results = []
        for idx, prompt in enumerate(prompts, 1):
                # 1) get raw scores list from the LLM
                resp = llm_evaluation(code, prompt, eval_model)
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

if __name__ == "__main__":
    output = llm_eval(
        file_path="updated_train.py",
        prompt_file="llm_eval_prompt.txt",
        eval_model="qwen"
    )
    return output
    