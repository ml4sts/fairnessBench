from fairnessBench.LLM import complete_text

 
def read_code(file_path: str):
        with open(file_path, 'r') as file:
             code = file.read()
        return code

def llm_evaluation(code: str, prompt_template: str):
        prompt = prompt_template.replace("[Insert Code Here]", code)
        # print(prompt)
        response = complete_text(prompt, "test.txt", "llama")
        return response
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


 # Hardcoded path to the file containing the code
file_path = '/work/pi_brownsarahm_uri_edu/ayman_uri/fairness/fairnessBench/final_exp_logs/llama/adult/1745027075/env_log/traces/step_final_files/updated_train.py'

# Read the code from the file
code = read_code(file_path)

# Read the prompts from a file (for evaluation)
prompt_file = 'llm_eval_prompt.txt'  # Path to the file containing multiple prompts



# Read the prompts
prompts = read_prompts(prompt_file)

# Loop through all the prompts and evaluate the code with each prompt
for fairness_prompt in prompts:
        result = llm_evaluation(code, fairness_prompt)
        print(f"Prompt============:\n {fairness_prompt}\n\n\n Response===========\n{result}\n\n")
        # print(result)