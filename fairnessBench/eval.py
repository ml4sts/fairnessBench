import os
import anthropic
from pathlib import Path
import re
import io
import sys
import json
import dataclasses
from dataclasses import dataclass
from typing import List, Dict
from importlib import util
import argparse
import importlib 
import matplotlib.pyplot as plt

# from .LLM import complete_text_gpt4, complete_text_claude
# from .environment import get_task_info
from .prepare_task import get_task_info
from .llm_eval import llm_eval
from flake8.main import application
app = application.Application()




class EnhancedJSONEncoder(json.JSONEncoder):
    def default(self, o):
        if dataclasses.is_dataclass(o):
            return dataclasses.asdict(o)
        #if it is a function, use its string name
        elif hasattr(o, '__call__'):
            return o.__name__

        return super().default(o)

def oom_error(path):
    log = path.replace("trace.json", "../log")
    main_log = path.replace("trace.json", "../agent_log/main_log")
    message = "CUDA out of memory"
    return (message in open(log, "r").read()) or (message in open(main_log, "r").read())
    

def connection_error(path):
    log = path.replace("trace.json", "../log")
    main_log = path.replace("trace.json", "../agent_log/main_log")
    bad = ["You exceeded your current quota, please check your plan and billing details.", "Error: 'text-similarity-ada-001'", "Error: 'text-embedding-ada-001'"]
    return ("Connection aborted" in open(log, "r").read()) or (any([b in open(main_log, "r").read() for b in bad])) 

def error(path):
    return os.path.exists(os.path.join(path.replace("trace.json", ""), "error.txt")) or not os.path.exists(os.path.join(path.replace("trace.json", ""), "overall_time.txt"))


def json_error(path):
    main_log = path.replace("trace.json", "../agent_log/main_log")
    return open(main_log, "r").read().count("JSONDecodeError") > 2

def long_prompt_error(path):
    main_log = path.replace("trace.json", "../agent_log/main_log")
    return "EnvError: too long input for the tool" in open(main_log, "r").read()


# AS: TODO: check if calling this function multiple times with different scripts affects the app or is initialize enough
def get_flake8(train_script):
    app.initialize([train_script])
    # Step 1: Save the real stdout
    original_stdout = sys.stdout
    # Step 2: Redirect stdout to a string buffer
    buffer = io.StringIO()
    sys.stdout = buffer
    # Step 3: Call the function that prints
    app.run_checks()
    # Step 4: Restore original stdout
    sys.stdout = original_stdout
    # Step 5: Get the output as a string
    flake8_score = buffer.getvalue().split(': ')[-1].strip()
    return flake8_score

@dataclass
class EvaluationResult:
    path: str
    summary: str
    rubric_questions: Dict[str, str]
    score: List[float]
    llm_score: List[List[str]]
    flake8_score: List[str]
    score_steps: List[float]
    submitted_final_answer: bool
    final_score: float
    final_llm_score: List[str]
    final_flake8_score: str
    total_time: float
    error: str
    extra: Dict[str, bool]


def run_eval(log_folder, benchmark_folder_name, eval_model = "qwen", eval_intermediate=False):
    results = {}    

    # Log folder is the specific log folder for one model and one task
    # Navigate the lof folder for all run logs in it
    for subdir, dirs, files in os.walk(log_folder):
        for file in files:

            if file == 'trace.json':
                result = EvaluationResult(
                    path=os.path.join(subdir, file),
                    summary="",
                    rubric_questions={},
                    score=[],
                    llm_score=[],
                    flake8_score=[],
                    score_steps=[],
                    final_score = -1,
                    final_llm_score=[],
                    final_flake8_score = "",
                    submitted_final_answer = False,
                    total_time = 0,
                    error = "",
                    extra = {}
                )
                try:
                    with open(os.path.join(subdir, file)) as f:
                        data = json.load(f)
                except:
                    continue
                num_steps = len(data['steps'])

                # Check if agent was able to complete the task and submit a final step
                for step in range(len(data['steps'])):
                    if data['steps'][step]["action"]["name"] == "Final Answer":
                        result.submitted_final_answer = True
                        
                # Setting a constant var to limit the evaluation to a maximum number of steps
                num_steps_eval = 50 # AS: move this to the top with eplaination

                # Create a sampled list of steps to limit the evaluation to the maximum number of steps
                step_list = range(num_steps)
                if num_steps_eval >= len(step_list):
                    subsampled_list = step_list
                else:
                    step = num_steps // num_steps_eval
                    subsampled_list = step_list[::step][:num_steps_eval]
                
                # Setup the get_score function from the task's eval.py
                module = importlib.import_module(f'fairnessBench.benchmarks.{benchmark_folder_name}.scripts.eval')
                

                # Having counted the steps the agent took and sampled them
                # We can either evaluate seach step or directly evaluate final step
                if eval_intermediate:
                    for step in subsampled_list:
                        eval_step_score = 0
                        try:
                            folder_path = os.path.join(subdir, f'traces/step_{step}_files')
                            if os.path.exists(folder_path):
                                print(folder_path)
                                # module = importlib.import_module(f'fairnessBench.benchmarks.{benchmark_folder_name}.scripts.eval')
                                eval_step_score = module.get_score(folder_path)
                                result.score.append(eval_step_score)
                                # AS: Attempting to get llm_eval here
                                try:
                                    train_script = os.path.join(folder_path, "train.py")
                                    llm_score = llm_eval(train_script, eval_model)
                                    result.llm_score.append(llm_score)
                                    flake8_score = get_flake8(train_script)
                                    result.flake8_score.append(flake8_score)
                                except Exception as e:
                                    print(e)
                                    pass
                                    
                        except Exception as e:
                            print(e)
                            result.score.append(eval_step_score)
                    # Add the ids of the steps that were evaluated to the JSON file
                    result.score_steps = list(subsampled_list)
                
                # Evaluate the final step
                folder_path = os.path.join(subdir, 'traces/step_final_files')
                try:
                    if os.path.exists(folder_path):
                        # module = importlib.import_module(f'fairnessBench.benchmarks.{benchmark_folder_name}.scripts.eval')
                        eval_final_score = module.get_score(folder_path)
                        result.score.append(eval_final_score)
                        result.final_score = eval_final_score
                        # AS: Attempting to get llm_eval here
                        train_script = os.path.join(folder_path, "train.py")
                        try:
                            llm_score = llm_eval(train_script, eval_model)
                            result.final_llm_score = llm_score
                        except Exception as e:
                            print("\nllm_eval didn't work\n")
                            # print(e)
                            pass
                        # AS: Attempting to get Flake8 score here
                        try:
                            flake8_score = get_flake8(train_script)
                            result.final_flake8_score = flake8_score
                            print(flake8_score)
                        except Exception as e:
                            print("\nFlake8_eval didn't work\n")
                            # print(e)
                            pass
                        print(eval_final_score)
                except Exception as e:
                    print(e)
                    pass
                
                # If environment error occurred we log it in the result JSON
                if os.path.exists(os.path.join(subdir, "error.txt")):
                    result.error = open(os.path.join(subdir, "error.txt")).read()
                
                # Log overall time in the result JSON
                if os.path.exists(os.path.join(subdir, "overall_time.txt")):
                    result.total_time = float(open(os.path.join(subdir, "overall_time.txt")).read())
                    print(result.total_time)
                
                result.extra = {
                    "oom_error": oom_error(os.path.join(subdir, file)),
                    "connection_error": connection_error(os.path.join(subdir, file)),
                    "error": error(os.path.join(subdir, file)),
                    "json_error": json_error(os.path.join(subdir, file)),
                    "long_prompt_error": long_prompt_error(os.path.join(subdir, file)),
                }
                    
                results[os.path.join(subdir, file)] = result
                    
        
    return results
            
if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--log-folder", type=str, default="logs")
    parser.add_argument("--task", type=str, default="adult")
    parser.add_argument("--output-file", type=str, default="results.json")
    parser.add_argument("--eval_model", type=str, default="qwen")
    parser.add_argument("--eval-intermediate", action="store_true")
    args = parser.parse_args()
    
    benchmark_folder_name = get_task_info(args.task)[0] 
    results = run_eval(args.log_folder, benchmark_folder_name, eval_model = args.eval_model, eval_intermediate = args.eval_intermediate)
              
    json.dump(results, open(args.output_file, "w"), indent=4, cls=EnhancedJSONEncoder)
                
       
