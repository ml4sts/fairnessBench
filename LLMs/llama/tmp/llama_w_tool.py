import transformers
import torch
import os

print("Imports done!")

# Directory work
current_dir = os.getcwd() #Hold current working directory for logging
print("Your current working directory is {current_dir}\nSaved...")
os.chdir("/")
print("Root dir set up: {os.getcwd()}")


### Helper functions
import datetime

def current_time():
    """Get the current local time as a string."""
    return str(datetime.now())


def multiply(first: int, second: int) -> int:
    """
    A function that multiplies two numbers
    
    Args:
        first: The first number to multiply
        second: The second number to multiply
    """
    return first*second


# “Tool use” LLMs can choose to call functions as external tools before generating an answer.
# When passing tools to a tool-use model, you can simply pass a list of functions to the tools argument:
from transformers import AutoModelForCausalLM, AutoTokenizer


tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16, device_map="auto")

tools=[multiply]

messages = [
    {"role": "system", "content": "You are a computer"},
    {"role": "user", "content": "Get me the result of 12 by 258 please and thank you"},
]

inputs = tokenizer.apply_chat_template(messages, tools=tools, add_generation_prompt=True, return_dict=True, return_tensors="pt")
inputs = {k: v.to(model.device) for k, v in inputs.items()}
out = model.generate(**inputs, max_new_tokens=128)
print(tokenizer.decode(out[0][len(inputs["input_ids"][0]):]))