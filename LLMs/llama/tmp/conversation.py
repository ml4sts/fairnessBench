import transformers
import torch
import os

print("Imports done!")

# Directory work
current_dir = os.getcwd() #Hold current working directory for logging
print("Your current working directory is {current_dir}\nSaved...")
os.chdir("/")
print("Root dir set up: {os.getcwd()}")


model_id="datasets/ai/llama3/huggingface/llama-3-8b-instruct"

pipeline=transformers.pipeline(
    "text-generation",
    model=model_id,
    model_kwargs={"torch_dtype": torch.bfloat16},
    device_map="auto",)

print("Model setup successful.")
print("Welcome to our research...\n")

user_question = input("Ask a question. or hit enter to skip")
if user_question == "":
    user_question = "How well adjusted are LLMs to real-life social issues and how aware are answers provided by LLMs to social fairness?"

old_answer = ""

messages = [
    # {"role": "system", "content": "You are a pirate chatbot who always responds in pirate speak!"},
    {"role": "system", "content": "You are a research assistant. Helping with social understanding"},
    {"role": "user", "content": "{user_question}"},
]


#Print output to terminal
print(old_answer[0]["generated_text"][-1])