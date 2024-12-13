# dependencies.py
# Import dependencies.py

# Import required dependencies
import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer, pipeline
from langchain import LLMChain, PromptTemplate
from langchain.llms import HuggingFacePipeline
print("Imports were made.")


# Move to root directory to be able to find llama API
os.chdir("/")
dir= os.getcwd()
print("CWD: {dir}\n")

# Set up variable helpers 
llama = "datasets/ai/llama3/huggingface/llama-3-8b-instruct"
tokenizer = AutoTokenizer.from_pretrained(llama)

model = AutoModelForCausalLM.from_pretrained(llama, device_map='auto',load_in_4bit = True)
print("1")
model.device
print("2")
model.hf_device_map
# !nvidia-smi ??

messages = [
    {"role": "user", "content": "Who are you?"},
]
# Use a pipeline as a high-level helper: https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct?library=transformers
pipe = pipeline("text-generation", model = model, tokenizer = tokenizer, max_length = 500)

# TODO: ask user to input a question, put it into the prompt and sent it
# userQuestion = input("Type a question you want to send to llama")
#prompt = PromptTemplate(template = prompt_template, input_variable = ['question'])

local_llm = HuggingFacePipeline(pipeline=pipe)
llm_chain = LLMChain(prompt=prompt, llm=local_llm)


#llm_output = llm_chain.invoke({'question':{userQuestion}})


