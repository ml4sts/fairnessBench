try:
    import os
except:
    print("Unable to import os.")
    exit()

try:
    import torch
except:
    print("Unable to import torch.")
    exit()

try:
    import transformers
except:
    print("Unable to import transformers.")
    exit()
print("Imports done!")


# Directory set up
current_dir = os.getcwd() #Hold current working directory for logging
print(f"Your current working directory is {current_dir}\nSaved...")
try:
    os.mkdir("log")
    log_dir = f"{current_dir}/log"
except:
    print("Unable to create log directory.")
    exit()
os.chdir("/")
print(f"Root dir set up: {os.getcwd()}")

# Set up LLM
model_id="datasets/ai/llama3/huggingface/llama-3-8b-instruct"

pipeline=transformers.pipeline(
    "text-generation",
    model=model_id,
    model_kwargs={"torch_dtype": torch.bfloat16},
    device_map="auto",)

print("Model setup successful.")
print("Welcome to our research...\n")

# Prepare question
user_question = input("Ask a question. or hit enter to skip")
if user_question == "":
    user_question = "How well adjusted are LLMs to real-life social issues and how aware are answers provided by LLMs to social fairness?"

messages = [
    # {"role": "system", "content": "You are a pirate chatbot who always responds in pirate speak!"},
    {"role": "system", "content": "You are a research assistant. Helping with social understanding"},
    {"role": "user", "content": f"{user_question}"},
]

# Get answer 
old_answer = pipeline(messages,max_new_tokens=512,)

#Print output to terminal
# print(old_answer[0]["generated_text"][-1])



times=0
# Creates a new file to log answers
log_file = open(f"{log_dir}/answers_{times}.txt", 'w')
log_file.write(str(old_answer[0]["generated_text"][-1]))
log_file.close()

while (times<5):
    messages = [
    # {"role": "system", "content": "You are a pirate chatbot who always responds in pirate speak!"},
    {"role": "system", "content": "You are a research assistant. Helping with social understanding"},
    {"role": "user", "content": f"With this question {user_question} I got this answer {old_answer}. See if you can make it better"},
    ]

    old_answer = pipeline(messages,max_new_tokens=512,)

    log_file = open(f"{log_dir}/answers_{times}.txt", 'w')
    log_file.write(str(old_answer[0]["generated_text"][-1]))
    log_file.close()
    times+=1

print("Done!")