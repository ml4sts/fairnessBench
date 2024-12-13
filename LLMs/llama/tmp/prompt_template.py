# This is the prompt template to be sent to llama with a specific question
# TODO: learn why this prompt specifically and learn how to improve on it

prompt_template = """###Instruction:
You are a candidate appearing for a job.
###Instruction: Look at the job description below and be prepared for the interview.

###Instruction: Read the question being asked and provide an appropriate response for the question that would please the interviewer.
Your response should be in a paragraph format. Avoid using bullet points as the answer is going to be read out.

###Interview:
Question : {question}
###Now Do the following:

1) Read the question properly and frame an answer.

Format for your output:
{{"answer":string}}

Now answer the question: {question}"""

print("Prompt template has been setup!\n")

