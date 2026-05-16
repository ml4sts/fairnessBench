""" This file contains the code for calling all LLM APIs. """

import os
import torch
import tiktoken
# from .saliency import *
from functools import partial
from .schema import TooLongPromptError, LLMError
from transformers import StoppingCriteria, StoppingCriteriaList
# Adding pipeline and BitsAndBytes to compress the llm
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline, BitsAndBytesConfig

enc = tiktoken.get_encoding("cl100k_base")

torch.cuda.empty_cache()

# Setup llama
loaded_hf_models = {}
def complete_text_hf(prompt, stop_sequences=[], model="llama", max_tokens_to_sample = 2500, temperature=0.5, log_file=None, device=0, **kwargs):
    if model in loaded_hf_models:
        hf_model, tokenizer = loaded_hf_models[model]
    else:
        model = "meta-llama/Llama-3.3-70B-Instruct"
        tokenizer = AutoTokenizer.from_pretrained(model)
        quant_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.float16)
        # Use device_map instead of .to() and make sure to specify "cuda:<#>" not "auto"
        hf_model = AutoModelForCausalLM.from_pretrained(model, quantization_config = quant_config, device_map=f"cuda:{device}",torch_dtype=torch.float16) 
        loaded_hf_models["llama"] = (hf_model, tokenizer)
        print(f"Loaded {model} successfuly using device:{hf_model.device}")

        
    encoded_input = tokenizer(prompt, return_tensors="pt", return_token_type_ids=False).to(f"cuda:{device}")
    # print(encoded_input.keys())
    # encoded_input["input_ids"] = encoded_input["input_ids"].to(hf_model.device)#.to(torch.float32)

    # For saliency score
    """
    encoded_input = tokenizer(prompt)
    input_tokens = encoded_input['input_ids']
    attention_ids = encoded_input['attention_mask']
    base_saliency_matrix, base_embd_matrix = saliency(hf_model, input_tokens, attention_ids)
    # Input x gradient
    base_explanation = input_x_gradient(base_saliency_matrix, base_embd_matrix, normalize=True)
    # Gradient norm
    base_explanation_l1 = l1_grad_norm(base_saliency_matrix, normalize=True)
    """

    stop_sequence_ids = tokenizer(stop_sequences, return_token_type_ids=False, add_special_tokens=False)
    # stop_sequence_ids["input_ids"] = stop_sequence_ids["input_ids"].to(hf_model.device)
    stopping_criteria = StoppingCriteriaList()

    for stop_sequence_input_ids in stop_sequence_ids.input_ids:
        type(stop_sequence_input_ids)
        stopping_criteria.append(StopAtSpecificTokenCriteria(stop_sequence=stop_sequence_input_ids))
    
    output = hf_model.generate(
        **encoded_input,
        temperature=temperature,
        max_new_tokens=max_tokens_to_sample,
        do_sample=True,
        return_dict_in_generate=True,
        output_scores=True,
        stopping_criteria = stopping_criteria,
        **kwargs,
    )
    sequences = output.sequences
    sequences = [sequence[len(encoded_input.input_ids[0]) :] for sequence in sequences]
    all_decoded_text = tokenizer.batch_decode(sequences)
    completion = all_decoded_text[0]
    if log_file is not None:
        log_to_file(log_file, prompt, completion, model, max_tokens_to_sample)
    return completion



# Set up qwen
loaded_qwen_models = {}
def complete_text_qwen(prompt, stop_sequences=[], model="qwen", max_tokens_to_sample = 2500, temperature=0.5, log_file=None, device=0, **kwargs):
    if model in loaded_qwen_models:
        qwen_model, tokenizer = loaded_qwen_models[model]
    else:
        model = "Qwen/Qwen2.5-72B-Instruct"
        quant_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.float16)
        tokenizer = AutoTokenizer.from_pretrained(model)
        qwen_model = AutoModelForCausalLM.from_pretrained(model, quantization_config = quant_config, device_map=f"cuda:{device}",torch_dtype=torch.float16) 
        loaded_qwen_models["qwen"] = (qwen_model, tokenizer)
        print(f"Loaded {model} successfuly using device:{qwen_model.device}")

    message=[
        {"role": "user", "content": prompt}
    ]
    text=tokenizer.apply_chat_template(
        message,
        tokenize=False,
        add_generation_prompt=True
    )
    encoded_input = tokenizer(
        # prompt, 
        [text],
        return_tensors="pt", 
        return_token_type_ids=False).to(f"cuda:{device}")

    stop_sequence_ids = tokenizer(stop_sequences, return_token_type_ids=False, add_special_tokens=False)
    stopping_criteria = StoppingCriteriaList()

    for stop_sequence_input_ids in stop_sequence_ids.input_ids:
        type(stop_sequence_input_ids)
        stopping_criteria.append(StopAtSpecificTokenCriteria(stop_sequence=stop_sequence_input_ids))
    
    output = qwen_model.generate(
        **encoded_input,
        temperature=temperature,
        max_new_tokens=max_tokens_to_sample,
        do_sample=True,
        return_dict_in_generate=True,
        output_scores=True,
        stopping_criteria = stopping_criteria,
        **kwargs,
    )
    sequences = output.sequences
    sequences = [sequence[len(encoded_input.input_ids[0]) :] for sequence in sequences]
    all_decoded_text = tokenizer.batch_decode(sequences, skip_special_tokens=True)
    completion = all_decoded_text[0]
    if log_file is not None:
        log_to_file(log_file, prompt, completion, model, max_tokens_to_sample)
    return completion



# Set up granite
loaded_granite_models = {}
def complete_text_granite(prompt, stop_sequences=[], model="granite", max_tokens_to_sample = 2000, temperature=0.2, log_file=None, device=0, **kwargs):
    if model in loaded_granite_models:
        granite_model, tokenizer = loaded_granite_models[model]
    else:
        model = "ibm-granite/granite-34b-code-instruct-8k"
        quant_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.float16)
        tokenizer = AutoTokenizer.from_pretrained(model)
        granite_model = AutoModelForCausalLM.from_pretrained(model, quantization_config = quant_config, device_map=f"cuda:{device}",torch_dtype=torch.float16)
        granite_model.eval()
        loaded_granite_models["granite"] = (granite_model, tokenizer)
        print(f"Loaded {model} successfuly using device:{granite_model.device}")

    chat = [
        {"role": "user", "content": prompt},
    ]
    chat = tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)

    encoded_input = tokenizer(
        chat, 
        return_tensors="pt").to(f"cuda:{device}")
    
    """
    # transfer tokenized inputs to the device
    for i in encoded_input:
        encoded_input[i] = encoded_input[i].to(f"cuda:{device}")"""

    stop_sequence_ids = tokenizer(stop_sequences, return_token_type_ids=False, add_special_tokens=False)
    stopping_criteria = StoppingCriteriaList()

    for stop_sequence_input_ids in stop_sequence_ids.input_ids:
        type(stop_sequence_input_ids)
        stopping_criteria.append(StopAtSpecificTokenCriteria(stop_sequence=stop_sequence_input_ids))
    
    output = granite_model.generate(
        **encoded_input,
        temperature=temperature,
        max_new_tokens=max_tokens_to_sample,
        do_sample=True,
        return_dict_in_generate=True,
        output_scores=True,
        stopping_criteria = stopping_criteria,
        **kwargs,
    )
    sequences = output.sequences
    sequences = [sequence[len(encoded_input.input_ids[0]) :] for sequence in sequences]
    all_decoded_text = tokenizer.batch_decode(sequences, skip_special_tokens=True)
    completion = all_decoded_text[0]
    if log_file is not None:
        log_to_file(log_file, prompt, completion, model, max_tokens_to_sample)
    return completion




# Set up deepseek
loaded_deepseek_models = {}
def complete_text_deepseek(prompt, stop_sequences=[], model="deepseek", max_tokens_to_sample = 2000, temperature=0.2, log_file=None, device=0, **kwargs):
    if model in loaded_deepseek_models:
        deepseek_model, tokenizer = loaded_deepseek_models[model]
    else:
        model = "deepseek-ai/deepseek-coder-33b-instruct"
        quant_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.float16)
        tokenizer = AutoTokenizer.from_pretrained(model, trust_remote_code=True)
        deepseek_model = AutoModelForCausalLM.from_pretrained(model, trust_remote_code=True, quantization_config = quant_config, device_map=f"cuda:{device}",torch_dtype=torch.float16)
        loaded_deepseek_models["deepseek"] = (deepseek_model, tokenizer)
        print(f"Loaded {model} successfuly using device:{deepseek_model.device}")

    message=[
    { 'role': 'user', 'content': prompt}
    ]

    chat = tokenizer.apply_chat_template(message, tokenize=False, add_generation_prompt=True)
    encoded_input = tokenizer(
        chat, 
        return_tensors="pt").to(f"cuda:{device}")
    

    stop_sequence_ids = tokenizer(stop_sequences, return_token_type_ids=False, add_special_tokens=False)
    stopping_criteria = StoppingCriteriaList()

    for stop_sequence_input_ids in stop_sequence_ids.input_ids:
        type(stop_sequence_input_ids)
        stopping_criteria.append(StopAtSpecificTokenCriteria(stop_sequence=stop_sequence_input_ids))
    
    output = deepseek_model.generate(
        **encoded_input,
        temperature=temperature,
        max_new_tokens=max_tokens_to_sample,
        do_sample=True,
        top_k=50,
        top_p=0.95,
        return_dict_in_generate=True,
        output_scores=True,
        stopping_criteria = stopping_criteria,
        **kwargs
    )
   
    sequences = output.sequences
    sequences = [sequence[len(encoded_input.input_ids[0]) :] for sequence in sequences]
    all_decoded_text = tokenizer.batch_decode(sequences, skip_special_tokens=True)
    completion = all_decoded_text[0]

    if log_file is not None:
        log_to_file(log_file, prompt, completion, model, max_tokens_to_sample)
    return completion



# Set up gemma
loaded_gemma_models = {}
def complete_text_gemma(prompt, stop_sequences=[], model="gemma", max_tokens_to_sample = 2000, temperature=0.2, log_file=None, device=0, **kwargs):
    if model in loaded_gemma_models:
        gemma_model, tokenizer = loaded_gemma_models[model]
    else:
        torch.cuda.empty_cache()

        model = "google/gemma-3-27b-it"
        # quant_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.float16)
        quant_config = BitsAndBytesConfig(load_in_8bit=True)
        tokenizer = AutoTokenizer.from_pretrained(model, trust_remote_code=True)
        gemma_model = AutoModelForCausalLM.from_pretrained(model, quantization_config = quant_config, device_map=f"cuda:{device}",torch_dtype=torch.bfloat16, trust_remote_code=True)
        # gemma_model.eval()
        loaded_gemma_models["gemma"] = (gemma_model, tokenizer)
        print(f"Loaded {model} successfuly using device:{gemma_model.device}")

    chat = [
        {"role": "user", "content": prompt},
    ]
    chat = tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)

    encoded_input = tokenizer(
        chat, 
        return_tensors="pt").to(f"cuda:{device}")
    
    """
    # transfer tokenized inputs to the device
    for i in encoded_input:
        encoded_input[i] = encoded_input[i].to(f"cuda:{device}")"""

    # stop_sequence_ids = tokenizer(stop_sequences, return_token_type_ids=False, add_special_tokens=False)
    # stopping_criteria = StoppingCriteriaList()

    # for stop_sequence_input_ids in stop_sequence_ids.input_ids:
    #     type(stop_sequence_input_ids)
    #     stopping_criteria.append(StopAtSpecificTokenCriteria(stop_sequence=stop_sequence_input_ids))
    

    # print(f"Input text: {chat}\n\n")
    # print(f"Tokenized input IDs: {encoded_input['input_ids']}\n\n")
    # print(f"Max token value: {encoded_input['input_ids'].max()}\n\n")
    # print(f"Min token value: {encoded_input['input_ids'].min()}\n\n")
    # print("Tokenizer vocab size:", tokenizer.vocab_size, "\n\n")

    output = gemma_model.generate(
        **encoded_input,
        temperature=temperature,
        max_new_tokens=max_tokens_to_sample,
        do_sample=True,
        return_dict_in_generate=True,
        output_scores=True,
        stopping_criteria = None,
        # stopping_criteria = stopping_criteria,
        **kwargs,
    )
    sequences = output.sequences
    sequences = [sequence[len(encoded_input.input_ids[0]) :] for sequence in sequences]
    all_decoded_text = tokenizer.batch_decode(sequences, skip_special_tokens=True)
    completion = all_decoded_text[0]
    if log_file is not None:
        log_to_file(log_file, prompt, completion, model, max_tokens_to_sample)
    return completion



# Claude
try:   
    import anthropic
    # setup anthropic API key
    anthropic_client = anthropic.Anthropic(api_key=open("claude_api_key.txt").read().strip())

    def complete_text_claude(prompt, stop_sequences=[anthropic.HUMAN_PROMPT], model="claude-v1", max_tokens_to_sample = 2000, temperature=0.5, log_file=None, messages=None, **kwargs):
        """ Call the Claude API to complete a prompt."""

        ai_prompt = anthropic.AI_PROMPT
        if "ai_prompt" in kwargs is not None:
            ai_prompt = kwargs["ai_prompt"]

        try:
            if model.startswith("claude-3"):
                while True:
                    try:
                        message = anthropic_client.messages.create(
                            messages=[
                                {
                                    "role": "user",
                                    "content": prompt,
                                }
                            ] if messages is None else messages,
                            model=model,
                            stop_sequences=stop_sequences,
                            temperature=temperature,
                            max_tokens=max_tokens_to_sample,
                            **kwargs
                        )
                    except anthropic.InternalServerError as e:
                        pass
                    try:
                        completion = message.content[0].text
                        break
                    except:
                        print("end_turn???")
                        pass
            else:
                rsp = anthropic_client.completions.create(
                    prompt=f"{anthropic.HUMAN_PROMPT} {prompt} {ai_prompt}",
                    stop_sequences=stop_sequences,
                    model=model,
                    temperature=temperature,
                    max_tokens_to_sample=max_tokens_to_sample,
                    **kwargs
                )
                completion = rsp.completion
        except anthropic.APIStatusError as e:
            print(e)
            raise TooLongPromptError()
        except Exception as e:
            raise LLMError(e)

    
        if log_file is not None:
            log_to_file(log_file, prompt, completion, model, max_tokens_to_sample)
        return completion

except Exception as e:
    print(e)
    print("Could not load anthropic API key claude_api_key.txt.")



# CRFM
try:
    from helm.common.authentication import Authentication
    from helm.common.request import Request, RequestResult
    from helm.proxy.accounts import Account
    from helm.proxy.services.remote_service import RemoteService
    # setup CRFM API
    auth = Authentication(api_key=open("crfm_api_key.txt").read().strip())
    service = RemoteService("https://crfm-models.stanford.edu")
    account: Account = service.get_account(auth)


    # CRFM: Looks like this function takes a prompt, uses the auth key that was set  at the top to send the prompt and get the return 
    def get_embedding_crfm(text, model="openai/gpt-4-0314"):
        request = Request(model="openai/text-embedding-ada-002", prompt=text, embedding=True)
        request_result: RequestResult = service.make_request(auth, request)
        return request_result.embedding 


    def complete_text_crfm(prompt="", stop_sequences = [], model="openai/gpt-4-0314",  max_tokens_to_sample=2000, temperature = 0.5, log_file=None, messages = None, **kwargs): 
        random = log_file
        if messages:
            request = Request(
                    prompt=prompt, 
                    messages=messages,
                    model=model, 
                    stop_sequences=stop_sequences,
                    temperature = temperature,
                    max_tokens = max_tokens_to_sample,
                    random = random
                )
        else:
            # print("model", model)
            # print("max_tokens", max_tokens_to_sample)
            request = Request(
                    # model_deployment=model,
                    prompt=prompt, 
                    model=model, 
                    stop_sequences=stop_sequences,
                    temperature = temperature,
                    max_tokens = max_tokens_to_sample,
                    random = random
            )

        try:      
            request_result: RequestResult = service.make_request(auth, request)
        except Exception as e:
            # probably too long prompt
            print(e)
            raise TooLongPromptError()

        if request_result.success == False:
            print(request.error)
            raise LLMError(request.error)
        completion = request_result.completions[0].text
        if log_file is not None:
            log_to_file(log_file, prompt if not messages else str(messages), completion, model, max_tokens_to_sample)
        return completion

except Exception as e:
    print(e)
    print("Could not load CRFM API key crfm_api_key.txt.")





# Setup openai API key and complete text function
try:
    import openai
    # setup OpenAI API key
    # openai.organization, openai.api_key  =  open("openai_api_key.txt").read().strip().split(":")    
    openai.api_key  =  open("openai_api_key.txt").read()   
    # os.environ["OPENAI_API_KEY"] = openai.api_key 


    def complete_text_openai(prompt, stop_sequences=[], model="gpt-4o", max_tokens_to_sample=2000, temperature=0.2, log_file=None, **kwargs):
        """ Call the OpenAI API to complete a prompt."""
        
        # Old code that was in old version of openai
        """ 
        raw_request = {
              "model": model,
              "temperature": temperature,
              "max_tokens": max_tokens_to_sample,
              "stop": stop_sequences or None,  # API doesn't like empty list
              **kwargs
        }
        """
        """
        response = client.chat.completions.create(
          model="gpt-4o-mini",
          messages=[],
          response_format={
            "type": "text"
          },
          temperature=temperature,
          max_completion_tokens=max_tokens_to_sample,
          stop=stop_sequences or None,
          **kwargs
        )
        """

        if model.startswith("gpt-3.5") or model.startswith("gpt-4"):
            response = openai.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                response_format={
                  "type": "text"
                },
                temperature=temperature,
                max_completion_tokens=max_tokens_to_sample,
                stop=stop_sequences or None,
                **kwargs
            )
            completion = response.choices[0].message.content
        else:
            response = openai.chat.completions.create(
                model=model,
                prompt=[{"role": "user", "content": prompt}],
                response_format={
                  "type": "text"
                },
                temperature=temperature,
                max_completion_tokens=max_tokens_to_sample,
                stop=stop_sequences or None,
                **kwargs
            )
            completion = response["choices"][0]["text"]
        if log_file is not None:
            log_to_file(log_file, prompt, completion, model, max_tokens_to_sample)
        return completion

except Exception as e:
    print(e)
    print("Could not load OpenAI API key openai_api_key.txt.")



# gemini
try:
    import vertexai
    from vertexai.preview.generative_models import GenerativeModel, Part
    from google.cloud.aiplatform_v1beta1.types import SafetySetting, HarmCategory
    vertexai.init(project=PROJECT_ID, location="us-central1")
    
    def complete_text_gemini(prompt, stop_sequences=[], model="gemini-pro", max_tokens_to_sample = 2000, temperature=0.5, log_file=None, **kwargs):
        """ Call the gemini API to complete a prompt."""
        # Load the model
        model = GenerativeModel("gemini-pro")
        # Query the model
        parameters = {
                "temperature": temperature,
                "max_output_tokens": max_tokens_to_sample,
                "stop_sequences": stop_sequences,
                **kwargs
            }
        safety_settings = {
                harm_category: SafetySetting.HarmBlockThreshold(SafetySetting.HarmBlockThreshold.BLOCK_NONE)
                for harm_category in iter(HarmCategory)
            }
        safety_settings = {
            }
        response = model.generate_content( [prompt], generation_config=parameters, safety_settings=safety_settings)
        completion = response.text
        if log_file is not None:
            log_to_file(log_file, prompt, completion, model, max_tokens_to_sample)
        return completion

except Exception as e:
    print(e)
    print("Could not load VertexAI API.")




class StopAtSpecificTokenCriteria(StoppingCriteria):
    def __init__(self, stop_sequence):
        super().__init__()
        self.stop_sequence = stop_sequence

    def __call__(self, input_ids, scores, **kwargs):
        # Create a tensor from the stop_sequence
        stop_sequence_tensor = torch.tensor(self.stop_sequence, device=input_ids.device, dtype=input_ids.dtype)
        # Check if the current sequence ends with the stop_sequence
        current_sequence = input_ids[:, -len(self.stop_sequence) :]
        return bool(torch.all(current_sequence == stop_sequence_tensor).item())

    
def log_to_file(log_file, prompt, completion, model, max_tokens_to_sample):
    """ Log the prompt and completion to a file."""
    with open(log_file, "a") as f:
        f.write("\n===================prompt=====================\n")
        f.write(f"{anthropic.HUMAN_PROMPT} {prompt} {anthropic.AI_PROMPT}")
        num_prompt_tokens = len(enc.encode(f"{anthropic.HUMAN_PROMPT} {prompt} {anthropic.AI_PROMPT}"))
        f.write(f"\n==================={model} response ({max_tokens_to_sample})=====================\n")
        f.write(completion)
        num_sample_tokens = len(enc.encode(completion))
        f.write("\n===================tokens=====================\n")
        f.write(f"Number of prompt tokens: {num_prompt_tokens}\n")
        f.write(f"Number of sampled tokens: {num_sample_tokens}\n")
        f.write("\n\n")


# Pick a model from the arg model and call the complete function of that model. We want to use hf (huggingface)
def complete_text(prompt, log_file, model, device=0, **kwargs):
    """ Complete text using the specified model with appropriate API. """
    
    if model.startswith("claude"):
        # use anthropic API
        completion = complete_text_claude(prompt, stop_sequences=[anthropic.HUMAN_PROMPT,"Observation:"], log_file=log_file, model=model, **kwargs)
    elif model.startswith("gemini"):
        completion = complete_text_gemini(prompt, stop_sequences=["Observation:"], log_file=log_file, model=model, **kwargs)
    elif model.startswith("llama"):
        completion = complete_text_hf(prompt, stop_sequences=["Observation:"], log_file=log_file, model=model, device=device, **kwargs)
    elif model.startswith("qwen"):
        completion = complete_text_qwen(prompt, stop_sequences=["Observation:"], log_file=log_file, model=model, device=device, **kwargs)
    elif model.startswith("granite"):
        completion = complete_text_granite(prompt, stop_sequences=["}"], log_file=log_file, model=model, device=device, **kwargs)
    elif model.startswith("deepseek"):
        completion = complete_text_deepseek(prompt, stop_sequences=["}"], log_file=log_file, model=model, device=device, **kwargs)
    elif model.startswith("gemma"):
        completion = complete_text_gemma(prompt, stop_sequences=["}"], log_file=log_file, model=model, device=device, **kwargs)
    elif "/" in model:
        # use CRFM API since this specifies organization like "openai/..."
        completion = complete_text_crfm(prompt, stop_sequences=["Observation:"], log_file=log_file, model=model, **kwargs)
    else:
        # use OpenAI API
        completion = complete_text_openai(prompt, stop_sequences=["Observation:"], log_file=log_file, model=model, **kwargs)
    return completion


# specify fast models for summarization etc. (just the default in case it wasn't passed....)
FAST_MODEL = "gpt-4o-mini"

def complete_text_fast(prompt, device=0, **kwargs):
    return complete_text(prompt = prompt, model = FAST_MODEL, temperature =0.01, device = device, **kwargs)
# complete_text_fast = partial(complete_text_openai, temperature= 0.01)

