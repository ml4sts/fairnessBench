Current errors:

No module named 'helm'
Could not load CRFM API key crfm_api_key.txt.
No module named 'anthropic'
Could not load anthropic API key claude_api_key.txt.
No module named 'openai'
Could not load OpenAI API key openai_api_key.txt.
No module named 'vertexai'
Could not load VertexAI API.
Traceback (most recent call last):
  File "<frozen runpy>", line 198, in _run_module_as_main
  File "<frozen runpy>", line 88, in _run_code
  File "/work/pi_brownsarahm_uri_edu/ayman_uri/fairness/fairnessBench/fairnessBench/runner.py", line 7, in <module>
    from fairnessBench import LLM
  File "/work/pi_brownsarahm_uri_edu/ayman_uri/fairness/fairnessBench/fairnessBench/LLM.py", line 140, in <module>
    def complete_text_claude(prompt, stop_sequences=[anthropic.HUMAN_PROMPT], model="claude-v1", max_tokens_to_sample = 2000, temperature=0.5, log_file=None, messages=None, **kwargs):
                                                     ^^^^^^^^^
NameError: name 'anthropic' is not defined




>>> tokenizer = AutoTokenizer.from_pretrained(model)
You are using the default legacy behaviour of the <class 'transformers.models.llama.tokenization_llama_fast.LlamaTokenizerFast'>. This is expected, and simply means that the `legacy` (previous) behavior will be used so nothing changes for you. If you want to use the new behaviour, set `legacy=False`. This should only be set if you understand what it means, and thoroughly read the reason why this was added as explained in https://github.com/huggingface/transformers/pull/24565 - if you loaded a llama tokenizer from a GGUF file you can ignore this message.
Traceback (most recent call last):
  File "/work/pi_brownsarahm_uri_edu/ayman_uri/fairness/llama_env/lib/python3.12/site-packages/transformers/convert_slow_tokenizer.py", line 1636, in convert_slow_tokenizer
    ).converted()
      ^^^^^^^^^^^
  File "/work/pi_brownsarahm_uri_edu/ayman_uri/fairness/llama_env/lib/python3.12/site-packages/transformers/convert_slow_tokenizer.py", line 1533, in converted
    tokenizer = self.tokenizer()
                ^^^^^^^^^^^^^^^^
  File "/work/pi_brownsarahm_uri_edu/ayman_uri/fairness/llama_env/lib/python3.12/site-packages/transformers/convert_slow_tokenizer.py", line 1526, in tokenizer
    vocab_scores, merges = self.extract_vocab_merges_from_model(self.vocab_file)
                           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/work/pi_brownsarahm_uri_edu/ayman_uri/fairness/llama_env/lib/python3.12/site-packages/transformers/convert_slow_tokenizer.py", line 1502, in extract_vocab_merges_from_model
    bpe_ranks = load_tiktoken_bpe(tiktoken_url)
                ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/work/pi_brownsarahm_uri_edu/ayman_uri/fairness/llama_env/lib/python3.12/site-packages/tiktoken/load.py", line 147, in load_tiktoken_bpe
    for token, rank in (line.split() for line in contents.splitlines() if line)
        ^^^^^^^^^^^
ValueError: not enough values to unpack (expected 2, got 1)

During handling of the above exception, another exception occurred:

Traceback (most recent call last):
  File "/work/pi_brownsarahm_uri_edu/ayman_uri/fairness/llama_env/lib/python3.12/site-packages/transformers/tokenization_utils_base.py", line 2447, in _from_pretrained
    tokenizer = cls(*init_inputs, **init_kwargs)
                ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/work/pi_brownsarahm_uri_edu/ayman_uri/fairness/llama_env/lib/python3.12/site-packages/transformers/models/llama/tokenization_llama_fast.py", line 157, in __init__
    super().__init__(
  File "/work/pi_brownsarahm_uri_edu/ayman_uri/fairness/llama_env/lib/python3.12/site-packages/transformers/tokenization_utils_fast.py", line 138, in __init__
    fast_tokenizer = convert_slow_tokenizer(self, from_tiktoken=True)
                     ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/work/pi_brownsarahm_uri_edu/ayman_uri/fairness/llama_env/lib/python3.12/site-packages/transformers/convert_slow_tokenizer.py", line 1638, in convert_slow_tokenizer
    raise ValueError(
ValueError: Converting from Tiktoken failed, if a converter for SentencePiece is available, provide a model path with a SentencePiece tokenizer.model file.Currently available slow->fast convertors: ['AlbertTokenizer', 'BartTokenizer', 'BarthezTokenizer', 'BertTokenizer', 'BigBirdTokenizer', 'BlenderbotTokenizer', 'CamembertTokenizer', 'CLIPTokenizer', 'CodeGenTokenizer', 'ConvBertTokenizer', 'DebertaTokenizer', 'DebertaV2Tokenizer', 'DistilBertTokenizer', 'DPRReaderTokenizer', 'DPRQuestionEncoderTokenizer', 'DPRContextEncoderTokenizer', 'ElectraTokenizer', 'FNetTokenizer', 'FunnelTokenizer', 'GPT2Tokenizer', 'HerbertTokenizer', 'LayoutLMTokenizer', 'LayoutLMv2Tokenizer', 'LayoutLMv3Tokenizer', 'LayoutXLMTokenizer', 'LongformerTokenizer', 'LEDTokenizer', 'LxmertTokenizer', 'MarkupLMTokenizer', 'MBartTokenizer', 'MBart50Tokenizer', 'MPNetTokenizer', 'MobileBertTokenizer', 'MvpTokenizer', 'NllbTokenizer', 'OpenAIGPTTokenizer', 'PegasusTokenizer', 'Qwen2Tokenizer', 'RealmTokenizer', 'ReformerTokenizer', 'RemBertTokenizer', 'RetriBertTokenizer', 'RobertaTokenizer', 'RoFormerTokenizer', 'SeamlessM4TTokenizer', 'SqueezeBertTokenizer', 'T5Tokenizer', 'UdopTokenizer', 'WhisperTokenizer', 'XLMRobertaTokenizer', 'XLNetTokenizer', 'SplinterTokenizer', 'XGLMTokenizer', 'LlamaTokenizer', 'CodeLlamaTokenizer', 'GemmaTokenizer', 'Phi3Tokenizer']


# AS: Fixed!
During handling of the above exception, another exception occurred:

Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
  File "/work/pi_brownsarahm_uri_edu/ayman_uri/fairness/llama_env/lib/python3.12/site-packages/transformers/models/auto/tokenization_auto.py", line 939, in from_pretrained
    return tokenizer_class_fast.from_pretrained(pretrained_model_name_or_path, *inputs, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/work/pi_brownsarahm_uri_edu/ayman_uri/fairness/llama_env/lib/python3.12/site-packages/transformers/tokenization_utils_base.py", line 2213, in from_pretrained
    return cls._from_pretrained(
           ^^^^^^^^^^^^^^^^^^^^^
  File "/work/pi_brownsarahm_uri_edu/ayman_uri/fairness/llama_env/lib/python3.12/site-packages/transformers/tokenization_utils_base.py", line 2448, in _from_pretrained
    except import_protobuf_decode_error():
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/work/pi_brownsarahm_uri_edu/ayman_uri/fairness/llama_env/lib/python3.12/site-packages/transformers/tokenization_utils_base.py", line 87, in import_protobuf_decode_error
    raise ImportError(PROTOBUF_IMPORT_ERROR.format(error_message))
ImportError: 
 requires the protobuf library but it was not found in your environment. Checkout the instructions on the
installation page of its repo: https://github.com/protocolbuffers/protobuf/tree/master/python#installation and follow the ones
that match your environment. Please note that you may need to restart your runtime after installation.