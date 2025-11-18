#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Imports

import os
import re

os.environ["TRANSFORMERS_CACHE"] = "/hpc/home/bfa6/work/llms/.cache"
os.environ["HF_HOME"] = "/hpc/home/bfa6/work/llms/.cache"

import time
import json

from unsloth import FastLanguageModel
import torch
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from dotenv import dotenv_values
from tqdm import tqdm
import asyncio
from tqdm.asyncio import tqdm_asyncio
from google import genai
import random
from datasets import Dataset
from trl import SFTTrainer, SFTConfig
from transformers import TextStreamer
import vllm
import nltk

config = dotenv_values("../.env")


# # Load the model

# In[2]:


max_seq_length = 2048 # Can increase for longer reasoning traces
lora_rank = 32 # Larger rank = smarter, but slower

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "unsloth/Qwen3-0.6B",
    cache_dir="/hpc/home/bfa6/work/llms/.cache",
    max_seq_length = max_seq_length,
    load_in_4bit = False, # False for LoRA 16bit
    fast_inference = True, # Enable vLLM fast inference
    max_lora_rank = lora_rank,
    gpu_memory_utilization = 0.6, # Reduce if out of memory
    # force_download=True,
)

model = FastLanguageModel.get_peft_model(
    model,
    r = lora_rank, # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
    target_modules = [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ],
    lora_alpha = lora_rank*2, # *2 speeds up training
    use_gradient_checkpointing = "unsloth", # Reduces memory usage
    random_state = 3407,
)


# # Create a chat template

# In[3]:


system_prompt = (
    "You are given some context.\n"
    "Your goal is to compress the information in the context and output the compressed version.\n"
    "Do not be constrained by language. Feel free to mix and match whatever helps you use less tokens.\n"
    "Use as few tokens as possible while keeping all information.\n"
    "Do NOT produce internal chain-of-thought or step-by-step reasoning.\n"
    "Start immediately with the compressed content (no extra preface)."
)


# In[4]:


# chat_template = \
#     "{% if messages[0]['role'] == 'system' %}"\
#         "{{ messages[0]['content'] + eos_token }}"\
#         "{% set loop_messages = messages[1:] %}"\
#     "{% else %}"\
#         "{{ '{system_prompt}' + eos_token }}"\
#         "{% set loop_messages = messages %}"\
#     "{% endif %}"\
#     "{% for message in loop_messages %}"\
#         "{% if message['role'] == 'user' %}"\
#             "{{ '<|user|>\\n' + message['content'] + eos_token }}"\
#         "{% elif message['role'] == 'assistant' %}"\
#             "{{ '<|assistant|>\\n' + message['content'] + eos_token }}"\
#         "{% endif %}"\
#     "{% endfor %}"

# # Replace with out specific template:
# chat_template = chat_template\
#     .replace("'{system_prompt}'",   f"'{system_prompt}'")

# from unsloth.chat_templates import qwen3_template
# tokenizer.chat_template = qwen3_template

# tokenizer.chat_template


# # Prepare dataset

# In[5]:


# load dataset
with open("/hpc/home/bfa6/work/github/yapper/dataset/chunks.json", "r") as f:
    dataset =  json.load(f)


# In[6]:

### Clean dataset
def clean_whitespace(text: str) -> str:
    # Collapse multiple spaces/tabs into a single space
    text = re.sub(r'[ \t]+', ' ', text)
    # Collapse multiple newlines into a single newline
    text = re.sub(r'\n+', '\n', text)
    # Strip leading/trailing whitespace
    return text.strip()


for data in tqdm(dataset):
    data["chunk"] = clean_whitespace(data["chunk"])


def get_text_len(text: str):
    inp = [{"role":"system", "content": system_prompt}, {"role":"user", "content": text}]
    text = tokenizer.apply_chat_template(
        inp,
        tokenize = False,
        add_generation_prompt=True,
        enable_thinking=False     
    )

    inputs = tokenizer(text, return_tensors = "pt").to("cuda")
    return len(inputs["input_ids"][0])


# Check token lengths
lens = []

for data in tqdm(dataset):
    length = get_text_len(data["chunk"])

    lens.append({"chunk": data["chunk"], "length": length})


df = pd.DataFrame.from_dict(lens)

df = df[df["length"] <= 1000]

cleaned_dataset = [{"chunk": chunk} for chunk in df["chunk"].to_list()]
random.Random(42).shuffle(cleaned_dataset)  # reproducible shuffle

n_total = len(cleaned_dataset)

# 80:10:10 split
n_train = int(0.8 * n_total)
n_eval  = int(0.1 * n_total)
n_test  = n_total - n_train - n_eval  # ensures all items are used

train_dataset = cleaned_dataset[:n_train]
eval_dataset  = cleaned_dataset[n_train:n_train + n_eval]
test_dataset  = cleaned_dataset[n_train + n_eval:]

print(len(train_dataset), len(eval_dataset), len(test_dataset))
print(f"Total: {n_total}")


# In[7]:


test = [{"role":"system", "content": system_prompt}, {"role":"user", "content": train_dataset[0]["chunk"]}]


# In[8]:


# train_dataset[0]


# In[9]:


# Test model

text = tokenizer.apply_chat_template(
    test,
    tokenize = False,
    add_generation_prompt=True,
    enable_thinking=False     
)

inputs = tokenizer(text, return_tensors = "pt").to("cuda")

outputs = model.generate(
    **inputs,
    temperature = 0.7,
    top_p=0.8,
    top_k=20,
    min_p=0.0,
    max_new_tokens = 2048,
    streamer = TextStreamer(tokenizer, skip_prompt = True),
)


# In[10]:


output = outputs[:, inputs["input_ids"].shape[1]:]

output_text = tokenizer.decode(output[0], skip_special_tokens=True)



# In[11]:


chunk_tokens = tokenizer(train_dataset[0]["chunk"])

tokenizer.decode(chunk_tokens["input_ids"], skip_special_tokens=True)


# In[12]:


print(f"""The number of tokens in the input was {len(chunk_tokens["input_ids"])}""")
print(f"The number of tokens in the output was {len(output[0])-1}")


# In[13]:


# Train Dataset
trainset = [
    {"prompt": sample["chunk"]} for sample in train_dataset
]
trainset = Dataset.from_list(trainset)
trainset = trainset.map(lambda x: {
    "prompt" : [
        {"role": "system", "content": system_prompt},
        {"role": "user",   "content": x["prompt"]},
    ],
})


# In[14]:


# Eval Dataset
evalset = [
    {"prompt": sample["chunk"]} for sample in eval_dataset[:20]
]
evalset = Dataset.from_list(evalset)
evalset = evalset.map(lambda x: {
    "prompt" : [
        {"role": "system", "content": system_prompt},
        {"role": "user",   "content": x["prompt"]},
    ],
})

# Disable thinking
_original_apply_chat_template = tokenizer.apply_chat_template

def wrapped_apply_chat_template(conversation, *args, **kwargs):
    # Always inject enable_thinking=False
    kwargs['enable_thinking'] = False
    return _original_apply_chat_template(conversation, *args, **kwargs)

tokenizer.apply_chat_template = wrapped_apply_chat_template

tokenizer.apply_chat_template


# # GRPO

# ## Reconstruction

# In[15]:


decoding_system_prompt = (
    "You are given compressed context created by another model.\n"
    "Your goal is to accurately reconstruct the original uncompressed content IN ENGLISH.\n"
    "Expand all abbreviated, shortened, or implied information back to its full form.\n"
    "Ensure that no information is lost or altered from the original meaning.\n"
    "Do NOT include any reasoning or commentary — only output the reconstructed content."
)


# In[16]:


def recontruct_input(output):

    messages = [
        {"role": "system", "content": decoding_system_prompt},
        {"role": "user", "content": output}
    ]

    text = tokenizer.apply_chat_template(
        messages,
        tokenize = False,
        add_generation_prompt=True,
        enable_thinking=False     
    )

    inputs = tokenizer(text, return_tensors = "pt").to("cuda")

    resp = model.generate(
        **inputs,
        temperature = 0.7,
        top_p=0.8,
        top_k=20,
        min_p=0.0,
        max_new_tokens = 1024,
    )

    reconstructed_tokens = resp[:, inputs["input_ids"].shape[1]:]

    reconstructed_text = tokenizer.decode(reconstructed_tokens[0], skip_special_tokens=True)

    return reconstructed_text




# ## Reward formulation

# In[17]:


def get_num_tokens(text: str):
    chunk_tokens = tokenizer(text)
    return float(len(chunk_tokens["input_ids"]))



def get_bleu_score(original, reconstructed):
    hypothesis = reconstructed.split()
    reference = original.split()
    
    BLEUscore = nltk.translate.bleu_score.sentence_bleu([reference], hypothesis)
    return BLEUscore - 1

def get_length_reward(original, compressed):
    len_original = get_num_tokens(original)
    len_compressed = get_num_tokens(compressed)
    
    # r = (len_original - len_compressed) * (1/len_original)
    r = np.clip((len_original - len_compressed) * (2/len_original), -1, 1)

    return r
    

def calculate_rewards(prompts, completions, alpha: float = 0.90, **kwargs):
    chunk = prompts[0][-1]["content"]
    responses = [completion[0]["content"] for completion in completions]

    rewards = []

    for response in responses:
        # First calculate r_len
        r_len = get_length_reward(chunk, response)

        # Now reconstruct input
        reconstructed = recontruct_input(response)

        r_bleu = get_bleu_score(chunk ,reconstructed)

        r_final = alpha * r_bleu + (1-alpha) * r_len

        rewards.append(r_final)

    return rewards



# ## Train the model

# In[18]:


max_prompt_length = 1024
max_completion_length = max_seq_length - max_prompt_length

from vllm import SamplingParams
vllm_sampling_params = SamplingParams(
    min_p = 0,
    top_p = 0.8,
    top_k = 20,
    seed = 3407,
    stop = [tokenizer.eos_token],
    include_stop_str_in_output = True,
)

from trl import GRPOConfig, GRPOTrainer
training_args = GRPOConfig(
    vllm_sampling_params = vllm_sampling_params,
    temperature = 0.7,
    learning_rate = 5e-4,
    weight_decay = 0.001,
    warmup_ratio = 0.1,
    lr_scheduler_type = "linear",
    optim = "adamw_8bit",
    logging_steps = 1,
    per_device_train_batch_size = 4,
    gradient_accumulation_steps = 1, # Increase to 4 for smoother training
    num_generations = 4, # Decrease if out of memory
    max_prompt_length = max_prompt_length,
    max_completion_length = max_completion_length,
    # num_train_epochs = 1, # Set to 1 for a full training run
    max_steps = 1000,
    save_steps = 50,
    report_to = "none", # Can use Weights & Biases
    output_dir = "/hpc/home/bfa6/work/github/yapper/results/experiment_0/test0/output",

    # For optional training + evaluation
    # fp16_full_eval = True,
    # per_device_eval_batch_size = 4,
    # eval_accumulation_steps = 1,
    # eval_strategy = "steps",
    # eval_steps = 300,
)


# In[ ]:


# For optional training + evaluation
# new_dataset = dataset.train_test_split(test_size = 0.01)

trainer = GRPOTrainer(
    model = model,
    processing_class = tokenizer,
    reward_funcs = [
        calculate_rewards
    ],
    args = training_args,
    # train_dataset = dataset,

    # For optional training + evaluation
    train_dataset = trainset,
    # eval_dataset = evalset,
)
trainer.train()


# In[ ]:
model.save_lora("/hpc/home/bfa6/work/github/yapper/results/experiment_0/test0/save") 



