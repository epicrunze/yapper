import os

os.environ["TRANSFORMERS_CACHE"] = "/hpc/home/bfa6/work/llms/.cache"
os.environ["HF_HOME"] = "/hpc/home/bfa6/work/llms/.cache"

import json
import random
import torch
from unsloth import FastLanguageModel
from tqdm import tqdm
import numpy as np

# Constants
max_seq_length = 2048 
DATASET_PATH = "/hpc/home/bfa6/work/github/yapper/dataset/splits.json"
SEED = 42
RESULTS_PATH = "/hpc/home/bfa6/work/github/yapper/experiments/experiment1_baselines/results/qwen_eval.json"

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Make CUDA operations deterministic
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(SEED)


# Load model
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "unsloth/Qwen3-4B-Instruct-2507",
    cache_dir="/hpc/home/bfa6/work/llms/.cache",
    max_seq_length = max_seq_length,
    load_in_4bit = False,
    fast_inference = True,
    gpu_memory_utilization = 0.6,
)

# Load test set
with open(DATASET_PATH, "r", encoding="utf-8") as f:
    data = json.load(f)

evalset = data["eval"]

# Prepare dataset prompts
evalset_prompts = []
prompt_template = """Context:
{chunk}

Now here is the question:
{question}
"""

for sample in evalset:
    for qa in sample["QAs"]:
        new_sample = {
            "prompt": prompt_template.format(chunk=sample["chunk"], question=qa["Question"]),
            "answer": qa["Answer"]
        }
        evalset_prompts.append(new_sample)

random.Random(SEED).shuffle(evalset_prompts)

# Yes/No Grader
def check_answer(predicted: str, true: str):
    def normalize(s: str):
        return s.strip().lower().removesuffix('.')
    
    return normalize(predicted) == normalize(true)

# System prompt
system_prompt = (
    "You will be given some context and a Yes/No question that can be answered from the context.\n"
    "Your job is to answer the question. You can only answer with \"yes\", \"no\", or \"idk\". Anything else will be considered incorrect.\n"
    "I repeat, answer with only \"yes\", \"no\", or \"idk\"."
    "Answer with \"idk\" if you can't extract the answer from the context."
)

# Evaluation loop
results = []
correct = 0
total = 0

for sample in tqdm(evalset_prompts, desc="Evaluating"):
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": sample["prompt"]}
    ]

    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False     
    )

    inputs = tokenizer(text, return_tensors="pt").to("cuda")

    outputs = model.generate(
        **inputs,
        temperature=0.7, # Not used since its greedy
        top_p=0.8, # Not used since its greedy
        top_k=20, # Not used since its greedy
        min_p=0.0, # Not used since its greedy
        max_new_tokens=10,
        do_sample=False # Greedy
    )

    output = outputs[:, inputs["input_ids"].shape[1]:]
    output_text = tokenizer.decode(output[0], skip_special_tokens=True)

    is_correct = check_answer(output_text, sample["answer"])
    
    result = {
        "prompt": sample["prompt"],
        "true_answer": sample["answer"],
        "model_response": output_text,
        "correct": is_correct
    }
    
    results.append(result)
    
    if is_correct:
        correct += 1
    total += 1

# Calculate accuracy
accuracy = (correct / total) * 100 if total > 0 else 0

# Prepare final output
final_results = {
    "accuracy": accuracy,
    "correct": correct,
    "total": total,
    "results": results
}

# Save results
with open(RESULTS_PATH, "w", encoding="utf-8") as f:
    json.dump(final_results, f, indent=2, ensure_ascii=False)

print(f"\nEvaluation Complete!")
print(f"Accuracy: {accuracy:.2f}%")
print(f"Correct: {correct}/{total}")
print(f"Results saved to: {RESULTS_PATH}")