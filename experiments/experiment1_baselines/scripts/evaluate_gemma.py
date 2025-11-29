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
RESULTS_PATH = "/hpc/home/bfa6/work/github/yapper/experiments/experiement1_baselines/results/gemma_eval.json"

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
    model_name = "unsloth/gemma-3-4b-it",
    cache_dir="/hpc/home/bfa6/work/llms/.cache",
    max_seq_length = max_seq_length,
    load_in_4bit = False,
    fast_inference = True,
    gpu_memory_utilization = 0.6,
)

# Load test set
with open(DATASET_PATH, "r", encoding="utf-8") as f:
    data = json.load(f)

testset = data["test"]

# Prepare dataset prompts
testset_prompts = []
prompt_template = """Context:
{chunk}

Now here is the question:
{question}

Respond ONLY with yes or no."""

for sample in testset:
    for qa in sample["QAs"]:
        new_sample = {
            "prompt": prompt_template.format(chunk=sample["chunk"], question=qa["Question"]),
            "answer": qa["Answer"]
        }
        testset_prompts.append(new_sample)

random.Random(SEED).shuffle(testset_prompts)

# Yes/No Grader
def check_answer(predicted: str, true: str):
    return predicted.strip().lower() == true.strip().lower()

# System prompt
system_prompt = """You will be given some context and a Yes/No question that can be answered from the context.
Your job is to answer the question. You can only answer with "yes" or "no". Anything else will be considered incorrect.
I repeat, answer with only "yes" or "no"."""

# Evaluation loop
results = []
correct = 0
total = 0

for sample in tqdm(testset_prompts, desc="Evaluating"):
    messages = [
        {"role": "system", "content": [{"type": "text", "text": system_prompt}]},
        {"role": "user", "content": [{"type": "text", "text": sample["prompt"]}]}
    ]

    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True, 
    )

    inputs = tokenizer(text, return_tensors="pt").to("cuda")

    with torch.inference_mode():
        outputs = model.generate(
            **inputs,
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