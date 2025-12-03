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
RESULTS_PATH = "/hpc/home/bfa6/work/github/yapper/experiments/experiment1_baselines/results/qwen_eval_compressed.json"

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
prompt_template = """Context:
{chunk}

Now here is the question:
{question}
"""

# Generate compressions
system_prompt = (
    "Your goal is to compress the information from the user in as few tokens as necessary and output the compressed version.\n"
    "Do NOT produce internal chain-of-thought or step-by-step reasoning.\n"
    "Start immediately with the compressed content (no extra preface)."
)

compressed_evalset = []
for sample in tqdm(evalset, desc="Generating compressions"):
    temp = sample.copy()
    
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": sample["chunk"]}
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
        max_new_tokens=1024,
        do_sample=True # Not greedy
    )

    output = outputs[:, inputs["input_ids"].shape[1]:]
    output_text = tokenizer.decode(output[0], skip_special_tokens=True)

    temp["compressed_chunk"] = output_text

    compressed_evalset.append(temp)

# Helper functions
def check_answer(predicted: str, true: str):
    def normalize(s: str):
        return s.strip().lower().removesuffix('.')
    
    return normalize(predicted) == normalize(true)

def get_num_tokens(text: str):
    chunk_tokens = tokenizer(text)
    return float(len(chunk_tokens["input_ids"]))

def get_compression_ratio(original, compressed):
    len_original = get_num_tokens(original)
    len_compressed = get_num_tokens(compressed)
    
    if len_compressed == 0:
        return 0.0
        
    return len_original / len_compressed

def calculate_qa_reward(chunk: str, question: str, answer: str):
    messages = [
        {"role": "system", "content": qa_system_prompt},
        {"role": "user", "content": prompt_template.format(chunk=chunk, question=question)}
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

    is_correct = check_answer(output_text, answer)
    return is_correct, output_text

# QA System prompt
qa_system_prompt = (
    "You will be given some context and a Yes/No question that can be answered from the context.\n"
    "Your job is to answer the question. You can only answer with \"yes\", \"no\", or \"idk\". Anything else will be considered incorrect.\n"
    "I repeat, answer with only \"yes\", \"no\", or \"idk\"."
    "Answer with \"idk\" if you can't extract the answer from the context."
)

# Evaluation loop
print("Starting evaluation...")
results = []
qa_accuracies = []
compression_ratios = []

for sample in tqdm(compressed_evalset, desc="Evaluating"):
    original_chunk = sample["chunk"]
    compressed_chunk = sample["compressed_chunk"]
    qas = sample["QAs"]
    
    # Calculate Compression Ratio
    comp_ratio = get_compression_ratio(original_chunk, compressed_chunk)
    compression_ratios.append(comp_ratio)
    
    # Calculate QA Accuracy (Average over all QAs for this chunk)
    chunk_qa_correct = 0
    detailed_qas = []
    for qa in qas:
        question = qa["Question"]
        answer = qa["Answer"]
        is_correct, predicted_answer = calculate_qa_reward(compressed_chunk, question, answer)
        
        if is_correct:
            chunk_qa_correct += 1
            
        detailed_qas.append({
            "question": question,
            "true_answer": answer,
            "predicted_answer": predicted_answer,
            "is_correct": is_correct
        })
            
    chunk_qa_acc = chunk_qa_correct / len(qas) if qas else 0.0
    qa_accuracies.append(chunk_qa_acc)
    
    results.append({
        "original": original_chunk,
        "compressed": compressed_chunk,
        "compression_ratio": comp_ratio,
        "qa_accuracy": chunk_qa_acc,
        "qas": detailed_qas
    })

# Calculate aggregate metrics
avg_qa_accuracy = np.mean(qa_accuracies)
avg_compression_ratio = np.mean(compression_ratios)

print(f"\n{'='*80}")
print(f"Evaluation Results")
print(f"{'='*80}")
print(f"Average QA Accuracy: {avg_qa_accuracy:.4f}")
print(f"Average Compression Ratio: {avg_compression_ratio:.4f}")
print(f"{'='*80}\n")

# Prepare final output
final_results = {
    "avg_qa_accuracy": avg_qa_accuracy,
    "avg_compression_ratio": avg_compression_ratio,
    "details": results
}

# Save results
with open(RESULTS_PATH, "w", encoding="utf-8") as f:
    json.dump(final_results, f, indent=2, ensure_ascii=False)

print(f"Evaluation Complete!")
print(f"Results saved to: {RESULTS_PATH}")