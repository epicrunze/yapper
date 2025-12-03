import os
from pathlib import Path

os.environ["TRANSFORMERS_CACHE"] = "/hpc/home/bfa6/work/llms/.cache"
os.environ["HF_HOME"] = "/hpc/home/bfa6/work/llms/.cache"

import json
import argparse
import torch
from unsloth import FastLanguageModel
from tqdm import tqdm
import numpy as np
from datasets import Dataset
import matplotlib.pyplot as plt
import seaborn as sns

# Parse arguments
parser = argparse.ArgumentParser(description="Evaluate trained compression model")
parser.add_argument(
    "--save_path",
    type=str,
    required=True,
    help="Path to the saved model adapters"
)
parser.add_argument(
    "--experiment_name",
    type=str,
    default=None,
    help="Name of the experiment (auto-detected from save_path if not provided)"
)
parser.add_argument(
    "--results_base_path",
    type=str,
    default="/hpc/home/bfa6/work/github/yapper/experiments/experiment4_multiseed_on_top5_normal/results",
    help="Base path for results directory"
)
parser.add_argument(
    "--dataset_path",
    type=str,
    default="/hpc/home/bfa6/work/github/yapper/dataset/splits.json",
    help="Path to the dataset splits"
)

args = parser.parse_args()

# Auto-detect experiment name from save_path if not provided
SAVE_PATH = Path(args.save_path)
if args.experiment_name is None:
    # save_path is typically: .../results/experiment_name/save
    # so we go up two levels to get the experiment directory and take its name
    EXPERIMENT_DIR = SAVE_PATH.parent
    EXPERIMENT_NAME = EXPERIMENT_DIR.name
    print(f"Auto-detected experiment name: {EXPERIMENT_NAME}")
else:
    EXPERIMENT_NAME = args.experiment_name
    RESULTS_PATH = Path(args.results_base_path)
    EXPERIMENT_DIR = RESULTS_PATH / EXPERIMENT_NAME

# Load metadata to get lambda_qa and num_qa_samples
metadata_path = EXPERIMENT_DIR / "metadata.json"
if not metadata_path.exists():
    raise FileNotFoundError(f"Metadata file not found: {metadata_path}")

with open(metadata_path, "r") as f:
    metadata = json.load(f)

lambda_qa = metadata["lambda_qa"]
num_qa_samples = metadata.get("num_qa_samples", 1)  # Default to 1 if not in metadata

print(f"Loaded from metadata: lambda_qa={lambda_qa}, num_qa_samples={num_qa_samples}")

# Constants
max_seq_length = 2048
YAPPER_DIR = Path("/hpc/home/bfa6/work/github/yapper")
DATASET_PATH = Path(args.dataset_path)
PLOTS_DIR = EXPERIMENT_DIR / "plots"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

# Prompts
train_system_prompt = (
    "Your goal is to compress the information from the user in as few tokens as necessary and output the compressed version.\n"
    "Do NOT produce internal chain-of-thought or step-by-step reasoning.\n"
    "Start immediately with the compressed content (no extra preface)."
)

qa_system_prompt = (
    "You will be given some context and a Yes/No question that can be answered from the context.\n"
    "Your job is to answer the question. You can only answer with \"yes\", \"no\", or \"idk\". Anything else will be considered incorrect.\n"
    "I repeat, answer with only \"yes\", \"no\", or \"idk\"."
    "Answer with \"idk\" if you can't extract the answer from the context."
)

qa_prompt_template = """Context:
{chunk}

Now here is the question:
{question}
"""

# Load model
print("Loading model...")
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "unsloth/Qwen3-4B-Instruct-2507",
    cache_dir="/hpc/home/bfa6/work/llms/.cache",
    max_seq_length = max_seq_length,
    load_in_4bit = False,
    fast_inference = True,
    gpu_memory_utilization = 0.6,
)

# Load adapters
print(f"Loading adapters from {SAVE_PATH}...")
model.load_adapter(str(SAVE_PATH))

# Load dataset
print(f"Loading dataset from {DATASET_PATH}...")
with open(DATASET_PATH, "r", encoding="utf-8") as f:
    data = json.load(f)

eval_data = data["eval"]

# Helper functions
# Yes/No Grader
def check_answer(predicted: str, true: str):
    def normalize(s: str):
        return s.strip().lower().removesuffix('.')
    
    return normalize(predicted) == normalize(true)

def calculate_qa_reward(chunk: str, question: str, answer: str):
    messages = [
        {"role": "system", "content": qa_system_prompt},
        {"role": "user", "content": qa_prompt_template.format(chunk=chunk, question=question)}
    ]

    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False     
    )

    inputs = tokenizer(text, return_tensors="pt").to("cuda")

    # Disable adapters before generation
    model.disable_adapters()
    
    outputs = model.generate(
        **inputs,
        temperature=0.0, # Greedy
        max_new_tokens=10,
        do_sample=False
    )
    
    # Re-enable adapters after generation
    model.enable_adapters()

    output = outputs[:, inputs["input_ids"].shape[1]:]
    output_text = tokenizer.decode(output[0], skip_special_tokens=True)

    is_correct = check_answer(output_text, answer)
    return is_correct, output_text

def get_num_tokens(text: str):
    chunk_tokens = tokenizer(text)
    return float(len(chunk_tokens["input_ids"]))

def get_length_reward(original, compressed):
    len_original = get_num_tokens(original)
    len_compressed = get_num_tokens(compressed)
    
    if len_original == 0:
        return 0.0
        
    r = (len_original - len_compressed) * (1/len_original)
    return r

def get_compression_ratio(original, compressed):
    len_original = get_num_tokens(original)
    len_compressed = get_num_tokens(compressed)
    
    if len_compressed == 0:
        return 0.0
        
    return len_original / len_compressed

def calculate_overall_reward(original, compressed, qa_accuracy):
    """
    Calculate the same reward as used in training:
    reward = qa_accuracy * r_len + (1 - qa_accuracy) * lambda_qa
    """
    r_len = get_length_reward(original, compressed)
    reward = qa_accuracy * r_len + (1 - qa_accuracy) * lambda_qa
    return reward

# Evaluation Loop
print("Starting evaluation...")
results = []
qa_accuracies = []
length_rewards = []
compression_ratios = []
overall_rewards = []

FastLanguageModel.for_inference(model)

for sample in tqdm(eval_data):
    original_chunk = sample["chunk"]
    qas = sample["QAs"]
    
    # Generate compressed version
    messages = [
        {"role": "system", "content": train_system_prompt},
        {"role": "user", "content": original_chunk}
    ]
    
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    
    inputs = tokenizer(text, return_tensors="pt").to("cuda")
    
    outputs = model.generate(
        **inputs,
        temperature=0.0,
        max_new_tokens=max_seq_length,
        do_sample=False
    )
    
    output_ids = outputs[:, inputs["input_ids"].shape[1]:]
    compressed_chunk = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    
    # Calculate Length Reward
    len_reward = get_length_reward(original_chunk, compressed_chunk)
    length_rewards.append(len_reward)
    
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
    
    # Calculate overall reward (matching training formula)
    reward = calculate_overall_reward(original_chunk, compressed_chunk, chunk_qa_acc)
    overall_rewards.append(reward)
    
    results.append({
        "original": original_chunk,
        "compressed": compressed_chunk,
        "length_reward": len_reward,
        "compression_ratio": comp_ratio,
        "qa_accuracy": chunk_qa_acc,
        "overall_reward": reward,
        "qas": detailed_qas
    })

# Calculate aggregate metrics
avg_qa_accuracy = np.mean(qa_accuracies)
avg_length_reward = np.mean(length_rewards)
avg_compression_ratio = np.mean(compression_ratios)
avg_overall_reward = np.mean(overall_rewards)

print(f"\n{'='*80}")
print(f"Evaluation Results")
print(f"{'='*80}")
print(f"Experiment: {EXPERIMENT_NAME}")
print(f"Lambda QA: {lambda_qa}")
print(f"Num QA Samples (from training): {num_qa_samples}")
print(f"\nMetrics:")
print(f"Average QA Accuracy: {avg_qa_accuracy:.4f}")
print(f"Average Length Reward: {avg_length_reward:.4f}")
print(f"Average Compression Ratio: {avg_compression_ratio:.4f}")
print(f"Average Overall Reward: {avg_overall_reward:.4f}")
print(f"{'='*80}\n")

# Save results
results_file = EXPERIMENT_DIR / "eval_results.json"
with open(results_file, "w") as f:
    json.dump({
        "experiment_name": EXPERIMENT_NAME,
        "lambda_qa": lambda_qa,
        "num_qa_samples": num_qa_samples,
        "avg_qa_accuracy": avg_qa_accuracy,
        "avg_length_reward": avg_length_reward,
        "avg_compression_ratio": avg_compression_ratio,
        "avg_overall_reward": avg_overall_reward,
        "details": results
    }, f, indent=4)
print(f"Results saved to {results_file}")

# Plots
print("Generating plots...")

# 1. Histogram of Length Rewards
plt.figure(figsize=(10, 6))
sns.histplot(length_rewards, bins=20, kde=True)
plt.title("Distribution of Length Rewards")
plt.xlabel("Length Reward")
plt.ylabel("Count")
plt.savefig(PLOTS_DIR / "length_reward_hist.png")
plt.close()

# 2. Histogram of QA Accuracies
plt.figure(figsize=(10, 6))
sns.histplot(qa_accuracies, bins=10, kde=False) # Discrete values likely
plt.title("Distribution of QA Accuracies")
plt.xlabel("QA Accuracy")
plt.ylabel("Count")
plt.savefig(PLOTS_DIR / "qa_accuracy_hist.png")
plt.close()

# 3. Scatter Plot: Length Reward vs QA Accuracy
plt.figure(figsize=(10, 6))
sns.scatterplot(x=length_rewards, y=qa_accuracies)
plt.title("Length Reward vs QA Accuracy")
plt.xlabel("Length Reward")
plt.ylabel("QA Accuracy")
plt.savefig(PLOTS_DIR / "length_vs_qa_scatter.png")
plt.close()

print(f"Plots saved to {PLOTS_DIR}")
