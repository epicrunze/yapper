import os
from pathlib import Path

os.environ["TRANSFORMERS_CACHE"] = "/hpc/home/bfa6/work/llms/.cache"
os.environ["HF_HOME"] = "/hpc/home/bfa6/work/llms/.cache"

import json
import random
import torch
from unsloth import FastLanguageModel
from tqdm import tqdm
import numpy as np
from datasets import Dataset
import argparse

# Parse arguments
parser = argparse.ArgumentParser(description="Train Qwen model with GRPO")
parser.add_argument("--lambda_qa", type=float, default=-0.2, help="Lambda QA penalty value")
parser.add_argument("--experiment_name", type=str, default="test2", help="Experiment name for output directory")
parser.add_argument("--max_steps", type=int, default=5000, help="Maximum training steps")
parser.add_argument("--learning_rate", type=float, default=1e-5, help="Learning rate")
parser.add_argument("--seed", type=int, default=42, help="Random seed")
parser.add_argument("--lora_rank", type=int, default=8, help="LoRA rank")
parser.add_argument("--per_device_train_batch_size", type=int, default=4, help="Per device training batch size")
parser.add_argument("--gradient_accumulation_steps", type=int, default=1, help="Gradient accumulation steps")
parser.add_argument("--num_generations", type=int, default=4, help="Number of generations per prompt")
parser.add_argument("--save_steps", type=int, default=50, help="Save checkpoint every N steps")
parser.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature")
parser.add_argument("--num_qa_samples", type=int, default=1, help="Number of QA samples to evaluate per generation")
args = parser.parse_args()

# Constants
lambda_qa = args.lambda_qa
max_seq_length = 2048 
YAPPER_DIR = Path("/hpc/home/bfa6/work/github/yapper")
DATASET_PATH = YAPPER_DIR / "dataset/splits.json"
SEED = args.seed
BASE_PATH = YAPPER_DIR / "experiments/experiment8_no_grammer"
RESULTS_PATH = BASE_PATH / "results"
lora_rank = args.lora_rank
UNSLOTH_SEED = 3407
EXPERIMENT_NAME = args.experiment_name
EXPERIMENT_DIR = RESULTS_PATH / EXPERIMENT_NAME
EXPERIMENT_DIR.mkdir(parents=True, exist_ok=True)


# Set seed
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

model = FastLanguageModel.get_peft_model(
    model,
    r = lora_rank, # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
    target_modules = [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ],
    lora_alpha = lora_rank*2, # *2 speeds up training
    use_gradient_checkpointing = "unsloth", # Reduces memory usage
    random_state = UNSLOTH_SEED,
)

qa_model, _ = FastLanguageModel.from_pretrained(
    model_name = "unsloth/Qwen3-4B-Instruct-2507",
    cache_dir="/hpc/home/bfa6/work/llms/.cache",
    max_seq_length = max_seq_length,
    load_in_4bit = False,
    fast_inference = False,
    gpu_memory_utilization = 0.3,  # Lower since it's just for inference
)
qa_model.eval()

# prompts
train_system_prompt = (
    "Your goal is to compress the information from the user in as few tokens as necessary and output the compressed version.\n"
    "Hint: Grammar & spelling don't need to be perfect—just keep the meaning clear.\n"
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


# Load train set
with open(DATASET_PATH, "r", encoding="utf-8") as f:
    data = json.load(f)

train_dataset = data["train"]

trainset = [
    {"prompt": sample["chunk"], "qa": sample["QAs"]} for sample in train_dataset
]
trainset = Dataset.from_list(trainset)
trainset = trainset.map(lambda x: {
    "prompt" : [
        {"role": "system", "content": train_system_prompt},
        {"role": "user",   "content": x["prompt"]},
    ],
})


# Yes/No Grader
def check_answer(predicted: str, true: str):
    def normalize(s: str):
        return s.strip().lower().removesuffix('.')
    
    return normalize(predicted) == normalize(true)

# Reward function
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

    qa_model.eval()
    with torch.no_grad():
        outputs = qa_model.generate(
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
    return is_correct

def get_num_tokens(text: str):
    chunk_tokens = tokenizer(text)
    return float(len(chunk_tokens["input_ids"]))

def get_length_reward(original, compressed):
    len_original = get_num_tokens(original)
    len_compressed = get_num_tokens(compressed)
    
    r = (len_original - len_compressed) * (1/len_original)
    # r = np.clip((len_original - len_compressed) * (2/len_original), -1, 1)
    return r

def calculate_rewards(prompts, completions, qa, **kwargs):
    chunk = prompts[0][-1]["content"]
    responses = [completion[0]["content"] for completion in completions]

    rewards = []
    qas = qa[0]

    for response in responses:
        # First calculate r_len
        r_len = get_length_reward(chunk, response)
        
        # Sample multiple random questions (with replacement if num_qa_samples > len(qas))
        num_samples = min(args.num_qa_samples, len(qas)) if args.num_qa_samples <= len(qas) else args.num_qa_samples
        
        # Sample questions
        if num_samples <= len(qas):
            sampled_indices = random.sample(range(len(qas)), num_samples)
        else:
            # Sample with replacement if we need more samples than available questions
            sampled_indices = [random.randint(0, len(qas) - 1) for _ in range(num_samples)]
        
        # Calculate r_qa for all sampled questions
        num_correct = 0
        for idx in sampled_indices:
            sampled_question = qas[idx]["Question"]
            sampled_answer = qas[idx]["Answer"]
            is_correct = calculate_qa_reward(response, sampled_question, sampled_answer)
            if is_correct:
                num_correct += 1
        
        # Calculate QA accuracy
        qa_accuracy = num_correct / num_samples
        
        # Reward is weighted average: qa_accuracy * r_len + (1 - qa_accuracy) * lambda_qa
        reward = qa_accuracy * r_len + (1 - qa_accuracy) * lambda_qa
        rewards.append(reward)

    return rewards

# Train
max_prompt_length = 1024
max_completion_length = max_seq_length - max_prompt_length

from vllm import SamplingParams
vllm_sampling_params = SamplingParams(
    min_p = 0,
    top_p = 0.8,
    top_k = 20,
    seed = UNSLOTH_SEED,
    stop = [tokenizer.eos_token],
    include_stop_str_in_output = True,
)

from trl import GRPOConfig, GRPOTrainer

class FixedGRPOConfig(GRPOConfig):
    def to_dict(self):
        d = super().to_dict()
        if "vllm_sampling_params" in d and d["vllm_sampling_params"] is not None:
            # Convert SamplingParams to a serializable dict
            params = d["vllm_sampling_params"]
            d["vllm_sampling_params"] = {
                "min_p": getattr(params, "min_p", None),
                "top_p": getattr(params, "top_p", None),
                "top_k": getattr(params, "top_k", None),
                "seed": getattr(params, "seed", None),
                "stop": getattr(params, "stop", None),
                "include_stop_str_in_output": getattr(params, "include_stop_str_in_output", None),
            }
        return d

training_params = dict(
    temperature = args.temperature,
    learning_rate = args.learning_rate,
    weight_decay = 0.001,
    warmup_ratio = 0.1,
    lr_scheduler_type = "linear",
    optim = "adamw_8bit",
    logging_steps = 1,
    per_device_train_batch_size = args.per_device_train_batch_size,
    gradient_accumulation_steps = args.gradient_accumulation_steps,
    num_generations = args.num_generations,
    max_prompt_length = max_prompt_length,
    max_completion_length = max_completion_length,
    # num_train_epochs = 1, # Set to 1 for a full training run
    max_steps = args.max_steps,
    save_steps = args.save_steps,
    report_to = "tensorboard", # Can use Weights & Biases
    logging_dir = str(EXPERIMENT_DIR / "tensorboard"),
    output_dir = str(EXPERIMENT_DIR / "output"),
)

training_args = FixedGRPOConfig(
    vllm_sampling_params = vllm_sampling_params,
    **training_params
)

# Save metadata
metadata = {
    "max_seq_length": max_seq_length,
    "dataset_path": str(DATASET_PATH),
    "seed": SEED,
    "lora_rank": lora_rank,
    "lambda_qa": lambda_qa,
    "num_qa_samples": args.num_qa_samples,
    "unsloth_seed": UNSLOTH_SEED,
    "experiment_name": EXPERIMENT_NAME,
    "model_name": "unsloth/Qwen3-4B-Instruct-2507",
    "training_params": training_params,
}

with open(EXPERIMENT_DIR / "metadata.json", "w") as f:
    json.dump(metadata, f, indent=4)

trainer = GRPOTrainer(
    model = model,
    processing_class = tokenizer,
    reward_funcs = [
        calculate_rewards
    ],
    args = training_args,
    train_dataset = trainset,
)
trainer.train()

model.save_lora(str(EXPERIMENT_DIR / "save")) 
