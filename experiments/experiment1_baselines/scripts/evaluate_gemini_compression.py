import os
import asyncio

# Set cache dirs before importing transformers
os.environ["TRANSFORMERS_CACHE"] = "/data/epicrunze/.cache/huggingface"
os.environ["HF_HOME"] = "/data/epicrunze/.cache/huggingface"

import json
import random
import torch
from unsloth import FastLanguageModel
from tqdm import tqdm
from tqdm.asyncio import tqdm as atqdm
import numpy as np
from dotenv import load_dotenv
from google import genai
from google.genai.types import GenerateContentConfig

# Load environment variables
load_dotenv()

# Constants
max_seq_length = 2048 
DATASET_PATH = "/data/epicrunze/Projects/yapper/dataset/splits.json"
SEED = 42
RESULTS_PATH = "/data/epicrunze/Projects/yapper/experiments/experiment1_baselines/results/gemini_flash_eval_compressed.json"
CHECKPOINT_PATH = "/data/epicrunze/Projects/yapper/experiments/experiment1_baselines/results/gemini_flash_checkpoint.json"
GEMINI_MODEL = "gemini-2.5-flash-lite"

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Make CUDA operations deterministic
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(SEED)

# Initialize Gemini client
gemini_client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

# Load Qwen model for QA evaluation
print("Loading Qwen model for QA evaluation...")
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "unsloth/Qwen3-4B-Instruct-2507",
    cache_dir="/data/epicrunze/.cache/huggingface",
    max_seq_length = max_seq_length,
    load_in_4bit = False,
    fast_inference = True,
    gpu_memory_utilization = 0.6,
)

# Load eval set
with open(DATASET_PATH, "r", encoding="utf-8") as f:
    data = json.load(f)

evalset = data["eval"]

# Prompts (same as Qwen baseline for fair comparison)
compression_system_prompt = (
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

prompt_template = """Context:
{chunk}

Now here is the question:
{question}
"""


def save_checkpoint(compressed_evalset: list, stage: str = "compression"):
    """Save checkpoint to allow resuming."""
    checkpoint = {
        "stage": stage,
        "compressed_evalset": compressed_evalset,
        "num_completed": len(compressed_evalset)
    }
    os.makedirs(os.path.dirname(CHECKPOINT_PATH), exist_ok=True)
    with open(CHECKPOINT_PATH, "w", encoding="utf-8") as f:
        json.dump(checkpoint, f, indent=2, ensure_ascii=False)
    print(f"Checkpoint saved: {len(compressed_evalset)} samples completed")


def load_checkpoint() -> tuple[list, str]:
    """Load checkpoint if it exists."""
    if os.path.exists(CHECKPOINT_PATH):
        with open(CHECKPOINT_PATH, "r", encoding="utf-8") as f:
            checkpoint = json.load(f)
        print(f"Resuming from checkpoint: {checkpoint['num_completed']} samples already completed")
        return checkpoint["compressed_evalset"], checkpoint["stage"]
    return [], "compression"


async def compress_with_gemini(chunk: str) -> str:
    """Compress a chunk using Gemini standard API."""
    config = GenerateContentConfig(
        system_instruction=compression_system_prompt,
        max_output_tokens=1024,
        temperature=0.7,
    )
    
    response = await gemini_client.aio.models.generate_content(
        model=GEMINI_MODEL,
        contents=chunk,
        config=config,
    )
    
    return response.text or ""


async def compress_all_chunks(evalset: list, resume_from: list = None) -> list:
    """Compress all chunks in the evalset using Gemini with checkpointing."""
    # Resume from checkpoint if available
    if resume_from:
        compressed_evalset = resume_from
        start_idx = len(resume_from)
        print(f"Resuming compression from sample {start_idx}")
    else:
        compressed_evalset = []
        start_idx = 0
    
    # Process remaining samples
    remaining_samples = evalset[start_idx:]
    
    for i, sample in enumerate(atqdm(remaining_samples, desc="Generating compressions with Gemini")):
        temp = sample.copy()
        
        try:
            compressed_chunk = await compress_with_gemini(sample["chunk"])
            temp["compressed_chunk"] = compressed_chunk
        except Exception as e:
            print(f"Error compressing chunk: {e}")
            temp["compressed_chunk"] = ""
        
        compressed_evalset.append(temp)
        
        # Save checkpoint every 10 samples
        if (start_idx + i + 1) % 10 == 0:
            save_checkpoint(compressed_evalset, stage="compression")
        
        # Small delay to be nice to the API
        await asyncio.sleep(0.1)
    
    # Final checkpoint save
    save_checkpoint(compressed_evalset, stage="compression_complete")
    
    return compressed_evalset


# Helper functions
def check_answer(predicted: str, true: str):
    def normalize(s: str):
        return s.strip().lower().removesuffix('.')
    
    return normalize(predicted) == normalize(true)

def get_num_tokens(text: str):
    """Use Qwen tokenizer for consistent token counting across baselines."""
    chunk_tokens = tokenizer(text)
    return float(len(chunk_tokens["input_ids"]))

def get_compression_ratio(original, compressed):
    len_original = get_num_tokens(original)
    len_compressed = get_num_tokens(compressed)
    
    if len_compressed == 0:
        return 0.0
        
    return len_original / len_compressed

def calculate_qa_reward(chunk: str, question: str, answer: str):
    """Evaluate QA using Qwen model."""
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


async def main():
    print("Starting Gemini Flash compression baseline evaluation...")
    print(f"Model: {GEMINI_MODEL}")
    
    # Load checkpoint if available
    checkpoint_data, stage = load_checkpoint()
    
    # Step 1: Compress all chunks with Gemini
    print("\n" + "="*80)
    print("Step 1: Compressing chunks with Gemini Flash")
    print("="*80)
    
    if stage == "compression_complete" and len(checkpoint_data) == len(evalset):
        print("Compression already complete, loading from checkpoint...")
        compressed_evalset = checkpoint_data
    else:
        compressed_evalset = await compress_all_chunks(evalset, resume_from=checkpoint_data if stage == "compression" else None)
    
    # Step 2: Evaluate each compressed chunk with Qwen QA
    print("\n" + "="*80)
    print("Step 2: Evaluating QA with Qwen")
    print("="*80)
    
    results = []
    qa_accuracies = []
    compression_ratios = []
    
    for sample in tqdm(compressed_evalset, desc="Evaluating QA"):
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
        "model": GEMINI_MODEL,
        "compression_model": "gemini-2.5-flash (Standard API)",
        "qa_model": "Qwen3-4B-Instruct-2507",
        "avg_qa_accuracy": avg_qa_accuracy,
        "avg_compression_ratio": avg_compression_ratio,
        "details": results
    }
    
    # Save results
    os.makedirs(os.path.dirname(RESULTS_PATH), exist_ok=True)
    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump(final_results, f, indent=2, ensure_ascii=False)
    
    # Clean up checkpoint file after successful completion
    if os.path.exists(CHECKPOINT_PATH):
        os.remove(CHECKPOINT_PATH)
        print("Checkpoint file cleaned up.")
    
    print(f"Evaluation Complete!")
    print(f"Results saved to: {RESULTS_PATH}")


if __name__ == "__main__":
    asyncio.run(main())
