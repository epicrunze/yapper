#!/usr/bin/env python3
"""
Sweep over top 5 configurations from experiment3 with multiple random seeds.
"""

import argparse
import subprocess
from pathlib import Path

# ============================================================================
# CONFIGURATION - Top 5 configurations from experiment3
# ============================================================================

# Top 5 configurations by qa_accuracy from experiment3_lambda_normal
TOP_5_CONFIGS = [
    {
        "config_name": "top1",
        "learning_rate": 1e-4,
        "lora_rank": 16,
        "gradient_accumulation_steps": 4,
        "lambda_qa": -1.0,
        "qa_accuracy": 0.827,  # For reference only
    },
    {
        "config_name": "top2",
        "learning_rate": 5e-5,
        "lora_rank": 8,
        "gradient_accumulation_steps": 4,
        "lambda_qa": -0.2,
        "qa_accuracy": 0.744,
    },
    {
        "config_name": "top3",
        "learning_rate": 5e-5,
        "lora_rank": 16,
        "gradient_accumulation_steps": 4,
        "lambda_qa": -1.0,
        "qa_accuracy": 0.741,
    },
    {
        "config_name": "top4",
        "learning_rate": 1e-4,
        "lora_rank": 8,
        "gradient_accumulation_steps": 4,
        "lambda_qa": -0.2,
        "qa_accuracy": 0.738,
    },
    {
        "config_name": "top5",
        "learning_rate": 5e-5,
        "lora_rank": 16,
        "gradient_accumulation_steps": 4,
        "lambda_qa": -0.5,
        "qa_accuracy": 0.731,
    },
]

# Seed sweep configuration
SEED_SWEEP = [42, 123, 456, 789, 1011]

# Fixed parameters (not swept)
FIXED_PARAMS = {
    "max_steps": 1000,
    "per_device_train_batch_size": 4,
    "num_generations": 4,
    "save_steps": 50,
    "temperature": 0.7,
    "num_qa_samples": 2,
}

# Experiment naming configuration
EXPERIMENT_CONFIG = {
    "prefix": "multiseed",
}

# SLURM Configuration
SLURM_CONFIG = {
    "partition": "scavenger-gpu",
    "gres": "gpu:6000_ada:1",
    "mail_type": "ALL",
    "mail_user": "baraa.abed@duke.edu",
    "memory": "16G"
}

# Path Configuration
PATH_CONFIG = {
    "yapper_dir": "/hpc/home/bfa6/work/github/yapper",
    "experiment_subdir": "experiments/experiment4_multiseed_on_top5_normal",
    "venv_activate": ".venv/bin/activate"
}

# ============================================================================
# End of Configuration
# ============================================================================


def generate_experiment_name(config, seed):
    """
    Generate experiment name based on configuration and seed.
    
    Args:
        config: Configuration dictionary
        seed: Random seed value
    """
    # Format learning rate
    lr = config["learning_rate"]
    if lr < 1:
        lr_str = f"{lr:.0e}".replace("-", "neg").replace("+", "p")
    else:
        lr_str = f"{lr:.2f}".replace(".", "p")
    
    # Format lambda
    lambda_str = f"{config['lambda_qa']:.1f}".replace(".", "p").replace("-", "neg")
    
    # Build experiment name
    parts = [
        EXPERIMENT_CONFIG["prefix"],
        config["config_name"],
        f"lr_{lr_str}",
        f"rank_{config['lora_rank']}",
        f"gradaccum_{config['gradient_accumulation_steps']}",
        f"lambda_{lambda_str}",
        f"seed_{seed}"
    ]
    
    return "_".join(parts)


def submit_slurm_job(config, seed, experiment_name, dryrun=False):
    """
    Submit a slurm job for a specific configuration and seed.
    
    Args:
        config: Configuration dictionary
        seed: Random seed value
        experiment_name: Name for this experiment run
        dryrun: If True, print the command instead of running it
    """
    # Paths (from PATH_CONFIG)
    yapper_dir = Path(PATH_CONFIG["yapper_dir"])
    experiment_dir = yapper_dir / PATH_CONFIG["experiment_subdir"]
    slurm_logs_dir = experiment_dir / "slurm_logs"
    train_script = experiment_dir / "scripts/train_qwen.py"
    venv_activate = yapper_dir / PATH_CONFIG["venv_activate"]
    
    # Ensure slurm_logs directory exists
    slurm_logs_dir.mkdir(parents=True, exist_ok=True)
    
    # Construct job name
    job_name = f"train_{experiment_name}"
    
    # Build training arguments
    train_args = [
        f"--lambda_qa {config['lambda_qa']}",
        f"--experiment_name {experiment_name}",
        f"--seed {seed}",
        f"--learning_rate {config['learning_rate']}",
        f"--lora_rank {config['lora_rank']}",
        f"--gradient_accumulation_steps {config['gradient_accumulation_steps']}",
    ]
    
    # Add fixed parameters
    for param_name, param_value in FIXED_PARAMS.items():
        train_args.append(f"--{param_name} {param_value}")
    
    train_args_str = " \\\n    ".join(train_args)
    
    # Construct the sbatch command using SLURM_CONFIG
    sbatch_script = f"""#!/bin/bash
#SBATCH -p {SLURM_CONFIG["partition"]}
#SBATCH --gres={SLURM_CONFIG["gres"]}
#SBATCH --mail-type={SLURM_CONFIG["mail_type"]}
#SBATCH --mail-user={SLURM_CONFIG["mail_user"]}
#SBATCH --mem={SLURM_CONFIG["memory"]}
#SBATCH --job-name={job_name}
#SBATCH --output={slurm_logs_dir}/{experiment_name}_%j.out
#SBATCH --error={slurm_logs_dir}/{experiment_name}_%j.err

export PYTHONUNBUFFERED=1

source {venv_activate}

python {train_script} \\
    {train_args_str}
"""
    
    if dryrun:
        print(f"\n{'='*80}")
        print(f"DRY RUN - Would submit job: {job_name}")
        print(f"Config: {config['config_name']} (QA Accuracy: {config['qa_accuracy']:.3f})")
        print(f"Seed: {seed}")
        print(f"Experiment name: {experiment_name}")
        print(f"{'='*80}")
        print("SBATCH script:")
        print(sbatch_script)
        print(f"{'='*80}\n")
    else:
        # Submit the job
        result = subprocess.run(
            ["sbatch"],
            input=sbatch_script,
            text=True,
            capture_output=True
        )
        
        if result.returncode == 0:
            print(f"✓ Submitted {config['config_name']} seed={seed}: {result.stdout.strip()}")
        else:
            print(f"✗ Failed to submit {config['config_name']} seed={seed}: {result.stderr.strip()}")


def main():
    parser = argparse.ArgumentParser(
        description="Sweep over top 5 configurations from experiment3 with multiple seeds"
    )
    parser.add_argument(
        "--dryrun",
        action="store_true",
        help="Print commands without actually submitting jobs"
    )
    
    args = parser.parse_args()
    
    # Print configuration summary
    print(f"\n{'='*80}")
    print(f"Multi-Seed Sweep Configuration Summary")
    print(f"{'='*80}")
    print(f"Top configurations: {len(TOP_5_CONFIGS)}")
    print(f"Seeds per configuration: {len(SEED_SWEEP)} ({SEED_SWEEP})")
    print(f"\nConfigurations:")
    for i, config in enumerate(TOP_5_CONFIGS, 1):
        print(f"  {i}. {config['config_name']}: lr={config['learning_rate']:.0e}, " 
              f"rank={config['lora_rank']}, gradaccum={config['gradient_accumulation_steps']}, "
              f"lambda={config['lambda_qa']:.1f} (qa_acc={config['qa_accuracy']:.3f})")
    print(f"\nTotal jobs to submit: {len(TOP_5_CONFIGS) * len(SEED_SWEEP)}")
    print(f"Dry run mode: {args.dryrun}")
    print(f"{'='*80}\n")
    
    if args.dryrun:
        print("DRY RUN MODE - No jobs will be submitted\n")
    
    # Submit jobs for each configuration and seed combination
    job_count = 0
    for config in TOP_5_CONFIGS:
        for seed in SEED_SWEEP:
            experiment_name = generate_experiment_name(config, seed)
            submit_slurm_job(config, seed, experiment_name, dryrun=args.dryrun)
            job_count += 1
    
    print(f"\n{'='*80}")
    if args.dryrun:
        print(f"DRY RUN COMPLETE - {job_count} jobs would be submitted")
    else:
        print(f"SWEEP COMPLETE - {job_count} jobs submitted")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()
