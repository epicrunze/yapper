#!/usr/bin/env python3
"""
Sweep across lambda_qa values and/or training parameters and submit slurm jobs for each configuration.
"""

import argparse
import subprocess
import numpy as np
from pathlib import Path
from itertools import product

# ============================================================================
# CONFIGURATION - Modify these parameters for your sweep
# ============================================================================

# Lambda Sweep Configuration
# Set lambda_values to a list to sweep specific values, or use linspace parameters
LAMBDA_CONFIG = {
    "mode": "list",  # "linspace" or "list"
    # For linspace mode:
    # "lambda_start": -0.1,
    # "lambda_end": -1.0,
    # "num_values": 5,
    # For list mode (uncomment to use):
    "lambda_values": [-0.2, -0.5, -0.8, -1.0],
}

# Training Parameter Sweep Configuration
# For each parameter, provide either:
#   - A single value (no sweep for this param)
#   - A list of values to sweep over
PARAM_SWEEP_CONFIG = {
    "max_steps": [1000],              # Single value = no sweep
    "learning_rate": [2e-5, 5e-5, 1e-4],          # Single value = no sweep
    "seed": [42],                     # Single value = no sweep
    "lora_rank": [8, 16, 32],         # Example: [8, 16, 32] to sweep
    "per_device_train_batch_size": [4],
    "gradient_accumulation_steps": [1, 4],
    "num_generations": [4],         # Example: [4, 8] to sweep
    "save_steps": [50],
    "temperature": [0.7],
    "num_qa_samples": [2]         # Example: [1, 2, 4] to sweep
}

# Experiment naming configuration
EXPERIMENT_CONFIG = {
    "prefix": "sweep",  # Base prefix for all experiments
    # Parameters to include in experiment name (will auto-add swept params)
    "name_params": ["learning_rate", "lora_rank", "gradient_accumulation_steps"]
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
    "yapper_dir": "/work/kz133/yapper",
    "experiment_subdir": "experiments/experiment6_explicit_language_mixing",
    "venv_activate": ".venv/bin/activate"
}

# ============================================================================
# End of Configuration
# ============================================================================


def generate_experiment_name(lambda_val, param_config):
    """
    Generate experiment name based on swept parameters.
    
    Args:
        lambda_val: Lambda value
        param_config: Dictionary of parameter values for this configuration
    """
    # Start with prefix
    parts = [EXPERIMENT_CONFIG["prefix"]]
    
    # Add parameters that are being swept or specified in name_params
    for param_name in EXPERIMENT_CONFIG["name_params"]:
        if param_name in param_config:
            val = param_config[param_name]
            # Format the value nicely
            if isinstance(val, float):
                if val < 1:
                    val_str = f"{val:.0e}".replace("-", "neg").replace("+", "p")
                else:
                    val_str = f"{val:.2f}".replace(".", "p")
            else:
                val_str = str(val)
            parts.append(f"{param_name}_{val_str}")
    
    # Add lambda value
    lambda_str = f"lambda_{lambda_val:.3f}".replace(".", "p").replace("-", "neg")
    parts.append(lambda_str)
    
    return "_".join(parts)


def submit_slurm_job(lambda_value, param_config, experiment_name, dryrun=False):
    """
    Submit a slurm job for a specific configuration.
    
    Args:
        lambda_value: The lambda_qa value to use
        param_config: Dictionary of training parameters
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
        f"--lambda_qa {lambda_value}",
        f"--experiment_name {experiment_name}",
    ]
    
    for param_name, param_value in param_config.items():
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
        print(f"Lambda value: {lambda_value}")
        print(f"Experiment name: {experiment_name}")
        print(f"Parameters: {param_config}")
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
            print(f"✓ Submitted job for lambda={lambda_value}: {result.stdout.strip()}")
        else:
            print(f"✗ Failed to submit job for lambda={lambda_value}: {result.stderr.strip()}")


def main():
    parser = argparse.ArgumentParser(
        description="Sweep across lambda_qa values and/or training parameters and submit slurm jobs"
    )
    parser.add_argument(
        "--dryrun",
        action="store_true",
        help="Print commands without actually submitting jobs"
    )
    
    args = parser.parse_args()
    
    # Generate lambda values
    if LAMBDA_CONFIG["mode"] == "list":
        lambda_values = LAMBDA_CONFIG["lambda_values"]
    else:  # linspace mode
        lambda_values = np.linspace(
            LAMBDA_CONFIG["lambda_start"],
            LAMBDA_CONFIG["lambda_end"],
            LAMBDA_CONFIG["num_values"]
        )
    
    # Generate all parameter combinations using grid search
    param_names = list(PARAM_SWEEP_CONFIG.keys())
    param_value_lists = [PARAM_SWEEP_CONFIG[name] for name in param_names]
    param_combinations = list(product(*param_value_lists))
    
    # Convert to list of dictionaries
    param_configs = [
        dict(zip(param_names, combo))
        for combo in param_combinations
    ]
    
    # Print configuration summary
    print(f"\n{'='*80}")
    print(f"Sweep Configuration Summary")
    print(f"{'='*80}")
    print(f"Lambda values: {len(lambda_values)} ({[f'{v:.3f}' for v in lambda_values]})")
    print(f"Parameter combinations: {len(param_configs)}")
    print(f"\nSwept parameters:")
    for param_name in param_names:
        values = PARAM_SWEEP_CONFIG[param_name]
        if len(values) > 1:
            print(f"  {param_name}: {values}")
    print(f"\nTotal jobs to submit: {len(lambda_values) * len(param_configs)}")
    print(f"Dry run mode: {args.dryrun}")
    print(f"{'='*80}\n")
    
    if args.dryrun:
        print("DRY RUN MODE - No jobs will be submitted\n")
    
    # Submit jobs for each combination
    job_count = 0
    for param_config in param_configs:
        for lambda_val in lambda_values:
            experiment_name = generate_experiment_name(lambda_val, param_config)
            submit_slurm_job(lambda_val, param_config, experiment_name, dryrun=args.dryrun)
            job_count += 1
    
    print(f"\n{'='*80}")
    if args.dryrun:
        print(f"DRY RUN COMPLETE - {job_count} jobs would be submitted")
    else:
        print(f"SWEEP COMPLETE - {job_count} jobs submitted")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()
