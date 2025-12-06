#!/usr/bin/env python3
"""
Submit SLURM jobs to evaluate all experiments in the results directory.
Skips experiments that already have eval_results.json unless --force is used.
"""

import argparse
import subprocess
from pathlib import Path
import json


# SLURM Configuration
SLURM_CONFIG = {
    "partition": "scavenger-gpu",
    "gres": "gpu:6000_ada:1",
    "mail_type": "FAIL",  # Only email on failure for evals
    "mail_user": "baraa.abed@duke.edu",
    "memory": "16G"
}

# Path Configuration
PATH_CONFIG = {
    "yapper_dir": "/work/kz133/yapper",
    "experiment_subdir": "experiments/experiment6_explicit_language_mixing",
    "dataset_path": "/work/kz133/yapper/dataset/splits.json",
    "venv_activate": ".venv/bin/activate"
}


def has_eval_results(experiment_dir):
    """Check if experiment already has eval_results.json"""
    eval_results_file = experiment_dir / "eval_results.json"
    return eval_results_file.exists()


def has_saved_model(experiment_dir):
    """Check if experiment has a saved model"""
    save_dir = experiment_dir / "save"
    return save_dir.exists() and save_dir.is_dir()


def submit_eval_job(experiment_dir, experiment_name, dataset_path, dryrun=False):
    """
    Submit a SLURM job to evaluate a specific experiment.
    
    Args:
        experiment_dir: Path to the experiment directory
        experiment_name: Name of the experiment
        dataset_path: Path to the dataset splits file
        dryrun: If True, print the script instead of submitting
    """
    yapper_dir = Path(PATH_CONFIG["yapper_dir"])
    experiment_subdir = yapper_dir / PATH_CONFIG["experiment_subdir"]
    eval_script = experiment_subdir / "scripts/eval_qwen.py"
    venv_activate = yapper_dir / PATH_CONFIG["venv_activate"]
    slurm_logs_dir = experiment_subdir / "slurm_logs"
    save_path = experiment_dir / "save"
    
    # Ensure slurm_logs directory exists
    slurm_logs_dir.mkdir(parents=True, exist_ok=True)
    
    # Construct job name
    job_name = f"eval_{experiment_name}"
    
    # Build evaluation command (experiment_name is auto-detected from save_path)
    # Use single line to avoid bash line continuation issues
    eval_command = f"python {eval_script} --save_path {save_path} --dataset_path {dataset_path}"
    
    # Construct the sbatch script
    sbatch_script = f"""#!/bin/bash
#SBATCH -p {SLURM_CONFIG["partition"]}
#SBATCH --gres={SLURM_CONFIG["gres"]}
#SBATCH --mail-type={SLURM_CONFIG["mail_type"]}
#SBATCH --mail-user={SLURM_CONFIG["mail_user"]}
#SBATCH --mem={SLURM_CONFIG["memory"]}
#SBATCH --job-name={job_name}
#SBATCH --output={slurm_logs_dir}/eval_{experiment_name}_%j.out
#SBATCH --error={slurm_logs_dir}/eval_{experiment_name}_%j.err

export PYTHONUNBUFFERED=1

source {venv_activate}

{eval_command}
"""
    
    if dryrun:
        print(f"\n{'='*80}")
        print(f"DRY RUN - Would submit job: {job_name}")
        print(f"Experiment: {experiment_name}")
        print(f"{'='*80}")
        print("SBATCH script:")
        print(sbatch_script)
        print(f"{'='*80}\n")
        return True
    else:
        # Submit the job
        result = subprocess.run(
            ["sbatch"],
            input=sbatch_script,
            text=True,
            capture_output=True
        )
        
        if result.returncode == 0:
            print(f"✓ Submitted eval job for {experiment_name}: {result.stdout.strip()}")
            return True
        else:
            print(f"✗ Failed to submit eval job for {experiment_name}: {result.stderr.strip()}")
            return False


def main():
    parser = argparse.ArgumentParser(
        description="Submit SLURM jobs to evaluate all experiments in the results directory"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-evaluate experiments even if eval_results.json already exists"
    )
    parser.add_argument(
        "--dryrun",
        action="store_true",
        help="Print SLURM scripts without actually submitting jobs"
    )
    parser.add_argument(
        "--results_base_path",
        type=str,
        default=None,
        help="Base path for results directory (default: auto-detected from PATH_CONFIG)"
    )
    
    args = parser.parse_args()
    
    # Setup paths
    yapper_dir = Path(PATH_CONFIG["yapper_dir"])
    experiment_dir = yapper_dir / PATH_CONFIG["experiment_subdir"]
    
    if args.results_base_path:
        results_base_path = Path(args.results_base_path)
    else:
        results_base_path = experiment_dir / "results"
    
    dataset_path = PATH_CONFIG["dataset_path"]
    
    # Find all experiment directories
    if not results_base_path.exists():
        print(f"Error: Results directory does not exist: {results_base_path}")
        return
    
    experiment_dirs = sorted([d for d in results_base_path.iterdir() if d.is_dir()])
    
    if not experiment_dirs:
        print(f"No experiment directories found in {results_base_path}")
        return
    
    # Filter experiments based on criteria
    experiments_to_eval = []
    skipped_no_model = []
    skipped_already_evaluated = []
    
    for exp_dir in experiment_dirs:
        exp_name = exp_dir.name
        
        # Check if it has a saved model
        if not has_saved_model(exp_dir):
            skipped_no_model.append(exp_name)
            continue
        
        # Check if already evaluated
        if has_eval_results(exp_dir) and not args.force:
            skipped_already_evaluated.append(exp_name)
            continue
        
        experiments_to_eval.append((exp_dir, exp_name))
    
    # Print summary
    print(f"\n{'='*80}")
    print(f"Evaluation Job Submission Summary")
    print(f"{'='*80}")
    print(f"Total experiment directories found: {len(experiment_dirs)}")
    print(f"Experiments to evaluate: {len(experiments_to_eval)}")
    print(f"Skipped (no saved model): {len(skipped_no_model)}")
    print(f"Skipped (already evaluated): {len(skipped_already_evaluated)}")
    print(f"Force mode: {args.force}")
    print(f"Dry run mode: {args.dryrun}")
    print(f"{'='*80}\n")
    
    if args.dryrun:
        print("DRY RUN MODE - No jobs will be submitted\n")
    
    if skipped_no_model:
        print(f"\nExperiments without saved models ({len(skipped_no_model)}):")
        for exp_name in skipped_no_model[:5]:  # Show first 5
            print(f"  - {exp_name}")
        if len(skipped_no_model) > 5:
            print(f"  ... and {len(skipped_no_model) - 5} more")
    
    if skipped_already_evaluated:
        print(f"\nExperiments already evaluated ({len(skipped_already_evaluated)}):")
        for exp_name in skipped_already_evaluated[:5]:  # Show first 5
            print(f"  - {exp_name}")
        if len(skipped_already_evaluated) > 5:
            print(f"  ... and {len(skipped_already_evaluated) - 5} more")
        print("\nUse --force to re-evaluate these experiments")
    
    # Submit evaluation jobs
    if not experiments_to_eval:
        print("\n✓ No experiments need evaluation")
        return
    
    print(f"\n{'='*80}")
    print(f"Submitting evaluation jobs...")
    print(f"{'='*80}\n")
    
    successful = 0
    failed = 0
    
    for exp_dir, exp_name in experiments_to_eval:
        success = submit_eval_job(exp_dir, exp_name, dataset_path, dryrun=args.dryrun)
        if success:
            successful += 1
        else:
            failed += 1
    
    # Final summary
    print(f"\n{'='*80}")
    if args.dryrun:
        print(f"DRY RUN COMPLETE")
        print(f"Would submit {len(experiments_to_eval)} evaluation jobs")
    else:
        print(f"JOB SUBMISSION COMPLETE")
        print(f"Successfully submitted: {successful}")
        print(f"Failed to submit: {failed}")
        print(f"Total: {len(experiments_to_eval)}")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()
