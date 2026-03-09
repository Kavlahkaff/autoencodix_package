import argparse
import time
import yaml
import logging
import sys
from autoencodix_runner.data import create_data_config
from autoencodix_runner.models import create_model
from autoencodix_runner.evaluation import evaluate
import pathlib
import json
import numpy as np
import pickle
from typing import List, Dict

# -----------------------------------------------------------------------------
# Logging setup (flushes immediately, cluster-safe)
# -----------------------------------------------------------------------------
def setup_logging():
    handler = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(process)d | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    handler.setFormatter(formatter)

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    logger.addHandler(handler)

    # Ensure line-buffered output (Python 3.7+)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(line_buffering=True)

    return logger


logger = setup_logging()


# -----------------------------------------------------------------------------
# Helper functions
# -----------------------------------------------------------------------------
def load_ontology_paths(dataset, ontology_name):
    cfg = yaml.safe_load(
        open(
            "/data/cat/ws/luth474h-autoencodix_hpo/"
            "autoencodix_package/benchmarking/configs/ontologies.yaml"
        )
    )
    if dataset not in cfg:
        raise ValueError(f"No ontology configuration for dataset: {dataset}")
    if ontology_name not in cfg[dataset]:
        raise ValueError(f"No ontology named {ontology_name} for dataset {dataset}")

    paths = cfg[dataset][ontology_name]["paths"]
    return [paths["lvl1"], paths["lvl2"]]


def get_epochs():
    cfg = yaml.safe_load(
        open(
            "/data/cat/ws/luth474h-autoencodix_hpo/"
            "autoencodix_package/benchmarking/configs/search_space.yaml"
        )
    )
    return cfg["fixed"]["epochs"]


def construct_output_path(job, base_dir="/data/cat/ws/luth474h-autoencodix_hpo/autoencodix_results"):
    """
    Construct output directory matching batch structure:
    base_dir/architecture/dataset/modality/[ontology/]seed_X/
    
    Example paths:
    - vanillix: results/vanillix/tcga/DNA_CLIN/seed_1/
    - ontix: results/ontix/tcga/DNA_CLIN/go_biological_process/seed_2/
    """
    path_parts = [
        base_dir,
        job["architecture"],
        job["dataset"],
        "_".join(job["modalities"]),  # Join modalities with underscore
    ]
    
    # Add ontology for ontix
    if job["architecture"] == "ontix" and job.get("ontology"):
        path_parts.append(job["ontology"])
    
    # Add seed directory
    #path_parts.append(f"seed_{job['seed']}")
    path_parts.append(f"beta_tests")
    result_dir = pathlib.Path(*path_parts)
    result_dir.mkdir(parents=True, exist_ok=True)
    
    return result_dir


def json_numpy_serializer(obj):
    """Helper function to handle Numpy/Tensor types automatically"""
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"Type {type(obj)} not serializable")


def load_job_configs(config_paths: List[str]) -> List[Dict]:
    """Load all job configurations from yaml files"""
    jobs = []
    for config_path in config_paths:
        logger.info(f"Loading config from: {config_path}")
        with open(config_path, "r") as f:
            job = yaml.safe_load(f)
            jobs.append(job)
    return jobs


def load_batch_config(batch_config_path: str) -> List[str]:
    """
    Load batch configuration file containing list of individual job configs.
    
    Expected format:
    configs:
      - path/to/job1.yaml
      - path/to/job2.yaml
      - path/to/job3.yaml
    """
    with open(batch_config_path, "r") as f:
        batch_cfg = yaml.safe_load(f)
    
    if "configs" not in batch_cfg:
        raise ValueError("Batch config must contain 'configs' key with list of config paths")
    
    return batch_cfg["configs"]


# -----------------------------------------------------------------------------
# Main job logic
# -----------------------------------------------------------------------------
def run_single_job(job: Dict, data_config_cache: Dict, epochs: int):
    """
    Run a single model training job.
    
    Args:
        job: Job configuration dictionary
        data_config_cache: Pre-loaded data configurations to avoid reloading
        epochs: Number of epochs from search space config
    """
    logger.info("="*80)
    logger.info("Starting Run ID: %s", job["run_id"])
    logger.info("Architecture: %s | Dataset: %s", job["architecture"], job["dataset"])
    logger.info("Modalities: %s | Seed: %s", job["modalities"], job["seed"])
    logger.info("Hyperparameters: %s", job["hyperparameters"])
    logger.info("="*80)

    # 1. Get Data Config (from cache if available)
    dataset_key = (job["dataset"], tuple(job["modalities"]))
    if dataset_key not in data_config_cache:
        logger.info("Creating data configuration for %s with modalities %s", 
                   job["dataset"], job["modalities"])
        data_config_cache[dataset_key] = create_data_config(
            job["dataset"], job["modalities"]
        )
    else:
        logger.info("Using cached data configuration")
    
    data_config = data_config_cache[dataset_key]

    # 2. Setup Ontology (if needed)
    ontology_paths = None
    if job["architecture"] == "ontix":
        if not job.get("ontology"):
            raise ValueError("Ontix architecture requires ontology defined in config.")
        logger.info("Loading ontology: %s", job["ontology"])
        ontology_paths = load_ontology_paths(job["dataset"], job["ontology"])

    # 3. Create Model
    logger.info("Creating model")
    model = create_model(
        arch=job["architecture"],
        data_config=data_config,
        hyperparams=job["hyperparameters"],
        seed=job["seed"],
        ontologies=ontology_paths,
        sep="\t",
    )
    logger.info("Model created successfully")

    # 4. Execute
    logger.info("Starting model training")
    start_time = time.perf_counter()
    model.run()
    runtime_sec = time.perf_counter() - start_time
    result = model.result
    logger.info("Model training finished (%.2f seconds)", runtime_sec)

    # 5. Evaluate
    tasks = {
        "tcga": [
            "CANCER_TYPE",
            "SUBTYPE",
            "ONCOTREE_CODE",
            "SEX",
            "AJCC_PATHOLOGIC_TUMOR_STAGE",
            "GRADE",
            "PATH_N_STAGE",
            "DSS_STATUS",
            "OS_STATUS",
        ],
        "schc": ["author_cell_type", "age_group", "sex"],
    }[job["dataset"]]

    logger.info("Starting evaluation")
    avg, rec, loss_per_epoch = evaluate(model, tasks, epochs)
    logger.info("Evaluation finished")

    # 6. Save Results
    results = {
        "RUN_ID": job["run_id"],
        "ARCHITECTURE": job["architecture"],
        "SEED": job["seed"],
        "DATASET": job["dataset"],
        "MODALITIES": job["modalities"],
        "ONTOLOGY": job.get("ontology", "N/A"),
        "HYPERPARAMETERS": job["hyperparameters"],
        "AVG_ML_TASK_PERFORMANCE": avg,
        "VALID_RECON_LOSS": rec,
        "loss_per_epoch": loss_per_epoch,
        "RUNTIME_SECONDS": round(runtime_sec, 4),
    }

    # Construct output directory matching batch structure
    result_dir = construct_output_path(job)
    result_dir.mkdir(parents=True, exist_ok=True)

    output_path = result_dir / f"{job['run_id']}_result.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=4, default=json_numpy_serializer)
    
    pkl_path = result_dir / f"{job['run_id']}_full_object.pkl"
    with open(pkl_path, "wb") as f:
        pickle.dump(result, f)
    
    logger.info("Results saved to %s", output_path)
    logger.info("Full object saved to %s", pkl_path)
    
    return results


def run_batch(config_paths: List[str]):
    """
    Run multiple jobs in sequence, reusing data configurations.
    
    Args:
        config_paths: List of paths to individual job config files
    """
    logger.info("Starting batch run with %d configurations", len(config_paths))
    
    # Load all job configurations
    jobs = load_job_configs(config_paths)
    
    # Load epochs once
    epochs = get_epochs()
    logger.info("Loaded epochs configuration: %d", epochs)
    
    # Cache for data configurations (key: (dataset, modalities_tuple))
    data_config_cache = {}
    
    # Track results
    all_results = []
    failed_jobs = []
    
    batch_start_time = time.perf_counter()
    
    for idx, job in enumerate(jobs, 1):
        logger.info("\n" + "="*80)
        logger.info(f"BATCH PROGRESS: Job {idx}/{len(jobs)}")
        logger.info("="*80)
        
        try:
            result = run_single_job(job, data_config_cache, epochs)
            all_results.append(result)
            logger.info("✓ Job %d/%d completed successfully", idx, len(jobs))
            
        except Exception as e:
            logger.error("✗ Job %d/%d FAILED: %s", idx, len(jobs), job["run_id"])
            logger.error("Error: %s", str(e), exc_info=True)
            failed_jobs.append({
                "run_id": job["run_id"],
                "error": str(e),
                "config_path": config_paths[idx-1]
            })
    
    batch_runtime = time.perf_counter() - batch_start_time
    
    # Print summary
    logger.info("\n" + "="*80)
    logger.info("BATCH RUN SUMMARY")
    logger.info("="*80)
    logger.info("Total jobs: %d", len(jobs))
    logger.info("Successful: %d", len(all_results))
    logger.info("Failed: %d", len(failed_jobs))
    logger.info("Total runtime: %.2f seconds (%.2f minutes)", 
               batch_runtime, batch_runtime/60)
    
    if failed_jobs:
        logger.warning("\nFailed jobs:")
        for fail in failed_jobs:
            logger.warning("  - %s: %s", fail["run_id"], fail["error"])
    
    logger.info("="*80)
    
    return all_results, failed_jobs


# -----------------------------------------------------------------------------
# Entry point
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    logger.info("Starting batch runner")

    parser = argparse.ArgumentParser(
        description="Run multiple model training jobs in batch mode"
    )
    parser.add_argument(
        "--batch-config",
        help="Path to batch config yaml containing list of job configs"
    )
    parser.add_argument(
        "--configs",
        nargs="+",
        help="List of individual job config yaml files"
    )
    
    args = parser.parse_args()
    
    # Get config paths from either batch config or individual configs
    if args.batch_config:
        logger.info("Loading batch configuration from: %s", args.batch_config)
        config_paths = load_batch_config(args.batch_config)
    elif args.configs:
        logger.info("Using %d individual config files", len(args.configs))
        config_paths = args.configs
    else:
        raise ValueError(
            "Must provide either --batch-config or --configs arguments"
        )
    
    logger.info("Found %d job configurations to run", len(config_paths))
    
    # Run the batch
    all_results, failed_jobs = run_batch(config_paths)
    
    # Exit with error code if any jobs failed
    if failed_jobs:
        sys.exit(1)
    else:
        logger.info("All jobs completed successfully!")
        sys.exit(0)
