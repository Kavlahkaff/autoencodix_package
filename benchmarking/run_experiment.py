import argparse
import time
import yaml
import pathlib as Path
import logging
import sys

from autoencodix_runner.data import create_data_config
from autoencodix_runner.models import create_model
from autoencodix_runner.evaluation import evaluate


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
            "/data/horse/ws/luth474h-autoencodix_synetune/"
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
            "/data/horse/ws/luth474h-autoencodix_synetune/"
            "autoencodix_package/benchmarking/configs/search_space.yaml"
        )
    )
    return cfg["fixed"]["epochs"]


# -----------------------------------------------------------------------------
# Main job logic
# -----------------------------------------------------------------------------
def run_job(config_path):
    logger.info("Loading config from: %s", config_path)

    with open(config_path, "r") as f:
        job = yaml.safe_load(f)

    logger.info("Starting Run ID: %s", job["run_id"])
    logger.info("Architecture: %s | Dataset: %s", job["architecture"], job["dataset"])
    logger.info("Hyperparameters: %s", job["hyperparameters"])

    # 1. Setup Data
    logger.info("Creating data configuration")
    data_config = create_data_config(job["dataset"], job["modalities"])

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
    logger.info("Starting model run")
    start_time = time.perf_counter()
    # Instead of result = model.run(), call the steps individually with logs:
    model.run()
    result = model.result
    print(result)
    logger.info("Model run finished")

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
    avg, rec, loss_per_epoch = evaluate(model, tasks, get_epochs())
    runtime_sec = time.perf_counter() - start_time
    logger.info("Evaluation finished (runtime %.2f sec)", runtime_sec)

    # 6. Save Results
    results = {
        "RUN_ID": job["run_id"],
        "ARCHITECTURE": job["architecture"],
        "SEED": job["seed"],
        "DATASET": job["dataset"],
        "MODALITIES": job["modalities"],
        "HYPERPARAMETERS": job["hyperparameters"],
        "AVG_ML_TASK_PERFORMANCE": avg,
        "VALID_RECON_LOSS": rec,
        "loss_per_epoch": loss_per_epoch,
        "RUNTIME_SECONDS": round(runtime_sec, 4),
    }

    result_dir = (
        Path.Path(
            "/data/horse/ws/luth474h-autoencodix_synetune/autoencodix_results/"
        )
        / job["architecture"]
    )
    result_dir.mkdir(parents=True, exist_ok=True)

    output_path = result_dir / f"{job['run_id']}_result.txt"
    with open(output_path, "w") as f:
        for key, value in results.items():
            f.write(f"{key}: {value}\n")

    logger.info("Finished successfully. Results saved to %s", output_path)


# -----------------------------------------------------------------------------
# Entry point
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    logger.info("Starting run")

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", required=True, help="Path to the job specific yaml file"
    )
    args = parser.parse_args()

    logger.info("Arguments parsed, launching job")
    run_job(args.config)
