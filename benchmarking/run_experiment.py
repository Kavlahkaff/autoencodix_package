import pathlib as Path
import argparse
import time
from autoencodix_runner.data import create_data_config
from autoencodix_runner.models import create_model
from autoencodix_runner.evaluation import evaluate
from autoencodix_runner.hp_loader import load_results, get_top_k_configs
from autoencodix_runner.hyperparams import sample_hyperparams
import torch
import pickle

import yaml

def load_ontology_paths(dataset, ontology_name):
    cfg = yaml.safe_load(open("/data/horse/ws/luth474h-autoencodix_synetune/autoencodix_package/benchmarking/configs/ontologies.yaml"))

    if dataset not in cfg:
        raise ValueError(f"No ontology configuration for dataset: {dataset}")

    if ontology_name not in cfg[dataset]:
        raise ValueError(f"No ontology named {ontology_name} for dataset {dataset}")

    paths = cfg[dataset][ontology_name]["paths"]
    # return list in correct order (lvl1, lvl2)
    return [paths["lvl1"], paths["lvl2"]]



def parse():
    ap = argparse.ArgumentParser()
    ap.add_argument("--architecture", required=True)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--modalities", nargs="+", required=True)
    ap.add_argument("--ontology", required=False)
    ap.add_argument("--search-config")
    ap.add_argument("--hp-source", choices=["random", "previous"], default="random",
                    help="Choose whether to sample new HPs or load best previous ones")

    ap.add_argument("--previous-results-path", type=str, default=None,
                    help="Path to a parquet file containing historical results")

    ap.add_argument("--top-k", type=int, default=1,
                    help="How many top configs to test (if hp-source=previous)")
    ap.add_argument("--seeds", type=int, default=3)
    return ap.parse_args()

def main():
    args = parse()

    data_config = create_data_config(args.dataset, args.modalities)

    # if random, samples new config
    if args.hp_source == "random":
        hyperparam_list = [sample_hyperparams()]
    # if previous loads the best k configs from previous results
    else:
        if not args.previous_results_path:
            raise ValueError("hp-source=previous requires --previous-results-path")

        df = load_results(args.previous_results_path)
        hyperparam_list = get_top_k_configs(df, k=args.top_k, architecture=args.architecture)

    ontology_paths = None
    if args.architecture == "ontix":
        if not args.ontology:
            raise ValueError("Ontix architecture requires --ontology argument.")
        ontology_paths = load_ontology_paths(
            dataset=args.dataset,
            ontology_name=args.ontology
        )
    for seed in range(args.seeds):
        random.seed(seed)
        for hp_idx, hyperparams in enumerate(hyperparam_list):

            print(f"Running HP set {hp_idx}: {hyperparams}")

            run_id = f"{args.architecture}_{args.dataset}_{seed}_{'_'.join(args.modalities)}_hp{hp_idx}"

            model = create_model(
                arch=args.architecture,
                data_config=data_config,
                hyperparams=hyperparams,
                seed=seed,
                ontologies=ontology_paths,
                sep="\t"
            )
            start_time = time.perf_counter()
            result = model.run()

            tasks = {
                "tcga": ["CANCER_TYPE", "SUBTYPE", "ONCOTREE_CODE", "SEX", "AJCC_PATHOLOGIC_TUMOR_STAGE", "GRADE",
                          "PATH_N_STAGE", "DSS_STATUS", "OS_STATUS"],
                "schc": ["author_cell_type", "age_group", "sex"],
            }[args.dataset]

            avg, rec = evaluate(model, tasks)
            end_time = time.perf_counter()
            runtime_sec = end_time - start_time

            print("AVG_SCORE:", avg)
            print("RECON_LOSS:", rec)
            config_dict = {
                    "RUN_ID": run_id,
                    "ARCHITECTURE": args.architecture,
                    "SEED": seed,
                    "DATASET": args.dataset,
                    "MODALITIES": args.modalities,
                    "HYPERPARAMETERS": hyperparams,
                    "AVG_ML_TASK_PERFORMANCE": avg,
                    "VALID_RECON_LOSS": rec,
                    "RUNTIME_SECONDS": round(runtime_sec, 2),
                    }
            run_type = "gpu" if torch.cuda.is_available() else "cpu"
            # Save to txt file in result directory
            result_dir = Path.Path("/data/horse/ws/luth474h-autoencodix_synetune/autoencodix_results/")
            results_dir = result_dir / run_type
            results_dir.mkdir(parents=True, exist_ok=True)
            # write file to result_dir
            result_path = results_dir / f"{run_id}_result.txt"
            with result_path.open('w') as f:
                for key, value in config_dict.items():
                    f.write(f"{key}: {value}\n")

if __name__ == "__main__":
    main()
