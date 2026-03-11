#### Step 0 - Definitions #####
import os
import sys
import pickle
import pandas as pd
import autoencodix as acx
from autoencodix.configs.ontix_config import OntixConfig

def keep_features_from_acxcontainer(acx_container, feature_ids_to_keep):
	import numpy as np

	for split in ['train', 'valid', 'test']:
		if getattr(acx_container, split) is None:
			continue

		dataset = getattr(acx_container, split)
		feature_id_array = np.array(dataset.feature_ids)
		keep_indices = [i for i, fid in enumerate(feature_id_array) if fid in feature_ids_to_keep]

		dataset.data = dataset.data[:, keep_indices]
		dataset.feature_ids = [dataset.feature_ids[i] for i in keep_indices]
		setattr(acx_container, split, dataset)

	return acx_container

data_final_folder = "/data/horse/ws/jaew523d-large_ontix_project/large_sc_data_taskRun/"
# data_final_folder = "/data/horse/ws/jaew523d-large_ontix_project/large_sc_data/"

# llm_ontology_folder = "/data/horse/ws/jaew523d-large_ontix_project/final_ontologies/"
llm_ontology_folder = "/data/horse/ws/jaew523d-large_ontix_project/final_ontologies/task-oriented/"

results_folder_tuning = "/data/horse/ws/jaew523d-large_ontix_project/results/large_ontix_save/fourth_run_e250/"
results_folder_save = "/data/horse/ws/jaew523d-large_ontix_project/results/large_ontix_save/fourth_run_e500_lowLR/"

ont_from_cli = sys.argv[1]  # "chatgpt_ontology__", "custom_ontology__"
tuning_experiment_file = sys.argv[2]  # Name of tuning experiment pickle file

file_processed = os.path.join(data_final_folder, "census-acxcontainer_train.pkl")

ont_files = [
	# Order from Latent Dim -> Hidden Dim -> Input Dim
	os.path.join(llm_ontology_folder, f"{ont_from_cli}ensembl_level1.tsv"),
	os.path.join(llm_ontology_folder, f"{ont_from_cli}ensembl_level2.tsv"),
	]

#### Step 1 - Load best hyperparameters from tuning experiment #####
with open(os.path.join(results_folder_tuning, tuning_experiment_file), "rb") as f:
	tuning_experiment = pickle.load(f)
best_hyperparams = tuning_experiment.best_config()

#### Step 2 - Load data and define config #####
with open(file_processed, "rb") as f:
	acx_container = pickle.load(f)

### Step 3 - Restrict to features in ontology level 2 ###
ont_lvl2 = pd.read_csv(ont_files[1], sep='\t', usecols=[0], header=None)
ont_lvl2.columns = ['feature_id']

acx_container = keep_features_from_acxcontainer(acx_container, ont_lvl2['feature_id'].values)

scconfig = OntixConfig(
	## Fixed params
	epochs=best_hyperparams['config_epochs']*2,	# Double epochs for reduced LR
	# epochs=5, # Reduce for testing
	checkpoint_interval= best_hyperparams['config_checkpoint_interval'],
	loss_reduction= best_hyperparams['config_loss_reduction'],
	## Tunable params
	batch_size= best_hyperparams['config_batch_size'],
	drop_p= best_hyperparams['config_drop_p'],
	enc_factor= best_hyperparams['config_enc_factor'],
	weight_decay= best_hyperparams['config_weight_decay'],
	beta= best_hyperparams['config_beta'],
	learning_rate= best_hyperparams['config_learning_rate']*0.1, # Reduce LR for testing
	n_layers= best_hyperparams['config_n_layers'],
	save_vram=True,
	save_memory=True,
)

#### Step 4 - Final training #####
ontix = acx.Ontix(
	data=acx_container,
	ontologies=ont_files,
	config=scconfig
	)

print("Starting preprocessing ...")
ontix.preprocess()
print("Starting training ...")
ontix.fit()
print("Visualizing losses ...")
ontix.visualize()
print("Saving plots ...")
ontix.visualizer.save_plots(os.path.join(results_folder_save, f"large_ontix_final_model_{ont_from_cli}_plots/"))
print("Training finished, saving model ...")
ontix.save(os.path.join(results_folder_save, f"large_ontix_final_model_{ont_from_cli}.pkl"), save_all=False)