#### Step 0 - Definitions #####
import os
import sys
import pickle
import pandas as pd
import autoencodix as acx
# from autoencodix.configs.ontix_config import OntixConfig
from autoencodix.configs.varix_config import VarixConfig

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

results_folder_tuning = "/data/horse/ws/jaew523d-large_ontix_project/results/large_varix_save/first_run_e250/"
results_folder_save = "/data/horse/ws/jaew523d-large_ontix_project/results/large_varix_save/second_run_e250/"

# Create folder results_folder_save if it doesn't exist
os.makedirs(results_folder_save, exist_ok=True)

dim_from_cli = sys.argv[1]  # "chatgpt_ontology__", "custom_ontology__"
tuning_experiment_file = sys.argv[2]  # Name of tuning experiment pickle file

file_processed = os.path.join(data_final_folder, "census-acxcontainer_train.pkl")

# ont_files = [
# 	# Order from Latent Dim -> Hidden Dim -> Input Dim
# 	os.path.join(llm_ontology_folder, f"{ont_from_cli}ensembl_level1.tsv"),
# 	os.path.join(llm_ontology_folder, f"{ont_from_cli}ensembl_level2.tsv"),
# 	]

#### Step 1 - Load best hyperparameters from tuning experiment #####
with open(os.path.join(results_folder_tuning, tuning_experiment_file), "rb") as f:
	tuning_experiment = pickle.load(f)
best_hyperparams = tuning_experiment.best_config()

#### Step 2 - Load data and define config #####
with open(file_processed, "rb") as f:
	acx_container = pickle.load(f)

# ### Step 3 - Restrict to features in ontology level 2 ###
# ont_lvl2 = pd.read_csv(ont_files[1], sep='\t', usecols=[0], header=None)
# ont_lvl2.columns = ['feature_id']

# acx_container = keep_features_from_acxcontainer(acx_container, ont_lvl2['feature_id'].values)

scconfig = VarixConfig(
	## Fixed params
	epochs=best_hyperparams['config_epochs'],	# Double epochs for reduced LR
	# epochs=5, # Reduce for testing
	checkpoint_interval= best_hyperparams['config_checkpoint_interval'],
	loss_reduction= best_hyperparams['config_loss_reduction'],
	## Tunable params
	# batch_size= best_hyperparams['config_batch_size'],
	batch_size= 1024, # Guessed average
	# drop_p= best_hyperparams['config_drop_p'],
	drop_p = 0.1, # Guessed average
	# enc_factor= best_hyperparams['config_enc_factor'],
	enc_factor = 3, # Guessed 
	# weight_decay= best_hyperparams['config_weight_decay'],
	weight_decay= 1e-2, # Guessed
	# beta= best_hyperparams['config_beta'],
	beta= 1e-3,
	# learning_rate= best_hyperparams['config_learning_rate'], 
	learning_rate= 0.5*1e-4, # Guessed
	# n_layers= best_hyperparams['config_n_layers'],
	n_layers= 2,
	latent_dim= int(dim_from_cli[3:]),
	save_vram=True,
	save_memory=True,
)

#### Step 4 - Final training #####
# ontix = acx.Ontix(
varix = acx.Varix(
	data=acx_container,
	# ontologies=ont_files,
	config=scconfig
	)

print("Starting preprocessing ...")
varix.preprocess()
print("Starting training ...")
varix.fit()
print("Visualizing losses ...")
varix.visualize()
print("Saving plots ...")
varix.visualizer.save_plots(os.path.join(results_folder_save, f"large_varix_final_model_{dim_from_cli}_plots/"))
print("Training finished, saving model ...")
varix.save(os.path.join(results_folder_save, f"large_varix_final_model_{dim_from_cli}.pkl"), save_all=False)