#### Step 0 - Definitions #####
import os
import sys
import pickle
import pandas as pd
import autoencodix as acx
# from autoencodix.configs.ontix_config import OntixConfig
from autoencodix.configs.disentanglix_config import DisentanglixConfig

data_final_folder = "/data/horse/ws/jaew523d-large_ontix_project/large_sc_data_taskRun/"
# data_final_folder = "/home/ewald/Github/autoencodix_package/data/large_sc_data/task-oriented/"

llm_ontology_folder = "/data/horse/ws/jaew523d-large_ontix_project/final_ontologies/task-oriented/"
# llm_ontology_folder = "/home/ewald/Github/autoencodix_package/data/llm_ontologies/final_ontologies/task-oriented/"

results_folder_save = "/data/horse/ws/jaew523d-large_ontix_project/results/large_disentanglix_save/tests/"
# results_folder_save = "/home/ewald/Github/autoencodix_package/results/large_disentanglix_save/tests/"

# Create folder results_folder_save if it doesn't exist
os.makedirs(results_folder_save, exist_ok=True)

dim_from_cli = sys.argv[1]  # "chatgpt_ontology__", "custom_ontology__"

file_processed = os.path.join(data_final_folder, "census-acxcontainer_train.pkl")
# file_processed = os.path.join(data_final_folder, "census-acxcontainer_tune.pkl")

# ont_files = [
# 	# Order from Latent Dim -> Hidden Dim -> Input Dim
# 	os.path.join(llm_ontology_folder, f"{ont_from_cli}ensembl_level1.tsv"),
# 	os.path.join(llm_ontology_folder, f"{ont_from_cli}ensembl_level2.tsv"),
# 	]

# #### Step 1 - Load best hyperparameters from tuning experiment #####
# with open(os.path.join(results_folder_tuning, tuning_experiment_file), "rb") as f:
# 	tuning_experiment = pickle.load(f)
# best_hyperparams = tuning_experiment.best_config()

#### Step 2 - Load data and define config #####
with open(file_processed, "rb") as f:
	acx_container = pickle.load(f)


# ## Downsample 
for split in ['train', 'valid', 'test']:
	if getattr(acx_container, split) is None:
		continue
	
	dataset = getattr(acx_container, split)
	dataset.data = dataset.data[:20000, :]
	dataset.sample_ids = dataset.sample_ids[:20000]
	dataset.metadata = dataset.metadata.loc[dataset.sample_ids, :]
	setattr(acx_container, split, dataset)

scconfig = DisentanglixConfig(
	## Fixed params
	epochs=250,	# Double epochs for reduced LR
	# epochs=5, # Reduce for testing
	checkpoint_interval= 250,
	loss_reduction= "sum",
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
	beta_mi = 0.0001,
	beta_tc = 100,
	beta_dimKL= 0.0001,
	# learning_rate= best_hyperparams['config_learning_rate'], 
	learning_rate= 0.2*1e-3, # Guessed
	# n_layers= best_hyperparams['config_n_layers'],
	n_layers= 2,
	latent_dim= int(dim_from_cli[3:]),
	save_vram=True,
	save_memory=True,
)

#### Step 4 - Final training #####
# ontix = acx.Ontix(
disentanglix = acx.Disentanglix(
	data=acx_container,
	# ontologies=ont_files,
	config=scconfig
	)

print("Starting preprocessing ...")
disentanglix.preprocess()
print("Starting training ...")
disentanglix.fit()
print("Visualizing losses ...")
disentanglix.visualize()

print("Saving plots ...")
disentanglix.visualizer.save_plots(os.path.join(results_folder_save, f"large_disentanglix_final_model_{dim_from_cli}_plots/"))
print("Training finished, saving model ...")
disentanglix.save(os.path.join(results_folder_save, f"large_disentanglix_final_model_{dim_from_cli}.pkl"), save_all=False)