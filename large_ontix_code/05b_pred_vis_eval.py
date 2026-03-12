#### Step 0 - Definitions #####

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

import os
import sys
import pandas as pd

# data_final_folder = "/data/horse/ws/jaew523d-large_ontix_project/large_sc_data/"
# data_final_folder = "/home/ewald/Github/autoencodix_package/results/large_ontix_save/third_run_e250/"
data_final_folder = "/data/horse/ws/jaew523d-large_ontix_project/large_sc_data_taskRun/"

results_folder = "/data/horse/ws/jaew523d-large_ontix_project/results/large_varix_save/baseline_run_e250/"

llm_ontology_folder = "/data/horse/ws/jaew523d-large_ontix_project/final_ontologies/task-oriented/"
# llm_ontology_folder = "./data/llm_ontologies/final_ontologies/task-oriented/"

ont_from_cli = sys.argv[1]  # "chatgpt_ontology__", "custom_ontology__"

file_holdout = os.path.join(data_final_folder, f"census-acxcontainer_holdout.pkl")
file_tuning = os.path.join(data_final_folder, f"census-acxcontainer_tune.pkl")
final_varix_model_file = os.path.join(results_folder, f"large_varix_final_model_{ont_from_cli}.pkl")


#### Step 1 - Load and predict on holdout set #####
import pickle
import autoencodix as acx
# from autoencodix.configs.ontix_config import OntixConfig

print("Preparing holdout data for feature size per ontology ...")
ont_files = [
	# Order from Latent Dim -> Hidden Dim -> Input Dim
	os.path.join(llm_ontology_folder, f"{ont_from_cli}ensembl_level1.tsv"),
	os.path.join(llm_ontology_folder, f"{ont_from_cli}ensembl_level2.tsv"),
	]

ont_lvl2 = pd.read_csv(ont_files[1], sep='\t', usecols=[0], header=None)
ont_lvl2.columns = ['feature_id']

print("Loading tune set to pre-fit PCA reducer ...")
with open(file_tuning, "rb") as f:
	acx_container = pickle.load(f) # Later overwritten to save RAM

acx_container = keep_features_from_acxcontainer(acx_container, ont_lvl2['feature_id'].values)
## Pre-fit PCA reducer 
from sklearn.decomposition import PCA

dim = int(ont_from_cli.split("_")[0].replace("Dim", ""))

pca = PCA(n_components=dim)
pca.fit(acx_container.test._to_df())

# Save the pre-fitted PCA reducer as pickle 
pca_reducer_file = os.path.join(results_folder, f"{ont_from_cli}pca_reducer.pkl")
with open(pca_reducer_file, "wb") as f:
	pickle.dump(pca, f)

print("Loading holdout data ...")
with open(file_holdout, "rb") as f:
	acx_container = pickle.load(f)

acx_container = keep_features_from_acxcontainer(acx_container, ont_lvl2['feature_id'].values)


# #####  Expand metadata with cell type tasks ###################
from flask import json

# with open("/home/ewald/Github/autoencodix_package/data/llm_ontologies/gemini_celltype_tasks2.json", "r") as f:
with open("/data/horse/ws/jaew523d-large_ontix_project/gemini_celltype_tasks2.json", "r") as f:
    gemini_celltype_tasks2 = json.load(f)


# ## Downsample holdout set for faster prediction (optional)
# acx_container.test.data = acx_container.test.data[:50000, :]
# acx_container.test.sample_ids = acx_container.test.sample_ids[:50000]
# acx_container.test.metadata = acx_container.test.metadata.loc[acx_container.test.sample_ids, :]
# print(f"Holdout test set size after downsampling: {acx_container.test.data.shape[0]} samples.")

print("Loading trained model ...")
# loaded_ontix = acx.Ontix.load(file_path=final_ontix_model_file)
loaded_varix = acx.Varix.load(file_path=final_varix_model_file)
loaded_varix._trainer._config.save_vram = True  # Enable memory saving for prediction on holdout set

print("Predicting on holdout data ...")
result = loaded_varix.predict(
	data= acx_container
)

#### Step 2 - create plots ####

# UMAP representations of latent space
params_umap = ["sex", "is_diseased", "assay", "tissue_general", "high_level_stage_name"]
# params_umap = ["high_level_stage_name"]
# params_umap = ["assay"]
loaded_varix.visualizer.show_latent_space(
	result=loaded_varix.result,
	plot_type='2D-scatter',
	param=params_umap,
	split='test',
	n_downsample=20000)
# Ridgeline plots of latent space
# params_ridge = ["sex","disease", "cell_type", "tissue_general", "development_stage"]
params_ridge = params_umap + list(gemini_celltype_tasks2.keys())[0:5] # Use the cell type tasks defined in the JSON file
# params_ridge = ["high_level_stage_name"]
# params_ridge.append("high_level_stage_name")

loaded_varix.visualizer.show_latent_space(
	result=loaded_varix.result,
	plot_type='Ridgeline',
	param=params_ridge,
	split='test',
	n_downsample=20000)
# # Heatmap representations of latent space
# params_heatmap = ["tissue_general", "development_stage", "sex", "disease"]
# params_heatmap = ["high_level_stage_name"]
# params_heatmap = ["assay"]
params_heatmap = params_umap

loaded_varix.visualizer.show_latent_space(
	result=loaded_varix.result,
	plot_type='Clustermap',
	param=params_heatmap,
	split='test',
	n_downsample=20000)

# ## Focused plots for specific metadata values
# # UMAP for disease "cystic fibrosis" vs. "COVID-19" vs. "normal" vs. others
# params_umap_disease = ["disease"]
# loaded_ontix.visualizer.show_latent_space(
# 	result=loaded_ontix.result,
# 	plot_type='2D-scatter',
# 	param=params_umap_disease,
# 	focus_labels=["cystic fibrosis", "COVID-19", "normal"],
# 	split='test',
# 	n_downsample=20000)
# # UMAP for tissue_general "lung" vs. "brain" vs. "liver" vs. others
# params_umap_tissue = ["tissue_general"]
# loaded_ontix.visualizer.show_latent_space(
# 	result=loaded_ontix.result,
# 	plot_type='2D-scatter',
# 	param=params_umap_tissue,
# 	focus_labels=["lung", "brain", "liver"],
# 	split='test',
# 	n_downsample=20000)

# # Ridgeline for disease "cystic fibrosis" vs. "COVID-19" vs. "normal" vs. others
# params_ridge_disease = ["disease"]
# loaded_ontix.visualizer.show_latent_space(
# 	result=loaded_ontix.result,
# 	plot_type='Ridgeline',
# 	param=params_ridge_disease,
# 	focus_labels=["cystic fibrosis", "COVID-19", "normal"],
# 	split='test',
# 	n_downsample=20000)
# # Ridgeline for tissue_general "lung" vs. "brain" vs. "liver" vs. others
# params_ridge_tissue = ["tissue_general"]
# loaded_ontix.visualizer.show_latent_space(
# 	result=loaded_ontix.result,
# 	plot_type='Ridgeline',
# 	param=params_ridge_tissue,
# 	focus_labels=["lung", "brain", "liver"],
# 	split='test',
# 	n_downsample=20000)

# # Ridgeline for cell_type "alternatively activated macrophage" vs. "inflammatory macrophage" vs. others
# params_ridge_celltype = ["cell_type"]
# loaded_ontix.visualizer.show_latent_space(
# 	result=loaded_ontix.result,
# 	plot_type='Ridgeline',
# 	param=params_ridge_celltype,
# 	focus_labels=["alternatively activated macrophage", "inflammatory macrophage"],
# 	split='test',
# 	n_downsample=20000)

# # Ridgeline for cell_type "CD4-positive helper T cell" vs. "CD8-positive, alpha-beta cytotoxic T cell" vs. "regulatory T cell" vs. others
# params_ridge_celltype2 = ["cell_type"]
# loaded_ontix.visualizer.show_latent_space(
# 	result=loaded_ontix.result,
# 	plot_type='Ridgeline',
# 	param=params_ridge_celltype2,
# 	focus_labels=["CD4-positive helper T cell", "CD8-positive, alpha-beta cytotoxic T cell", "regulatory T cell"],
# 	split='test',
# 	n_downsample=20000)

# # Ridgeline for cell_type "type I muscle cell" vs. "type II muscle cell" vs. others
# params_ridge_celltype3 = ["cell_type"]
# loaded_ontix.visualizer.show_latent_space(
# 	result=loaded_ontix.result,
# 	plot_type='Ridgeline',
# 	param=params_ridge_celltype3,
# 	focus_labels=["type I muscle cell", "type II muscle cell"],
# 	split='test',
# 	n_downsample=20000)


# # Clustermap for disease "cystic fibrosis" vs. "COVID-19" vs. "normal" vs. others
# params_heatmap_disease = ["disease"]
# loaded_ontix.visualizer.show_latent_space(
# 	result=loaded_ontix.result,
# 	plot_type='Clustermap',
# 	param=params_heatmap_disease,
# 	focus_labels=["cystic fibrosis", "COVID-19", "normal"],
# 	split='test',
# 	n_downsample=20000)

# # Clustermap for tissue_general "lung" vs. "brain" vs. "liver" vs. others
# params_heatmap_tissue = ["tissue_general"]
# loaded_ontix.visualizer.show_latent_space(
# 	result=loaded_ontix.result,
# 	plot_type='Clustermap',
# 	param=params_heatmap_tissue,
# 	focus_labels=["lung", "brain", "liver"],
# 	split='test',
# 	n_downsample=20000)

# # Clustermap for cell_type "alternatively activated macrophage" vs. "inflammatory macrophage" vs. "CD4-positive helper T cell" vs. "CD8-positive, alpha-beta cytotoxic T cell" vs. "regulatory T cell" vs. "type I muscle cell" vs. "type II muscle cell" vs. others
# params_heatmap_celltype = ["cell_type"]
# loaded_ontix.visualizer.show_latent_space(
# 	result=loaded_ontix.result,
# 	plot_type='Clustermap',
# 	param=params_heatmap_celltype,
# 	focus_labels=["alternatively activated macrophage", "inflammatory macrophage", "CD4-positive helper T cell", "CD8-positive, alpha-beta cytotoxic T cell", "regulatory T cell", "type I muscle cell", "type II muscle cell"],
# 	split='test',
# 	n_downsample=20000)


#### Step 3 - Evaluate embeddings ####
import sklearn
from sklearn import linear_model
from sklearn.ensemble import RandomForestClassifier
# tasks = ["cell_type", "tissue_general", "development_stage", "sex", "disease"] 
# tasks = list(gemini_celltype_tasks2.keys()) # Use the cell type tasks defined in the JSON file
# tasks.append("high_level_stage_name")
# tasks = ["assay"]
tasks = list(gemini_celltype_tasks2.keys()) + ["tissue_general", "sex", "disease", "high_level_stage_name"]
sklearn.set_config(enable_metadata_routing=True)

sklearn_ml_class = linear_model.LogisticRegression(
							solver="saga",
							n_jobs=-1,
							class_weight="balanced",
							max_iter=500,
) 

sklearn_ml_regression = linear_model.LinearRegression() ## Unused, only classification tasks
own_metric_class = 'roc_auc_ovo'  
own_metric_regression = 'r2' 

loaded_varix.evaluate(
		ml_model_class=sklearn_ml_class, 
		ml_model_regression=sklearn_ml_regression, 
		params= tasks,	
		metric_class = own_metric_class, 
		metric_regression = own_metric_regression, 
		reference_methods = ["PCA"], # No reference methods for tuning
		reference_reducer = {"PCA": pca}, # Use the pre-fitted PCA reducer for the reference method
		split_type = "CV-3",
		top_k_classes = 20,
  		# n_downsample = int(acx_container.train.data.shape[0]*0.5), # Use a subset of the data for faster evaluation
		n_downsample = None, # Use a subset of the data for faster evaluation
		exclude_classes = ["other", "unknown"],
	)

# Test RandomForest as additional model for evaluation
sklearn_ml_class = RandomForestClassifier(
							n_estimators=100,
							n_jobs=-1,
							# min_samples_split=100,
							max_depth=5,
							min_samples_leaf=4,
							class_weight="balanced",
)

loaded_varix.evaluate(
		ml_model_class=sklearn_ml_class, 
		ml_model_regression=sklearn_ml_regression, 
		params= tasks,	
		metric_class = own_metric_class, 
		metric_regression = own_metric_regression, 
		reference_methods = ["PCA"], # No reference methods for tuning
		reference_reducer = {"PCA": pca}, # Use the pre-fitted PCA reducer for the reference method
		split_type = "CV-3",
		top_k_classes = 20,
  		# n_downsample = int(acx_container.train.data.shape[0]*0.5), # Use a subset of the data for faster evaluation
		n_downsample = None, # Use a subset of the data for faster evaluation
		exclude_classes = ["other", "unknown"],
)

#### Step 4 - Save ####
# Save latent embeddings as dataframe
loaded_varix.result.get_latent_df(epoch=-1, split='test').to_parquet(os.path.join(results_folder, f"holdout_latent_{ont_from_cli}.parquet"))

# Save plots
path_plots = os.path.join(results_folder, f"large_varix_holdout_{ont_from_cli}_plots/")
# Create directory if it does not exist
os.makedirs(path_plots, exist_ok=True)
loaded_varix.visualizer.save_plots(
	path=path_plots, which ='all', format ='png'
)

# Save evaluation results
path_eval = os.path.join(results_folder, f"large_varix_holdout_{ont_from_cli}_evaluation.csv")
loaded_varix.result.embedding_evaluation.to_csv(path_eval, index=False)