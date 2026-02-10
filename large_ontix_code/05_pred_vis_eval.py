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
data_final_folder = "/home/ewald/Github/autoencodix_package/results/large_ontix_save/first_run_e250/"
# data_final_folder = "/home/ewald/Github/autoencodix_package/results/large_ontix_save/second_run_e250/"

# results_folder = "/data/horse/ws/jaew523d-large_ontix_project/results/large_ontix_save/first_run_e250/"
results_folder = "/home/ewald/Github/autoencodix_package/results/large_ontix_save/second_run_e250/"

# llm_ontology_folder = "/data/horse/ws/jaew523d-large_ontix_project/final_ontologies/"
llm_ontology_folder = "./data/llm_ontologies/final_ontologies/"

ont_from_cli = sys.argv[1]  # "chatgpt_ontology__", "custom_ontology__"

file_holdout = os.path.join(data_final_folder, f"census-acxcontainer_holdout.pkl")
final_ontix_model_file = os.path.join(results_folder, f"large_ontix_final_model_{ont_from_cli}.pkl")


#### Step 1 - Load and predict on holdout set #####
import pickle
import autoencodix as acx
# from autoencodix.configs.ontix_config import OntixConfig

print("Loading holdout data ...")
with open(file_holdout, "rb") as f:
	acx_container = pickle.load(f)

print("Preparing holdout data for feature size per ontology ...")
ont_files = [
	# Order from Latent Dim -> Hidden Dim -> Input Dim
	os.path.join(llm_ontology_folder, f"{ont_from_cli}ensembl_level1.tsv"),
	os.path.join(llm_ontology_folder, f"{ont_from_cli}ensembl_level2.tsv"),
	]

ont_lvl2 = pd.read_csv(ont_files[1], sep='\t', usecols=[0], header=None)
ont_lvl2.columns = ['feature_id']

acx_container = keep_features_from_acxcontainer(acx_container, ont_lvl2['feature_id'].values)

#####  Expand metadata with cell type tasks ###################
from flask import json

with open("/home/ewald/Github/autoencodix_package/data/llm_ontologies/gemini_celltype_tasks2.json", "r") as f:
    gemini_celltype_tasks2 = json.load(f)

# Iterate over tasks and expand metadata with new columns for each task
for task_name, cell_types in gemini_celltype_tasks2.items():
	# Create a new column for the task, initialized to "other"
	acx_container.test.metadata[task_name] = "other"
	
	# For each cell type in the task, set the corresponding rows to the cell type name
	for cell_type in cell_types:
		cell_type_id = cell_type["id"]
		cell_type_name = cell_type["name"]
		acx_container.test.metadata.loc[
			acx_container.test.metadata["cell_type_ontology_term_id"] == cell_type_id,
			task_name
		] = cell_type_name
	# # Print the value counts for the new task column
	# print(f"Value counts for {task_name}:")
	# print(acx_container.test.metadata[task_name].value_counts())

#####  Improve development_stage metadata ###################
from collections import defaultdict, deque
from typing import Dict, Iterable, List
import pronto

def descendants_by_part_of(onto: pronto.Ontology, terms: Iterable[str]) -> Dict[str, List[str]]:
    """
    Return {term_id: [all descendant term_ids]} using part_of relationships.
    """

    # Build parent -> direct children map for part_of
    children = defaultdict(list)
    for term in onto.terms():
        if term.obsolete:
            continue
        # term.relationships is a dict: {RelationshipType: set(terms)}
        for rel_type, targets in term.relationships.items():
            if rel_type.id == "part_of":
                for parent in targets:
                    children[parent.id].append(term.id)

    # Collect all descendants
    result: Dict[str, List[str]] = {}
    for root in terms:
        seen = set()
        queue = deque(children.get(root, []))
        while queue:
            child = queue.popleft()
            if child in seen:
                continue
            seen.add(child)
            queue.extend(children.get(child, []))
        result[root] = sorted(seen)
    return result


# New manual high-level stages from Human Development:
stages_list = [
    "HsapDv:0000037", ## fetal stage
    "HsapDv:0000002", ## embryonic stage
    "HsapDv:0000260", ## nursing stage (0-11 months)
    "HsapDv:0000265", ## child stage (1-4yo)
    "HsapDv:0000271", ## juvenile stage (5-14yo)
    "HsapDv:0000268", ## 15-19yo
    "HsapDv:0000237", ## third decade stage
    "HsapDv:0000238", ## fourth decade stage
    "HsapDv:0000239", ## fifth decade stage
    "HsapDv:0000240", ## sixth decade stage
    "HsapDv:0000272", ## 60-79 year-old stage
    "HsapDv:0000095", ## 80 year-old and over stage
    
]
ontology_path = "/home/ewald/Github/autoencodix_package/data/hsapdv.obo"
onto = pronto.Ontology(ontology_path)
result = descendants_by_part_of(onto, stages_list)
# Number of descendants for each stage
for stage_id, descendants in result.items():
	print(f"{stage_id}: {len(descendants)} descendants")

# Total unique descendants
all_descendants = set()
for descs in result.values():
	all_descendants.update(descs)
# print(f"Total unique descendants: {len(all_descendants)}")

# Reverse result dictionary for mapping term_id to high_level_stage_id
reverse_result = {}
for stage_id, descendants in result.items():
	for desc in descendants:
		reverse_result[desc] = stage_id

# Add top level stages themselves
for stage_id in stages_list:
    reverse_result[stage_id] = stage_id
    
onto_names = dict()
for term_id in reverse_result.keys():
	onto_names[term_id] = onto.get_term(term_id).name

acx_container.test.metadata.loc[:, "high_level_stage_id"] = acx_container.test.metadata.loc[:,"development_stage_ontology_term_id"].map(reverse_result)
acx_container.test.metadata.loc[:, "high_level_stage_name"] = acx_container.test.metadata.loc[:,"high_level_stage_id"].map(onto_names)
# Set NA values to "other"
acx_container.test.metadata.loc[:, "high_level_stage_name"] = acx_container.test.metadata.loc[:, "high_level_stage_name"].fillna("other")

## Downsample holdout set for faster prediction (optional)
acx_container.test.data = acx_container.test.data[:50000, :]
acx_container.test.sample_ids = acx_container.test.sample_ids[:50000]
acx_container.test.metadata = acx_container.test.metadata.loc[acx_container.test.sample_ids, :]
print(f"Holdout test set size after downsampling: {acx_container.test.data.shape[0]} samples.")

print("Loading trained model ...")
loaded_ontix = acx.Ontix.load(file_path=final_ontix_model_file)
loaded_ontix._trainer._config.save_vram = True  # Enable memory saving for prediction on holdout set

print("Predicting on holdout data ...")
result = loaded_ontix.predict(
	data= acx_container
)

# #### Step 2 - create plots ####

# UMAP representations of latent space
# params_umap = ["sex", "disease", "cell_type", "tissue_general", "development_stage"]
params_umap = ["high_level_stage_name"]
# params_umap = ["assay"]
loaded_ontix.visualizer.show_latent_space(
	result=loaded_ontix.result,
	plot_type='2D-scatter',
	param=params_umap,
	split='test',
	n_downsample=20000)
# Ridgeline plots of latent space
# params_ridge = ["sex","disease", "cell_type", "tissue_general", "development_stage"]
params_ridge = list(gemini_celltype_tasks2.keys()) # Use the cell type tasks defined in the JSON file
# params_ridge = ["high_level_stage_name"]
params_ridge.append("high_level_stage_name")

loaded_ontix.visualizer.show_latent_space(
	result=loaded_ontix.result,
	plot_type='Ridgeline',
	param=params_ridge,
	split='test',
	n_downsample=20000)
# # Heatmap representations of latent space
# params_heatmap = ["tissue_general", "development_stage", "sex", "disease"]
params_heatmap = ["high_level_stage_name"]
# params_heatmap = ["assay"]
loaded_ontix.visualizer.show_latent_space(
	result=loaded_ontix.result,
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
tasks = list(gemini_celltype_tasks2.keys()) # Use the cell type tasks defined in the JSON file
tasks.append("high_level_stage_name")
# tasks = ["assay"]
sklearn.set_config(enable_metadata_routing=True)

sklearn_ml_class = linear_model.LogisticRegression(
							solver="sag",
							n_jobs=-1,
							class_weight="balanced",
							max_iter=50,
) 

sklearn_ml_regression = linear_model.LinearRegression() ## Unused, only classification tasks
own_metric_class = 'roc_auc_ovo'  
own_metric_regression = 'r2' 

loaded_ontix.evaluate(
	ml_model_class=sklearn_ml_class, 
	ml_model_regression=sklearn_ml_regression, 
	params= tasks,	
	metric_class = own_metric_class, 
	metric_regression = own_metric_regression, 
	reference_methods = ["PCA"], # No reference methods for tuning
	split_type = "CV-2",
	# n_downsample = int(acx_container.test.data.shape[0]*0.01), # Use a subset of the data for faster evaluation
	n_downsample = None, # Use a subset of the data for faster evaluation
	top_k_classes = None, # Keep top 20 classes for classification tasks
	# top_k_classes = 10,
	exclude_classes = ["other"],
)

# # Test RandomForest as additional model for evaluation
# sklearn_ml_class = RandomForestClassifier(
# 							n_estimators=50,
# 							n_jobs=-1,
# 							min_samples_split=100,
# )

# loaded_ontix.evaluate(
# 	ml_model_class=sklearn_ml_class, 
# 	ml_model_regression=sklearn_ml_regression, 
# 	params= tasks,	
# 	metric_class = own_metric_class, 
# 	metric_regression = own_metric_regression, 
# 	reference_methods = ["PCA"], # No reference methods for tuning
# 	split_type = "CV-3",
# 	# n_downsample = int(acx_container.test.data.shape[0]*0.01), # Use a subset of the data for faster evaluation
# 	n_downsample = None, # Use a subset of the data for faster evaluation
# 	top_k_classes = 30, # Keep top 20 classes for classification tasks
# )

#### Step 4 - Save ####

# Save plots
path_plots = os.path.join(results_folder, f"large_ontix_holdout_{ont_from_cli}_plots/")
# Create directory if it does not exist
os.makedirs(path_plots, exist_ok=True)
loaded_ontix.visualizer.save_plots(
	path=path_plots, which ='all', format ='png'
)

# Save evaluation results
path_eval = os.path.join(results_folder, f"large_ontix_holdout_{ont_from_cli}_evaluation.csv")
loaded_ontix.result.embedding_evaluation.to_csv(path_eval, index=False)