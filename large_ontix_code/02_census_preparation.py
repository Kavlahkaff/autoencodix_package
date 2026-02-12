#### STEP 0 - Definitions #####
import os
import sys
import scanpy
import anndata
# data_folder = "./data/census_chunks/"
# data_folder = "/data/horse/ws/jaew523d-large_ontix_project/census_chunks/"
data_folder = "/data/horse/ws/jaew523d-large_ontix_project/census_chunks_taskRun/"

# data_final_folder = "/data/horse/ws/jaew523d-large_ontix_project/large_sc_data/"
data_final_folder = "/data/horse/ws/jaew523d-large_ontix_project/large_sc_data_taskRun/"

# create data_final_folder if it doesn't exist
if not os.path.exists(data_final_folder):
	os.makedirs(data_final_folder)
# llm_ontology_folder = "./data/llm_ontologies/final_ontologies/"
# llm_ontology_folder = "/data/horse/ws/jaew523d-large_ontix_project/final_ontologies/"
llm_ontology_folder = "/data/horse/ws/jaew523d-large_ontix_project/final_ontologies/task-oriented/"

mock_config_file = "/data/horse/ws/jaew523d-large_ontix_project/large-ontix.yaml"
# ontology_path = "/home/ewald/Github/autoencodix_package/data/hsapdv.obo"
ontology_path = "/data/horse/ws/jaew523d-large_ontix_project/hsapdv.obo"
tasks_file_path = "/data/horse/ws/jaew523d-large_ontix_project/gemini_celltype_tasks2.json"


step_from_cli = sys.argv[1]  # "step1", "step2", "..."
fraction_for_tuning = 0.05
fraction_for_holdout = 0.05
seed = 42

def load_rename_adata(file_path: str) -> anndata.AnnData:
	adata = scanpy.read_h5ad(file_path)
	adata.obs_names = adata.obs.soma_joinid.astype(str)
	adata.var_names = adata.var.feature_id.astype(str)
	return adata

#### STEP 1 - Combine and split census chunks #####
if step_from_cli == "step1":
	print("STEP 1 - Combine and split census chunks")
	import glob 
	import numpy as np

	## Combine all census chunks
	print("Combining census chunks")
	file_paths = glob.glob(os.path.join(data_folder, "*.h5ad"))

	adata_list = [load_rename_adata(fp) for fp in file_paths[:]]  # Limit to first 1000 files for testing
	adata = anndata.concat(adata_list, merge="same")
	adata.obs.drop("soma_joinid", axis=1, inplace=True)
	adata.var.drop("feature_id", axis=1, inplace=True)
 
	##### Expanding Metadata #####
	## Improve developmental stage metadata ##
	

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

	# Load ontology
	onto = pronto.Ontology(ontology_path)
	# Get descendants for each stage
	stage_to_descendants = descendants_by_part_of(onto, stages_list)

	# Optional, print the number of descendants for each stage
	for stage, descendants in stage_to_descendants.items():
		print(f"{stage} has {len(descendants)} descendants")
	
	# Total unique descendants
	all_descendants = set()
	for descs in stage_to_descendants.values():
		all_descendants.update(descs)
	print(f"Total unique descendants: {len(all_descendants)}")

	# Reverse result dictionary for mapping term_id to high_level_stage_id
	term_to_stage = {}
	for stage, descendants in stage_to_descendants.items():
		for desc in descendants:
			term_to_stage[desc] = stage

	# Add top level stages themselves
	for stage in stages_list:
		term_to_stage[stage] = stage

	# Dict for id to name mapping
	onto_names = dict()
	for term_id in term_to_stage.keys():
		onto_names[term_id] = onto.get_term(term_id).name

	# New columns in adata.obs
	adata.obs["high_level_stage_id"] = adata.obs["development_stage_ontology_term_id"].map(term_to_stage)
	adata.obs["high_level_stage_name"] = adata.obs["high_level_stage_id"].map(onto_names)
	# Set NA values to "other"
	adata.obs.fillna({"high_level_stage_id": "other", "high_level_stage_name": "other"}, inplace=True)

	#####  Expand metadata with cell type tasks ####
	from flask import json

	with open(tasks_file_path, "r") as f:
		gemini_celltype_tasks2 = json.load(f)

	# Iterate over tasks and expand metadata with new columns for each task
	for task_name, cell_types in gemini_celltype_tasks2.items():
		# Create a new column for the task, initialized to "other"
		adata.obs[task_name] = "other"
		# For each cell type in the task, set the corresponding rows to the cell type name
		for cell_type in cell_types:
			cell_type_id = cell_type["id"]
			cell_type_name = cell_type["name"]
			adata.obs.loc[
				adata.obs["cell_type_ontology_term_id"] == cell_type_id,
				task_name
			] = cell_type_name
   
	### New column for is_diseased (not normal)
	adata.obs["is_diseased"] = adata.obs["disease"].apply(lambda x: "normal" if x == "normal" else "diseased")
	adata.obs["is_diseased"].value_counts() 
	###### End of metadata expansion #####
 
 
	## Log-normalize the data inplace
	print("Log-normalizing the data")
	scanpy.pp.log1p(adata, copy=False)

	## Split samples in 5% for tuning, 5% as holdout, 90% for training
	print("Splitting data into train, tune, and holdout sets")
	holdout_names, holdout_idx = scanpy.pp.sample(np.array(adata.obs_names), fraction=fraction_for_holdout, copy=True, rng=seed)
	tune_names, tune_idx = scanpy.pp.sample(np.array(adata.obs_names.difference(holdout_names)), fraction=fraction_for_tuning, copy=True, rng=seed)  
	# combine numpy arrays using numpy's union1d (works with numpy.ndarray)
	combined = np.union1d(holdout_names, tune_names)
	train_names = adata.obs_names.difference(combined)

	## Save splits to h5ad files
	print("Saving train, tune, and holdout splits to files")
	scanpy.write(os.path.join(data_final_folder, "census_train_split.h5ad"), adata[train_names])
	scanpy.write(os.path.join(data_final_folder, "census_tune_split.h5ad"), adata[tune_names])
	scanpy.write(os.path.join(data_final_folder, "census_holdout_split.h5ad"), adata[holdout_names])	

#### STEP 2 - Prepare each split for large Ontix training #####
if step_from_cli == "step2":
	print("STEP 2 - Prepare each split for large Ontix training")
	split_from_cli = sys.argv[2]  # "train", "tune", "holdout"
	import scanpy as sc
	# import pandas as pd
	import numpy as np
	from sklearn.model_selection import train_test_split
	from autoencodix.data._numeric_dataset import NumericDataset
	from autoencodix.data._datasetcontainer import DatasetContainer

	from autoencodix.configs.ontix_config import OntixConfig
	import yaml
	from pathlib import Path

	anndata_file = os.path.join(data_final_folder, f"census_{split_from_cli}_split.h5ad")
	adata = sc.read_h5ad(anndata_file)
	n_samples = adata.n_obs
	if split_from_cli in ["train", "tune"]:
		if split_from_cli == "train":
			first_test_size = 0.01 # Must be non-zero since acx requires all splits to be non-empty
			valid_proportion = 0.5 # final train: 99%, valid : 0.5%, test: 0.5%
		if split_from_cli == "tune":
			first_test_size = 0.4
			valid_proportion = 0.9 # in tune: 60%, valid: 36%, test: 4%
		

		train_idx, temp_idx = train_test_split(
			np.arange(n_samples),
			test_size=first_test_size,
			random_state=seed,
		)
		val_idx, test_idx = train_test_split(
			temp_idx,
			test_size=(1 - valid_proportion),  # Remaining proportion goes to test
			random_state=seed,
		)

		# Create split indicators as numpy arrays
		train_split = np.zeros(n_samples, dtype=bool)
		train_split[train_idx] = True

		val_split = np.zeros(n_samples, dtype=bool)
		val_split[val_idx] = True

		test_split = np.zeros(n_samples, dtype=bool)
		test_split[test_idx] = True 

		scconfig = OntixConfig.model_validate(
			yaml.safe_load(Path(mock_config_file).read_text())
		)


		train_dataset = NumericDataset(
			# data=torch.from_numpy(adata.X[train_idx].toarray()),
			data=adata.X[train_idx],
			config=scconfig,
			sample_ids=adata.obs.index[train_idx],
			metadata=adata.obs.loc[adata.obs.index[train_idx],:],
			split_indices=train_split,
			feature_ids=adata.var.index,
		)
		print("train ready")
		test_dataset = NumericDataset(
			# data=torch.from_numpy(adata.X[test_idx].toarray()),
			data=adata.X[test_idx],
			config=scconfig,
			sample_ids=adata.obs.index[test_idx],
			metadata=adata.obs.loc[adata.obs.index[test_idx],:],
			split_indices=test_split,
			feature_ids=adata.var.index,
		)
		print("test ready")

		val_dataset = NumericDataset(
			# data=torch.from_numpy(adata.X[val_idx].toarray()),
			data=adata.X[val_idx],
			config=scconfig,
			sample_ids=adata.obs.index[val_idx],
			metadata=adata.obs.loc[adata.obs.index[val_idx],:],
			split_indices=val_split,
			feature_ids=adata.var.index,
		)
		print("val ready")
		del adata  # Free up memory
		print("adata deleted")
		processed_data = DatasetContainer(train=train_dataset, valid=val_dataset, test=test_dataset)
		print("DatasetContainer ready")
		del train_dataset, val_dataset, test_dataset  # Free up memory
		print("individual datasets deleted")
		## Save the processed data as pickle file
		import pickle
		with open(os.path.join(data_final_folder, f"census-acxcontainer_{split_from_cli}.pkl"), "wb") as f:
			pickle.dump(processed_data, f)

		print("processed data saved")
	elif split_from_cli == "holdout":
		# For holdout, we only need a test split
		test_split = np.ones(n_samples, dtype=bool)  # All samples in holdout are for testing

		scconfig = OntixConfig.model_validate(
			yaml.safe_load(Path(mock_config_file).read_text())
		)

		test_dataset = NumericDataset(
			data=adata.X,
			config=scconfig,
			sample_ids=adata.obs.index,
			metadata=adata.obs.loc[adata.obs.index,:],
			split_indices=test_split,
			feature_ids=adata.var.index,
		)
		print("holdout test ready")
		del adata  # Free up memory
		print("adata deleted")
		processed_data = DatasetContainer(train=None, valid=None, test=test_dataset)
		print("Holdout DatasetContainer ready")
		del test_dataset  # Free up memory
		print("individual datasets deleted")
		## Save the processed data as pickle file
		import pickle
		with open(os.path.join(data_final_folder, f"census-acxcontainer_{split_from_cli}.pkl"), "wb") as f:
			pickle.dump(processed_data, f)

		print("holdout processed data saved")
	else:
		print("Invalid split_from_cli argument. Use 'train', 'tune', or 'holdout'.")
	

