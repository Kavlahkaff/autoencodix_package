import torch
import warnings
import pandas as pd
from typing import List, Dict, Any, Optional, Union
from autoencodix.base._base_dataset import BaseDataset
from torch.profiler import record_function
from autoencodix.configs.default_config import DefaultConfig
import numpy as np


class MultiModalDataset(BaseDataset, torch.utils.data.Dataset):  # type: ignore
    """Handles multiple datasets of different modalities.
    Attributes:
        datasets: Dictionary of datasets for each modality.
        n_modalities: Number of modalities.
        sample_to_modalities: Mapping from sample IDs to available modalities.
        sample_ids: List of all unique sample IDs across modalities.
        config: Configuration object.
        data: Data from the first modality (for compatibility).
        feature_ids: Feature IDs (currently None, to be implemented).
        _id_to_idx: Reverse lookup tables for sample IDs to indices per modality.
        paired_sample_ids: List of sample IDs that have data in all modalities.
        unpaired_sample_ids: List of sample IDs that do not have data in all modalities.
    """

    def __init__(self, datasets: Dict[str, BaseDataset], config: DefaultConfig):
        """
        Initialize the MultiModalDataset.
        Args:
            datasets: Dictionary of datasets for each modality.
            config: Configuration object.
        """
        self.datasets = datasets
        self.modalities = list(datasets.keys())
        self.n_modalities = len(self.modalities)
        self.sample_to_modalities = self._build_sample_map()
        self.sample_ids: List[Any] = list(self.sample_to_modalities.keys())
        self.config = config
        self.data = next(iter(self.datasets.values())).data
        self.feature_ids = None  # TODO
        # Build reverse lookup tables once
        for ds_name, ds in self.datasets.items():
            if ds.sample_ids is None:
                raise ValueError(f"There are no sample_ids for {ds_name}")
        self._id_to_idx = {
            mod: {sid: idx for idx, sid in enumerate(ds.sample_ids)}  # type: ignore
            for mod, ds in self.datasets.items()
        }
        self.paired_sample_ids = self._get_paired_sample_ids()
        self.unpaired_sample_ids = list(
            set(self.sample_ids) - set(self.paired_sample_ids)
        )

    def _to_df(self, modality: Optional[str] = None) -> pd.DataFrame:
        """Convert the dataset to a pandas DataFrame.
        Returns:
            DataFrame representation of the dataset
        """
        if modality is None:
            all_modality = self.modalities
        else:
            all_modality = [modality]
        df_all = pd.DataFrame()
        for modality in all_modality:
            if modality not in self.datasets:
                raise ValueError(f"Unknown modality: {modality}")
            ds = self.datasets[modality]
            if isinstance(ds.data, torch.Tensor):
                df = pd.DataFrame(
                    ds.data.numpy(), columns=ds.feature_ids, index=ds.sample_ids
                )
            elif isinstance(ds.data, list):
                # Handle image modality
                # Get the list of tensors
                tensor_list = ds.data
                if not isinstance(tensor_list[0], torch.Tensor):
                    raise TypeError(
                        f" Image List is not a List[torch.Tensor], but a {type(tensor_list[0])} and cannot be converted to DataFrame."
                    )
                rows = [
                    (
                        t.flatten().cpu().numpy()
                        if isinstance(t, torch.Tensor)
                        else t.flatten()
                    )
                    for t in tensor_list
                ]
                df = pd.DataFrame(
                    rows,
                    index=ds.sample_ids,
                    columns=["Pixel_" + str(i) for i in range(len(rows[0]))],
                )
            else:
                raise TypeError(
                    f"Data is not a torch.Tensor or image data, but a {type(ds.data)} and cannot be converted to DataFrame."
                )
            df = df.add_prefix(f"{modality}_")
            if df_all.empty:
                df_all = df
            else:
                df_all = pd.concat([df_all, df], axis=1, join="inner")
        return df_all

    def _build_sample_map(self):
        sample_to_mods = {}
        for modality, dataset in self.datasets.items():
            for sid in dataset.sample_ids:
                sample_to_mods.setdefault(sid, set()).add(modality)
        return sample_to_mods

    def _get_paired_sample_ids(self):
        all_mods = set(self.modalities)
        return [
            sid for sid, mods in self.sample_to_modalities.items() if mods == all_mods
        ]

    def __len__(self):
        return len(self.paired_sample_ids)

    def __getitem__(self, idx: Union[int, str]):
        sid = self.paired_sample_ids[idx] if isinstance(idx, int) else idx
        out = {"sample_id": sid}
        for mod in self.modalities:
            idx_map = self._id_to_idx.get(mod)
            if idx_map is None or sid not in idx_map:  # missing modality
                out[mod] = None
                continue
            _, data, _ = self.datasets[mod][idx_map[sid]]
            out[mod] = data
        return out

    @property
    def is_fully_paired(self) -> bool:
        """Returns True if all samples are fully paired across all modalities (no unpaired samples)."""
        return len(self.unpaired_sample_ids) == 0


class CoverageEnsuringSampler(torch.utils.data.Sampler):  # type: ignore
    """
    Sampler that ensures all samples are seen at least once per epoch for each modality.
    Attributes:
        dataset: The MultiModalDataset to sample from.
        paired_ids: List of sample IDs that have data in all modalities.
        unpaired_ids: List of sample IDs that do not have data in all modalities.
        batch_size: Number of samples per batch.
        paired_ratio: Ratio of paired samples in each batch.
        modality_samples: Dictionary mapping each modality to its list of sample IDs.
    """

    def __init__(
        self, multimodal_dataset: MultiModalDataset, paired_ratio=0.5, batch_size=64
    ):
        """
        Initialize the sampler.
        Args:
            multimodal_dataset: The MultiModalDataset to sample from.
            paired_ratio: Ratio of paired samples in each batch.
            batch_size: Number of samples per batch.
        """
        self.dataset = multimodal_dataset
        self.paired_ids = multimodal_dataset.paired_sample_ids
        self.unpaired_ids = multimodal_dataset.unpaired_sample_ids
        self.batch_size = batch_size
        self.paired_ratio = paired_ratio
        total_paired = len(self.paired_ids)
        total_unpaired = len(self.unpaired_ids)
        if total_paired == 0:
            self.paired_ratio = 0.0
        elif total_unpaired == 0:
            self.paired_ratio = 1.0
        else:
            # Use requested ratio, but ensure we have enough samples
            max_possible_paired = total_paired / (total_paired + total_unpaired)
            self.paired_ratio = min(paired_ratio, max_possible_paired)
        # Build modality-specific sample lists as numpy arrays for faster choice
        self.modality_samples = {
            modality: np.array(multimodal_dataset.datasets[modality].sample_ids)
            for modality in multimodal_dataset.modalities
        }

    def __iter__(self):
        coverage_batches = self._generate_coverage_batches()
        random_batches = self._generate_random_batches(coverage_batches)
        all_batches = coverage_batches + random_batches
        for batch in all_batches:
            if len(batch) > 1:
                yield batch
            elif len(batch) == 1:
                current_sample = batch[0]
                candidate_pool = (
                    np.concatenate((self.paired_ids, self.unpaired_ids))
                    if self.unpaired_ids
                    else np.array(self.paired_ids)
                )
                candidate_pool = candidate_pool[candidate_pool != current_sample]
                if len(candidate_pool) == 0:
                    raise ValueError(
                        "Cannot form a batch of size > 1 because the dataset contains "
                        "only a single unique sample. To proceed, use a larger sample "
                        "Not this case should not happen, probably something is very odd with your data size "
                    )
                sample_to_add = np.random.choice(candidate_pool)
                batch.append(sample_to_add)
                warnings.warn(
                    "Your combination of batch_size and number of samples whil create a batch of len 1, this will fail all model with a BatchNorm Layer,chose another batch_size to avoid this. We handled this by adding random samples from your data to this 'problem' batch to the current batch. This is an extremely rare case, for our Custom Sampler for unpaired XModalix we don't support this."
                )
                yield batch

    def _generate_coverage_batches(self):
        """Generate batches that ensure all samples are covered
        Returns:
        List of batches ensuring coverage of all samples
        """
        coverage_batches = []
        # Use sets for uncovered samples per modality for O(1) checks and pops
        uncovered_per_mod = {
            mod: set(samples) for mod, samples in self.modality_samples.items()
        }
        num_mods = len(self.modality_samples)
        paired_np = (
            np.array(self.paired_ids) if self.paired_ids else np.empty(0, dtype=object)
        )
        unpaired_np = (
            np.array(self.unpaired_ids)
            if self.unpaired_ids
            else np.empty(0, dtype=object)
        )

        while any(uncovered_per_mod.values()):
            batch = []
            batch_set = set()  # Track unique samples in current batch
            for mod in self.modality_samples:
                uncovered_set = uncovered_per_mod[mod]
                if uncovered_set:
                    take = min(
                        len(uncovered_set),
                        (self.batch_size - len(batch)) // num_mods,
                    )
                    # Select random samples from uncovered set
                    selected = np.random.choice(
                        list(uncovered_set), size=take, replace=False
                    )
                    # Filter out any that might already be in batch (unlikely but safe)
                    selected = [s for s in selected if s not in batch_set]
                    batch.extend(selected)
                    batch_set.update(selected)
                    # Remove selected from uncovered
                    uncovered_set.difference_update(selected)

            # Fill remaining batch slots with random samples, avoiding duplicates
            while len(batch) < self.batch_size:
                candidate_pool = None
                if (
                    len(batch) < self.batch_size * self.paired_ratio
                    and len(paired_np) > 0
                ):
                    candidate_pool = paired_np[~np.isin(paired_np, list(batch_set))]
                elif len(unpaired_np) > 0:
                    candidate_pool = unpaired_np[~np.isin(unpaired_np, list(batch_set))]

                if candidate_pool is None or len(candidate_pool) == 0:
                    # If no unique candidates, allow repeats from appropriate pool
                    if (
                        len(batch) < self.batch_size * self.paired_ratio
                        and len(paired_np) > 0
                    ):
                        candidate_pool = paired_np
                    elif len(unpaired_np) > 0:
                        candidate_pool = unpaired_np
                    else:
                        break

                sample = np.random.choice(candidate_pool)
                batch.append(sample)
                batch_set.add(sample)

            if len(batch) > self.batch_size:
                batch = batch[: self.batch_size]
            if batch:
                coverage_batches.append(batch)
        return coverage_batches

    def _generate_random_batches(self, coverage_batches: List[Any]):
        """Generate additional random batches to fill the epoch
        Args:
            coverage_batches: Batches already generated to ensure coverage
        Returns:
            List of additional random batches
        """
        total_samples = len(self.paired_ids) + len(self.unpaired_ids)
        covered_samples = sum(len(batch) for batch in coverage_batches)
        remaining_samples = max(0, total_samples - covered_samples)
        random_batches = []
        num_random_batches = remaining_samples // self.batch_size
        paired_np = (
            np.array(self.paired_ids) if self.paired_ids else np.empty(0, dtype=object)
        )
        unpaired_np = (
            np.array(self.unpaired_ids)
            if self.unpaired_ids
            else np.empty(0, dtype=object)
        )
        for _ in range(num_random_batches):
            batch = []
            # Add paired samples
            paired_needed = int(self.batch_size * self.paired_ratio)
            if paired_needed > 0 and len(paired_np) > 0:
                paired_samples = np.random.choice(
                    paired_np,
                    size=min(paired_needed, len(paired_np)),
                    replace=True,
                )
                batch.extend(paired_samples)
            # Add unpaired samples
            unpaired_needed = self.batch_size - len(batch)
            if unpaired_needed > 0 and len(unpaired_np) > 0:
                unpaired_samples = np.random.choice(
                    unpaired_np,
                    size=min(unpaired_needed, len(unpaired_np)),
                    replace=True,
                )
                batch.extend(unpaired_samples)
            if batch:
                random_batches.append(batch)
        return random_batches

    def __len__(self):
        total_samples = len(self.paired_ids) + len(self.unpaired_ids)
        return max(total_samples // self.batch_size, len(self.modality_samples))


def create_multimodal_collate_fn(multimodal_dataset: MultiModalDataset):
    """
    Factory function to create a collate function with access to the dataset.
    This allows us to get metadata and original indices.
    Args:
        multimodal_dataset: The multimodal dataset
    Returns:
        A collate function for DataLoader
    """

    def collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
        if not batch:
            return {}
        result = {}
        modalities = multimodal_dataset.modalities
        class_col = multimodal_dataset.config.class_param
        for modality in modalities:
            dataset = multimodal_dataset.datasets[modality]
            has_metadata = class_col and hasattr(dataset, "metadata")
            # Collect only for samples with this modality
            relevant_samples = [s for s in batch if s.get(modality) is not None]
            if not relevant_samples:
                raise ValueError(f"Modality {modality} has no data in batch")
            data_list = [s[modality] for s in relevant_samples]
            sample_ids = [s["sample_id"] for s in relevant_samples]
            sampled_index = [
                multimodal_dataset._id_to_idx[modality].get(s["sample_id"], None)
                for s in relevant_samples
            ]
            if has_metadata:
                class_labels: List[str] = [
                    dataset.metadata.at[s["sample_id"], class_col]
                    for s in relevant_samples
                ]
            else:
                class_labels = [None] * len(relevant_samples)
            result[modality] = {
                "data": torch.stack(data_list),
                "sample_ids": sample_ids,
                "sampled_index": sampled_index,
                "class_labels": class_labels,  # List; convert to tensor if needed for loss
            }
        return result

    return collate_fn
