import torch
import warnings
import pandas as pd
from typing import List, Dict, Any, Optional, Union
from autoencodix.base._base_dataset import BaseDataset
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
        self.n_modalities = len(self.datasets.keys())
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
            all_modality = list(self.datasets.keys())
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
                tensor_list = self.datasets[modality].data
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
        return [
            sid
            for sid, mods in self.sample_to_modalities.items()
            if all(mod in mods for mod in self.datasets.keys())
        ]

    def __len__(self):
        return len(self.paired_sample_ids)

    def __getitem__(self, idx: Union[int, str]):
        sid = self.paired_sample_ids[idx] if isinstance(idx, int) else idx
        out = {"sample_id": sid}
        for mod in self.modalities:
            if sid not in self._id_to_idx[mod]:  # missing modality
                out[mod] = None
                continue
            _, data, _ = self.datasets[mod][self._id_to_idx[mod][sid]]
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
        # Build modality-specific sample lists
        self.modality_samples = {}
        for modality in multimodal_dataset.datasets.keys():
            self.modality_samples[modality] = multimodal_dataset.datasets[
                modality
            ].sample_ids

    def __iter__(self):
        coverage_batches = self._generate_coverage_batches()
        random_batches = self._generate_random_batches(coverage_batches)
        all_batches = coverage_batches + random_batches
        for batch in all_batches:
            if len(batch) > 1:
                yield batch
            elif len(batch) == 1:
                current_sample = batch[0]
                candidate_pool = set(self.paired_ids) | set(self.unpaired_ids)
                candidate_pool.discard(current_sample)

                if not candidate_pool:
                    raise ValueError(
                        "Cannot form a batch of size > 1 because the dataset contains "
                        "only a single unique sample. To proceed, use a larger sample "
                        "Not this case should not happen, probably something is very odd with your data size "
                    )
                sample_to_add = np.random.choice(list(candidate_pool))
                batch.append(sample_to_add)
                warnings.warn(
                    "Your combination of batch_size and number of samples whil create a batch of len 1, this will fail all model with a BatchNorm Layer,chose another batch_size to avoid this. We handled this by adding random samples from your data to this 'problem' batch to the current batch. This is an extremely rare case, for our Custom Sampler for unpaired XModalix we don't support this."
                )
                yield batch

    def _generate_coverage_batches(self):
        coverage_batches = []

        # 1. OPTIMIZATION: Create shuffled queues (indices/IDs) upfront.
        # This is O(N) done exactly once.
        queues = {
            mod: np.random.permutation(ids).tolist()
            for mod, ids in self.modality_samples.items()
        }

        # Track active queues (those that still have samples)
        active_mods = list(queues.keys())

        while active_mods:
            batch = []
            batch_set = set()

            # 2. Round-robin greedy fill from queues
            # We don't scan full lists; we just pop from the end of the queue (O(1))
            samples_per_mod = max(1, (self.batch_size - len(batch)) // len(active_mods))

            for mod in list(active_mods):  # iterate copy to allow modification
                # Pop samples from this modality's queue
                take = min(len(queues[mod]), samples_per_mod)

                # Pop from end is O(k) where k is 'take', efficient
                selected = [queues[mod].pop() for _ in range(take)]

                # Deduplicate against current batch
                for s in selected:
                    if s not in batch_set:
                        batch.append(s)
                        batch_set.add(s)

                # If queue empty, remove from active list
                if not queues[mod]:
                    active_mods.remove(mod)

            # 3. Fill remaining slots with RANDOM samples
            # No need for expensive np.isin(dataset). Just pick randomly.
            # Collisions are rare; if they happen, we just try again or accept it.
            while len(batch) < self.batch_size:
                # Fast fallback pool selection
                pool = (
                    self.paired_ids
                    if (
                        len(batch) < self.batch_size * self.paired_ratio
                        and self.paired_ids
                    )
                    else self.unpaired_ids
                )
                if not pool:
                    pool = self.paired_ids or self.unpaired_ids
                if not pool:
                    break

                # Pick one random sample (O(1))
                s = np.random.choice(pool)

                # Only add if unique in this batch
                if s not in batch_set:
                    batch.append(s)
                    batch_set.add(s)
                # If we fail to find a unique sample after X tries, just break or duplicate.
                # In practice for large N, loops are minimal.

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

        for _ in range(num_random_batches):
            batch = []

            # Add paired samples
            paired_needed = int(self.batch_size * self.paired_ratio)
            if paired_needed > 0 and self.paired_ids:
                paired_samples = np.random.choice(
                    self.paired_ids,
                    size=min(paired_needed, len(self.paired_ids)),
                    replace=True,
                )
                batch.extend(paired_samples)

            # Add unpaired samples
            unpaired_needed = self.batch_size - len(batch)
            if unpaired_needed > 0 and self.unpaired_ids:
                unpaired_samples = np.random.choice(
                    self.unpaired_ids,
                    size=min(unpaired_needed, len(self.unpaired_ids)),
                    replace=True,
                )
                batch.extend(unpaired_samples)

            if batch:
                random_batches.append(batch)

        return random_batches

    def __len__(self):
        total_samples = len(self.paired_ids) + len(self.unpaired_ids)
        # return total_samples // self.batch_size
        return max(total_samples // self.batch_size, len(self.modality_samples))


def create_multimodal_collate_fn(multimodal_dataset: MultiModalDataset):
    # 1. PRE-OPTIMIZATION: Build fast lookups
    class_col = multimodal_dataset.config.class_param
    label_maps = {}

    if class_col:
        for mod, ds in multimodal_dataset.datasets.items():
            if hasattr(ds, "metadata") and class_col in ds.metadata:
                # Convert Series to Dict: O(1) lookup speed
                label_maps[mod] = ds.metadata[class_col].to_dict()

    def collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
        if not batch:
            return {}

        result = {}
        modalities = multimodal_dataset.modalities

        for modality in modalities:
            relevant_samples = [s for s in batch if s.get(modality) is not None]
            if not relevant_samples:
                continue

            data_list = [s[modality] for s in relevant_samples]
            sample_ids = [s["sample_id"] for s in relevant_samples]

            # Lookup indices
            idx_map = multimodal_dataset._id_to_idx[modality]
            sampled_index = [idx_map.get(sid) for sid in sample_ids]

            # 2. FAST LOOKUP using dict
            labels = []
            if modality in label_maps:
                mapping = label_maps[modality]
                labels = [mapping.get(sid) for sid in sample_ids]
            else:
                labels = [None] * len(sample_ids)

            result[modality] = {
                "data": torch.stack(data_list),
                "sample_ids": sample_ids,
                "sampled_index": sampled_index,
                "class_labels": labels,
            }

        return result

    return collate_fn
