"""
Converted from Jupyter Notebook: notebook.ipynb
Conversion Date: 2025-12-05T06:12:07.432Z
"""

# ## Profiling and Benchmarking XModalix
# XModalix is rather slow compared to Varix, we assume this is because of the MultiModalDataSet and the CustomSampler.
# Before improving the code we profile and benchmark the components of the XModalix pipeline. We think the following modules could be relevant:
# - MultiModalDataset
#   - which is comprised of NumericDataset, or ImageDataset
# - CoverageEnsuringSampler
# - XModlixTrainer
#   - which calls multiple GeneralTrainers
#     - which is a child of BaseTrainer
#
#
# We should also profile for four different cases:
#   - single cell vs standard tabular
#   - paired vs unpaired
#   - image vs standard tabulas
#   - image vs single cell

import os
import sys
import time

import torch
import numpy as np
import pandas as pd
from torch import nn
from torch.profiler import profile, ProfilerActivity, record_function
import autoencodix as acx
from autoencodix.trainers import _xmodal_trainer, _general_trainer
from autoencodix.base import BaseTrainer, BaseDataset
from autoencodix.base._base_dataset import DataSetTypes
from autoencodix.data import NumericDataset, MultiModalDataset, ImageDataset
from autoencodix.data._multimodal_dataset import (
    create_multimodal_collate_fn,
    CoverageEnsuringSampler,
)


from autoencodix.data._multimodal_dataset import CoverageEnsuringSampler
from autoencodix.utils.example_data import EXAMPLE_MULTI_SC, EXAMPLE_MULTI_BULK
from autoencodix.configs.xmodalix_config import XModalixConfig
from autoencodix.configs.default_config import DataConfig, DataInfo, DataCase

from autoencodix.configs.xmodalix_config import XModalixConfig
from autoencodix.configs.default_config import DataConfig, DataInfo, DataCase
from autoencodix.modeling._imgfast_architecture import ImageVAEFastArchitecture
from autoencodix.modeling._varix_architecture import VarixArchitecture


import torch.utils.benchmark as benchmark


def profile_x_modal_fst():
    clin_file = os.path.join("./data/XModalix-Tut-data/combined_clin_formatted.parquet")
    rna_file = os.path.join("data/XModalix-Tut-data/combined_rnaseq_formatted.parquet")
    img_root = os.path.join("data/XModalix-Tut-data/images/tcga_fake")

    xmodalix_config = XModalixConfig(
        checkpoint_interval=100,
        class_param="CANCER_TYPE",
        epochs=EPOCHS,
        beta=0.1,
        gamma=10,
        delta_class=100,
        delta_pair=300,
        latent_dim=6,
        k_filter=1000,
        batch_size=512,
        learning_rate=0.0005,
        requires_paired=False,
        profiling=True,
        profile_logs="XM_FastArch_TCGA",
        loss_reduction="sum",
        data_case=DataCase.IMG_TO_IMG,
        data_config=DataConfig(
            data_info={
                "img": DataInfo(
                    file_path=img_root,
                    img_height_resize=32,
                    img_width_resize=32,
                    data_type="IMG",
                    scaling="STANDARD",
                    translate_direction="to",
                    pretrain_epochs=0,
                ),
                "rna": DataInfo(
                    file_path=rna_file,
                    data_type="NUMERIC",
                    scaling="STANDARD",
                    pretrain_epochs=0,
                    translate_direction="from",
                ),
                "anno": DataInfo(file_path=clin_file, data_type="ANNOTATION", sep="\t"),
            },
            annotation_columns=["CANCER_TYPE_ACRONYM"],
        ),
    )

    xmodalix = acx.XModalix(
        config=xmodalix_config,
        model_map={
            DataSetTypes.NUM: VarixArchitecture,
            DataSetTypes.IMG: ImageVAEFastArchitecture,
        },
    )
    result = xmodalix.run()
    del xmodalix
    del result


def profile_x_modal_st():
    rna_file = os.path.join("data/XModalix-Tut-data/combined_rnaseq_formatted.parquet")
    img_root = os.path.join("data/XModalix-Tut-data/images/tcga_fake")

    #

    clin_file = os.path.join("./data/XModalix-Tut-data/combined_clin_formatted.parquet")
    rna_file = os.path.join("data/XModalix-Tut-data/combined_rnaseq_formatted.parquet")
    img_root = os.path.join("data/XModalix-Tut-data/images/tcga_fake")

    xmodalix_config = XModalixConfig(
        checkpoint_interval=100,
        class_param="CANCER_TYPE",
        epochs=EPOCHS,
        beta=0.1,
        gamma=10,
        delta_class=100,
        delta_pair=300,
        latent_dim=6,
        k_filter=1000,
        batch_size=512,
        profiling=True,
        profile_logs="XM_StArch_TCGA",
        learning_rate=0.0005,
        requires_paired=False,
        loss_reduction="sum",
        data_case=DataCase.IMG_TO_BULK,
        data_config=DataConfig(
            data_info={
                "img": DataInfo(
                    file_path=img_root,
                    img_height_resize=32,
                    img_width_resize=32,
                    data_type="IMG",
                    scaling="STANDARD",
                    translate_direction="to",
                    pretrain_epochs=0,
                ),
                "rna": DataInfo(
                    file_path=rna_file,
                    data_type="NUMERIC",
                    scaling="STANDARD",
                    pretrain_epochs=0,
                    translate_direction="from",
                ),
                "anno": DataInfo(file_path=clin_file, data_type="ANNOTATION", sep="\t"),
            },
            annotation_columns=["CANCER_TYPE_ACRONYM"],
        ),
    )

    xmodalix = acx.XModalix(config=xmodalix_config)
    result = xmodalix.run()
    del xmodalix
    del result

    #


def profile_x_modal_sc():
    from autoencodix.configs.xmodalix_config import XModalixConfig
    from autoencodix.configs.default_config import DataConfig, DataInfo, DataCase
    from autoencodix.modeling._imgfast_architecture import ImageVAEFastArchitecture
    from autoencodix.modeling._varix_architecture import VarixArchitecture

    rna_file = os.path.join("data/raw/scRNA_human_cortex.h5ad")

    atac_file = os.path.join("data/raw/scATAC_human_cortex.h5ad")

    xmodalix_config = XModalixConfig(
        checkpoint_interval=100,
        epochs=EPOCHS,
        class_param="cell_type",
        beta=0.1,
        gamma=10,
        delta_class=100,
        delta_pair=300,
        latent_dim=6,
        k_filter=1000,
        batch_size=512,
        profiling=True,
        profile_logs="XM_SC_TCGA",
        learning_rate=0.0005,
        requires_paired=False,
        loss_reduction="sum",
        data_case=DataCase.SINGLE_CELL_TO_SINGLE_CELL,
        data_config=DataConfig(
            data_info={
                "atac": DataInfo(
                    file_path=atac_file,
                    data_type="NUMERIC",
                    translate_direction="to",
                    is_single_cell=True,
                    pretrain_epochs=0,
                ),
                "rna": DataInfo(
                    file_path=rna_file,
                    data_type="NUMERIC",
                    scaling="STANDARD",
                    pretrain_epochs=0,
                    translate_direction="from",
                    is_single_cell=True,
                ),
            },
            annotation_columns=["cell_type"],
        ),
    )

    xmodalix = acx.XModalix(config=xmodalix_config)

    result = xmodalix.run()
    del xmodalix
    del result


def run_and_time(fn, label):
    start = time.perf_counter()
    fn()
    end = time.perf_counter()
    elapsed = end - start
    print(f"{label} completed in {elapsed:.3f} seconds")


if __name__ == "__main__":
    
    EPOCHS: int = int(sys.argv[1])
    print("Running XModalix TCGA with Uhler architecture, key: XM-StArch_TCGA")
    run_and_time(profile_x_modal_st, "XM-StArch_TCGA")

    print("Running XModalix TCGA with fast image architecture, key: XM_FastArch_TCGA")
    run_and_time(profile_x_modal_fst, "XM_FastArch_TCGA")

    print("Running XModalix profiling with two SC modalities, key: XM_SC_TCGA")
    run_and_time(profile_x_modal_sc, "XM_SC_TCGA")
