import torch
import numpy as np
import scipy
from autoencodix.base._base_autoencoder import BaseAutoencoder
from ._model_output import ModelOutput
import torch.nn as nn
import pandas as pd
from autoencodix.configs.default_config import DefaultConfig
from autoencodix.modeling._captum_forward import CaptumForward
from captum.attr import (
    LRP,
    DeepLiftShap,
    GradientShap,
    IntegratedGradients,
    Lime,
    LimeBase,
)

class FeatureImportanceExplainer:
    """
    More complete version with:
      - method selection (DeepLiftShap, IG, etc.)
      - baseline construction (mean / random / grouped)
      - subset sampling
      - reproducible randomness
    """

    def __init__(
        self,
        adata_ACX,
        model,
        method: str = "DeepLiftShap",
        n_subset: int = 100,
        seed_int: int = 12,
        baseline_type: str = "mean",        # ["mean", "random"]
        baseline_group: str = "all",        # "all" or obs_col category
        obs_col: str = None,       # column in .obs for grouping
    ):
        super(FeatureImportanceExplainer, self).__init__()
        self.adata_ACX = adata_ACX
        self.model = model
        self.latent_dim = model.config.latent_dim

        self.method = method
        if self.method not in {"DeepLiftShap", "IntegratedGradients"}:
            raise ValueError(
                f"Invalid method {method}."
        )
        self.n_subset = n_subset
        self.seed_int = seed_int
        self.baseline_type = baseline_type
        self.baseline_group = baseline_group
        self.obs_col = obs_col

        torch.manual_seed(seed_int)
        np.random.seed(seed_int)
        
    def explain(self):
        adata_ACX = self.adata_ACX
        gene_names = adata_ACX.var_names
        inputs, baselines = return_inputs_baseline(adata_ACX,self.baseline_group,self.baseline_type)
        indices_keep = np.random.choice(
            inputs.shape[0], size=self.n_subset, replace=False
            )
        #for latent_dim in range(self.latent_dim):
        all_attr = []
        for latent_dim in range(self.latent_dim):
            cp_forward_dim = CaptumForward(model=self.model,dim=latent_dim                        )
            if self.method == "DeepLiftShap":
                cp_explainer = DeepLiftShap(cp_forward_dim)
            if self.method == "IntegratedGradients":
                cp_explainer = IntegratedGradients(cp_forward_dim)
            attributions, convergence = cp_explainer.attribute(
                    inputs=inputs[indices_keep].float(),
                    baselines=baselines[indices_keep].float(),
                    return_convergence_delta=True
                    )
            avg_abs_attributions = attributions.abs().mean(dim=0)
            all_attr.append(avg_abs_attributions.detach().cpu())
        attr_matrix = torch.stack(all_attr).T.numpy()
        df_attributions = pd.DataFrame(
            attr_matrix,
            index=list(gene_names),
            columns=[f"latent_dimension_{i}" for i in range(self.latent_dim)],
    )
        
          
        return  df_attributions

def return_inputs_baseline(input_adata,baseline_group,baseline_type):
    inputs = torch.tensor(
        input_adata.X.toarray()
        if scipy.sparse.issparse(input_adata.X)
        else input_adata.X
    )
    if baseline_group == "all":
        if baseline_type == "mean":
            baseline_mean = inputs.mean(axis=0)  # gene_means
            baselines = torch.tensor(np.tile(baseline_mean, (inputs.shape[0], 1)))
        if baseline_type == "random_sample":
            baseline_random = inputs[torch.randint(0, inputs.size(0), (1,)).item()]
            baselines = torch.tensor(np.tile(baseline_random, (inputs.shape[0], 1)))
    else:
        input_adata_filtered = input_adata[input_adata.obs[obs_col] == baseline_group]
        inputs_filtered = torch.tensor(
            input_adata_filtered.X.toarray()
            if scipy.sparse.issparse(input_adata_filtered.X)
            else input_adata_filtered.X
        )
        if baseline_type == "mean":
            baseline_mean = inputs_filtered.mean(axis=0)  # gene_means
            baselines = torch.tensor(np.tile(baseline_mean, (inputs.shape[0], 1)))
        if baseline_type == "random_sample":
            baseline_random = inputs_filtered[
                torch.randint(0, inputs_filtered.size(0), (1,)).item()
            ]
            baselines = torch.tensor(np.tile(baseline_random, (inputs.shape[0], 1)))
    return inputs, baselines 