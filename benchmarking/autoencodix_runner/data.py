from src.autoencodix.configs.default_config import DataConfig, DataInfo


#tcga dataset files
tcga_files = {
    "METH": "/data/horse/ws/luth474h-autoencodix_synetune/data/data_methylation_per_gene_formatted.parquet",
    "RNA": "/data/horse/ws/luth474h-autoencodix_synetune/data/data_mrna_seq_v2_rsem_formatted.parquet",
    "DNA": "/data/horse/ws/luth474h-autoencodix_synetune/data/data_combi_MUT_CNA_formatted.parquet",
    "CLIN": "/data/horse/ws/luth474h-autoencodix_synetune/data/data_clinical_formatted.parquet",
}

#schc dataset files
schc_files = {
    "METH": "/data/horse/ws/luth474h-autoencodix_synetune/data/scATAC_human_cortex_formatted.parquet",
    "RNA": "/data/horse/ws/luth474h-autoencodix_synetune/data/scRNA_human_cortex_formatted.parquet",
    "CLIN": "/data/horse/ws/luth474h-autoencodix_synetune/data/scATAC_human_cortex_clinical_formatted.parquet",
}


def create_data_config(dataset: str, modalities: list[str]) -> DataConfig:
    if dataset == "tcga":
        files = tcga_files
    else:
        files = schc_files

    data_info = {}

    for modality in modalities:
        if modality == "DNA" and dataset == "schc":
            continue
        if modality == "CLIN":
            data_info[modality] = DataInfo(
                file_path=files[modality],
                data_type="ANNOTATION"
            )
        else:
            data_info[modality] = DataInfo(
                file_path=files[modality]
            )

    return DataConfig(data_info=data_info)