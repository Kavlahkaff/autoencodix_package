import math, random, yaml

def random_log(min_val, max_val):
    u = random.random()
    return math.exp(math.log(min_val) + u * (math.log(max_val) - math.log(min_val)))

def sample_hyperparams(config_path="/data/horse/ws/luth474h-autoencodix_synetune/autoencodix_package/benchmarking/configs/search_space.yaml", architecture=None):
    yaml_cfg = yaml.safe_load(open(config_path))  # keep full config

    cfg = {
        "epochs": yaml_cfg["fixed"]["epochs"],
        "checkpoint_interval": yaml_cfg["fixed"]["checkpoint_interval"],
        "loss_reduction": yaml_cfg["fixed"]["loss_reduction"],
        "k_filter": random.choice(yaml_cfg["search"]["k_filter"]),
        "n_layers": random.choice(yaml_cfg["search"]["n_layers"]),
        "enc_factor": random.choice(yaml_cfg["search"]["enc_factor"]),
        "latent_dim": random.choice(yaml_cfg["search"]["latent_dim"]),
        "learning_rate": random_log(*yaml_cfg["search"]["learning_rate"]),
        "weight_decay": random_log(*yaml_cfg["search"]["weight_decay"]),
        "batch_size": random.choice(yaml_cfg["search"]["batch_size"]),
        "drop_p": random.uniform(*yaml_cfg["search"]["drop_p"])
    }

    if architecture == "disentanglix":
        cfg["beta_mi"] = random.choice(yaml_cfg["search"]["beta_mi"])
        cfg["beta_tc"] = random.choice(yaml_cfg["search"]["beta_tc"])
        cfg["beta_dimKL"] = random_log(*yaml_cfg["search"]["beta_dimKL"])
    elif architecture == "varix":
        cfg["beta"] = random_log(*yaml_cfg["search"]["beta"])

    return cfg


if __name__ == "__main__":
    print(sample_hyperparams(architecture="vanillix"))
    print(sample_hyperparams(architecture="disentanglix"))
    print(sample_hyperparams(architecture="varix"))
    print(sample_hyperparams(architecture="ontix"))
