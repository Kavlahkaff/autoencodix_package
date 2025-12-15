import math, random, yaml

def random_log(min_val, max_val):
    u = random.random()
    return math.exp(math.log(min_val) + u * (math.log(max_val) - math.log(min_val)))

def sample_hyperparams(config_path="configs/search_space.yaml", architecture=None):
    cfg = yaml.safe_load(open(config_path))
    cfg = {
        "epochs": cfg["fixed"]["epochs"],
        "checkpoint_interval": cfg["fixed"]["checkpoint_interval"],
        "loss_reduction": cfg["fixed"]["loss_reduction"],
        "k_filter": random.choice(cfg["search"]["k_filter"]),
        "n_layers": random.choice(cfg["search"]["n_layers"]),
        "enc_factor": random.choice(cfg["search"]["enc_factor"]),
        "latent_dim": random.choice(cfg["search"]["latent_dim"]),
        "learning_rate": random_log(*cfg["search"]["learning_rate"]),
        "weight_decay": random_log(*cfg["search"]["weight_decay"]),
        "batch_size": random.choice(cfg["search"]["batch_size"]),
        "drop_p": random.uniform(*cfg["search"]["drop_p"])
    }

    if architecture == "disentanglix":
        cfg["beta_mi"] = random.choice(*cfg["search"]["beta_mi"])
        cfg["beta_tc"] = random.choice(*cfg["search"]["beta_tc"])
        cfg["beta_dimKL"] = random_log(*cfg["search"]["beta_dimKL"])
    else:
        cfg["beta"]: random_log(*cfg["search"]["beta"])

    return cfg