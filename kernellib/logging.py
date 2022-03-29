import omegaconf


def config_to_wandb(config):
    # convert config to dict
    config_ = omegaconf.OmegaConf.to_container(
        config, resolve=False, throw_on_missing=True
    )

    # remove intermediate dictionaries
    config_.pop("load")
    config_.pop("subset")
    return config_
