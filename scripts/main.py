from multiprocessing.sharedctypes import Value
import sys, os
from pyprojroot import here

# spyder up to find the local root
local = here(project_files=[".local"])
sys.path.append(str(local))

from omegaconf import OmegaConf, DictConfig
import hydra
from kernellib.pipeline.baseline import run_baseline_pipeline
from kernellib.pipeline.opt import run_opt_pipeline


@hydra.main(config_path="../config/", config_name="main.yaml")
def main(config: DictConfig):

    # load the data from config
    print("Running pipeline...")
    if config.exp_type == "loop_noopt":
        run_baseline_pipeline(config)
    elif config.exp_type == "opt":
        run_opt_pipeline(config)
    else:
        raise ValueError(f"Unrecognized experiment: {config.exp_type}")

    return None


if __name__ == "__main__":
    main()
