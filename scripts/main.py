import sys, os
from pyprojroot import here

# spyder up to find the local root
local = here(project_files=[".local"])
sys.path.append(str(local))

from omegaconf import OmegaConf, DictConfig
import hydra
from kernellib.pipeline.baseline import run_baseline_pipeline


@hydra.main(config_path="../config/", config_name="main.yaml")
def main(config: DictConfig):

    # load the data from config
    print("Running pipeline...")
    run_baseline_pipeline(config)

    return None


if __name__ == "__main__":
    main()
