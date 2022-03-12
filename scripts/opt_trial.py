import sys, os
from pyprojroot import here

# spyder up to find the local root
local = here(project_files=[".local"])
sys.path.append(str(local))

import xarray as xr
import numpy as np
import requests as rq
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from tqdm import tqdm, trange
from kernellib.types import GeoData, Dimensions
from kernellib.preprocessing import create_oi_grid, correct_lon, add_vtime
from kernellib.data import load_data


from omegaconf import OmegaConf, DictConfig
import hydra


@hydra.main(config_path="../config/", config_name="main.yaml")
def main(config: DictConfig):

    # load the data from config
    glon, glat, gtime = hydra.utils.instantiate(config.aoi)

    return None


if __name__ == "__main__":
    main()
