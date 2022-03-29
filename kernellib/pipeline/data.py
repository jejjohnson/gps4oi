from kernellib.features.aoi import AOIParams
from kernellib.features.oi import OIParams
from kernellib.data.io import load_multiple_data
from kernellib.models.gps import gp_model_factory
from kernellib.models.kernels import kernel_fn_factory, KernelParams
from kernellib.features.aoi import aoi_factory, subset_data
from kernellib.features.oi import oi_params_factory, create_oi_grid
from kernellib.features.time import add_vtime


def run_data_pipeline(config):
    """
    **Preprocessing**

    2. Load Data
    3. Subset Data

    **Model Init**

    * Initialize kernel function
    * Initialize gp params
    * Initialize gp model, likelihood

    **Run Baseline Experiment**

    **Save Results
    """

    # create AOI
    aoi_params = AOIParams().init_from_config(config.aoi)

    # create OI
    oi_params = OIParams().init_from_config(config.oi)

    # load data
    ds_obs = load_multiple_data(config.server.train_data_dir)

    # subset data
    ds_obs = subset_data(ds_obs, aoi_params, oi_params)

    # transform coords
    ds_obs = add_vtime(ds_obs, aoi_params.time_min)

    # initialize oi grid
    ds_oi_grid = create_oi_grid(aoi_params, config.oi.n_samples)

    return aoi_params, oi_params, ds_obs, ds_oi_grid
