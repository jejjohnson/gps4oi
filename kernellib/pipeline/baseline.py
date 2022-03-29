import torch
import wandb
import numpy as np
from kernellib.logging import config_to_wandb
from kernellib.pipeline.data import run_data_pipeline
from kernellib.pipeline.model import run_model_pipeline
from kernellib.pipeline.results import run_results_pipeline
from kernellib.types import GeoData, Dimensions
from kernellib.models.utils import gp_batch_predict, gp_samples
from tqdm import trange


def run_baseline_pipeline(config):
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
    wandb.init(
        project=config.project,
        entity=config.entity,
        dir=config.server.logs_dir,
        config=config_to_wandb(config),
    )

    print("Getting data...")
    aoi_params, oi_params, ds_obs, ds_oi_grid = run_data_pipeline(config)

    print("Getting model...")
    model, likelihood = run_model_pipeline(config)

    def step(i_time, pbar):

        ind1 = np.where(
            (
                np.abs(ds_obs.time.values - ds_oi_grid.gtime.values[i_time])
                < 2.0 * oi_params.Lt
            )
        )[0]

        n_obs = len(ind1)

        # get observation data
        obs_data = GeoData(
            lat=ds_obs.latitude.values[ind1],
            lon=ds_obs.longitude.values[ind1],
            time=ds_obs.time.values[ind1],
            data=ds_obs.sla_unfiltered.values[ind1],
        )

        # get state data
        state_data = Dimensions(
            lat=ds_oi_grid.fglat.values,
            lon=ds_oi_grid.fglon.values,
            time=ds_oi_grid.gtime.values[i_time],
        )

        state_coords = state_data.coord_vector()
        obs_coords = obs_data.coord_vector()

        # ML MODEL
        train_x = torch.Tensor(obs_coords)
        train_y = torch.Tensor(obs_data.data)

        if config.aoi.smoketest is not None:
            print("Subsetting data!")
            train_x = train_x[:1000]
            train_y = train_y[:1000]

        test_x = torch.Tensor(state_coords)  # [:1000]

        pbar.set_description("Fitting GP Model...")
        model_fitted = model(train_x, train_y)

        pbar.set_description("Predictions...")
        y_mu, y_var = gp_batch_predict(
            model_fitted, likelihood, test_x, config.model.n_batches_pred
        )

        # samples
        pbar.set_description("Drawing Samples...")
        y_samples = gp_samples(model_fitted, likelihood, test_x, config.oi.n_samples)

        return y_mu, y_var, y_samples, n_obs

    for i_time in (pbar := trange(len(ds_oi_grid.gtime))):
        pbar.set_description_str(f"time: {i_time}")

        # get indices where there are observations
        pbar.set_description("Subsetting Data...")

        y_mu, y_var, y_samples, n_obs = step(i_time, pbar)

        # save into data arrays
        pbar.set_description("Putting in Data...")
        ds_oi_grid.gssh_mu[i_time, :, :] = y_mu.reshape(
            ds_oi_grid.lat.size, ds_oi_grid.lon.size
        )
        ds_oi_grid.gssh_var[i_time, :, :] = y_var.reshape(
            ds_oi_grid.lat.size, ds_oi_grid.lon.size
        )
        ds_oi_grid.gssh_samples[:, i_time, :, :] = y_samples.reshape(
            config.oi.n_samples, ds_oi_grid.lat.size, ds_oi_grid.lon.size
        )
        ds_oi_grid.nobs[i_time] = n_obs

        if config.aoi.smoketest is not None:
            break

    nrmse, nrmse_std, psds_score = run_results_pipeline(ds_oi_grid, config, wandb)

    print("Saving statistics...")
    wandb.log({"nrmse": nrmse, "nrmse_std": nrmse_std, "psds": psds_score})
