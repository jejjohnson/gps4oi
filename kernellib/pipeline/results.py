from pathlib import Path
import xarray as xr
from kernellib.data.l3 import read_l3_data
from kernellib.data.oi import reformat_oi_output
from kernellib.utils import ResultsFileName
from kernellib.features.interp import interp_on_alongtrack
from kernellib.types import RMSEBinning
from kernellib.features.stats.base import compute_stats
from kernellib.types import SpectralStats
from kernellib.features.stats.spectral import compute_spectral_scores
from kernellib.viz.temporal import plot_temporal_statistics
from kernellib.viz.spatial import plot_spatial_statistics
from kernellib.viz.psd import plot_psd_score


# def load_test_data(data_dir, aoi):

#     # load alongtrack data
#     ds_along_track = read_l3_data(data_dir, aoi=aoi)

#     return ds_along_track


def run_results_pipeline(ds_oi, config, wandb_logger=None):

    if wandb_logger is not None:
        id = wandb_logger.util.generate_id()
        results_dir = Path(config.server.results_dir).joinpath(f"{id}")

        wandb_logger.log({"data_id": id})
    else:
        results_dir = Path(config.server.results_dir)

    if not results_dir.is_dir():
        results_dir.mkdir()

    # reformat oi results
    print("reformatting oi results...")
    ref_ds = xr.open_dataset(
        Path(config.server.ref_data_dir).joinpath(config.server.ref_data_filename),
    )
    ds_oi = reformat_oi_output(ds_oi, ref_ds)

    print("Saving results...")
    ds_oi.to_netcdf(results_dir.joinpath(f"{config.server.results_filename}"))

    # load test data
    print("opening test dataset...")
    ds_along_track = read_l3_data(
        Path(config.server.test_data_dir).joinpath(config.server.test_data_filename),
        config.aoi,
    )

    # interpolation
    print("interpolating along track...")
    (
        time_alongtrack,
        lat_alongtrack,
        lon_alongtrack,
        ssh_alongtrack,
        ssh_map_interp,
    ) = interp_on_alongtrack(ds_oi, ds_along_track, aoi=config.aoi, is_circle=True)

    filenames = ResultsFileName(filename=config.server.results_filename)

    # stats config
    rmse_binning = RMSEBinning()
    leaderboard_nrmse, leaderboard_nrmse_std = compute_stats(
        time_alongtrack,
        lat_alongtrack,
        lon_alongtrack,
        ssh_alongtrack,
        ssh_map_interp,
        rmse_binning.bin_lon_step,
        rmse_binning.bin_lat_step,
        rmse_binning.bin_time_step,
        results_dir.joinpath(f"stats_{config.server.results_filename}"),
        results_dir.joinpath(f"stats_ts_{config.server.results_filename}"),
    )

    # plot_spatial_statistics(
    #     results_dir.joinpath(f"stats_{config.server.results_filename}")
    # )

    # plot_temporal_statistics(
    #     results_dir.joinpath(f"stats_ts_{config.server.results_filename}")
    # )

    psd_stats = SpectralStats()

    compute_spectral_scores(
        time_alongtrack,
        lat_alongtrack,
        lon_alongtrack,
        ssh_alongtrack,
        ssh_map_interp,
        psd_stats.length_scale,
        psd_stats.delta_x,
        psd_stats.delta_t,
        results_dir.joinpath(f"psd_{config.server.results_filename}"),
    )

    leaderboard_psds_score = plot_psd_score(
        results_dir.joinpath(f"psd_{config.server.results_filename}"), wandb_logger
    )

    return leaderboard_nrmse, leaderboard_nrmse_std, leaderboard_psds_score


def run_viz_pipeline(config):

    return None
