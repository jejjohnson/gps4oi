from pathlib import Path
import xarray as xr
from kernellib.dat.l3 import read_l3_data
from kernellib.dat.oi import reformat_oi_output


def load_test_data(data_dir, aoi):
    
    # load alongtrack data
    ds_along_track = read_l3_data(
        data_dir,
        aoi=aoi
    )
    
    return ds_along_track


def run_results_pipeline(config):
    
    # load test data
    print("opening test dataset...")
    ds_along_track = read_l3_data(
        Path(config.test_data_dir).joinpath(config.test_data_filename), 
        config.aoi
    )
    
    # load oi results
    print("opening oi results...")
    ds_oi = xr.open_dataset(
        Path(config.results_dir).joinpath(config.results_filename)
    )
    
    # reformat oi results
    print("reformatting oi results...")
    ref_ds = xr.open_dataset(
        Path(config.ref_data_dir).joinpath(config.ref_data_filename)
    )
    ds_oi = reformat_oi_output(ds_oi, ref_ds)
    
    
    # interpolation
    print("interpolating along track...")
    time_alongtrack, lat_alongtrack, lon_alongtrack, ssh_alongtrack, ssh_map_interp = interp_on_alongtrack(
        ds_oi, 
        ds_along_track,
        aoi=config.aoi,
        is_circle=True
    )
    
    return None
    
    
def run_viz_pipeline(config):
    
    return None