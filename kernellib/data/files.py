from dataclasses import dataclass



def files_factory(system: str):
    
    if system == "cal1":
        return ParamsCal1()
    elif system == "gricad":
        return ParamsGricad()
    elif system == "jeanzay":
        raise NotImplementedError()
    else:
        raise ValueError(f"Unrecognized system: {system}")



@dataclass
class ParamsCal1:
    # train data
    train_data_dir = "/home/johnsonj/data/dc_2021/raw/train/*.nc"
    # test data
    test_data_dir = "/home/johnsonj/data/dc_2021/raw/test/"
    test_data_filename = "dt_gulfstream_c2_phy_l3_20161201-20180131_285-315_23-53.nc"
    
    # ref data
    ref_data_dir = "/home/johnsonj/data/dc_2021/raw/ref/"
    ref_data_filename = "mdt.nc"
    
    # results data
    results_dir = "/home/johnsonj/data/dc_2021/interim"
    # results_filename = "OSE_ssh_mapping_BASELINE.nc"
    # results_filename = "OSE_ssh_mapping_BASELINE_TRAIN_LOOP.nc"
    results_filename = "OSE_ssh_mapping_BASELINE_TRAIN.nc"
    
    # stats data
    stats_dir = "/home/johnsonj/data/dc_2021/interim"
    
    # wandb logs
    logs_dir = "/mnt/meom/workdir/johnsonj/logs/wandb/gps4oi/"
    project = "gps4oi"
    entity = "ige"
    
    

@dataclass
class ParamsGricad:
    # train data
    train_data_dir = "/home/johnsonj/data/dc_2021/raw/train/*.nc"
    # test data
    test_data_dir = "/bettik/johnsonj/data/data_challenges/ssh_mapping_2021/raw/netcdf"
    test_data_filename = "dt_gulfstream_c2_phy_l3_20161201-20180131_285-315_23-53.nc"
    
    # ref data
    ref_data_dir = "/bettik/johnsonj/data/data_challenges/ssh_mapping_2021/raw/netcdf"
    ref_data_filename = "mdt.nc"
    
    # results data
    results_dir = "/bettik/johnsonj/data/data_challenges/ssh_mapping_2021/interim"
    # results_filename = "OSE_ssh_mapping_BASELINE.nc"
    # results_filename = "OSE_ssh_mapping_BASELINE_TRAIN_LOOP.nc"
    results_filename = "OSE_ssh_mapping_BASELINE_TRAIN.nc"
    
    # stats data
    stats_dir = "/bettik/johnsonj/data/data_challenges/ssh_mapping_2021/interim"
    