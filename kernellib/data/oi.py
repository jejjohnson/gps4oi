def reformat_oi_output(ds, ref_ds):
    
    ds = ds.drop(['gtime', 'ng', 'glon2', 'glat2', 'fglon', 'fglat', 'nobs'])
    ds = ds.rename({'gssh_mu': 'sla'})
    
    ref_ds = ref_ds.interp(lon=ds.lon, lat=ds.lat)
    
    ds["ssh"] = ds["sla"] + ref_ds["mdt"]
    
    return ds
    
    
def load_test_filename():
    return "dt_gulfstream_c2_phy_l3_20161201-20180131_285-315_23-53.nc"