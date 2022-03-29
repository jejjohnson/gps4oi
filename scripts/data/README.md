
```bash
bash dl_data.sh johnsonj@univ-grenoble-alpes.fr yWH9us /linkhome/rech/genige01/uvo53rl/workdir/data/data_challenges/ssh_mapping_2021/raw/netcdf
```

```bash
bash dl_data.sh johnsonj@univ-grenoble-alpes.fr yWH9us /bettik/johnsonj/data/data_challenges/ssh_mapping_2021
```

```bash
python -u scripts/main.py aoi=smoketest experiment=baseline model.kernel.kernel_fn="rbf"
```

* /linkhome/rech/genige01/uvo53rl/workdir/data/data_challenges/ssh_mapping_2021/raw/netcdf


```bash
oarsub -l /nodes=1/gpu=1,walltime=2:00:00 --project pr-data-ocean -p "gpumodel='V100'" -I
```