#!/bin/bash

#SBATCH --job-name=baseoi        # name of job
#SBATCH --account=cli@gpu
#SBATCH --export=ALL
#SBATCH --nodes=1                    # we request one node
#SBATCH --ntasks-per-node=1          # with one task per node (= number of GPUs here)
#SBATCH --gres=gpu:1                 # number of GPUs (1/4 of GPUs)
#SBATCH -C v100-32g          # GPU Partition (32GB)
#SBATCH --cpus-per-task=10           # number of cores per task (1/4 of the 4-GPUs node)
#SBATCH --hint=nomultithread         # hyperthreading is deactivated
#SBATCH --time=10:00:00              # maximum execution time requested (HH:MM:SS)
#SBATCH --output=/linkhome/rech/genige01/uvo53rl/workdir/data/data_challenges/ssh_mapping_2021/logs/slurm/baseline_oi_gpu%j.out    # name of output file
#SBATCH --error=/linkhome/rech/genige01/uvo53rl/workdir/data/data_challenges/ssh_mapping_2021/logs/slurm/baseline_oi_gpu%j.err     # name of error file (here, in common with the output file)

# get tunneling info
XDG_RUNTIME_DIR=""
node=$(hostname -s)
user=$(whoami)
cluster="dahu_ciment"
port=3008


 
# loading of modules
module load git
module load cuda/10.2

# activate conda environment
source activate oi_torch_gpu_py39

# go to appropriate directory
cd ${WORK}/projects/gps4oi
export PYTHONPATH=${WORK}/projects/gps4oi:${PYTHONPATH}

 
# srun python -u scripts/main.py server=jz experiment=baseline model.kernel.kernel_fn="rbf"
srun python -u scripts/main.py server=jz experiment=baseline model.kernel.kernel_fn="matern52"
# srun python -u scripts/main.py aoi.dlon=0.1 aoi.dlat=0.1 server=jz experiment=baseline model.kernel.kernel_fn="matern52"
