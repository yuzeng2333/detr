module purge
module load anaconda3/2023.3
conda activate torch-env
export MASTER_ADDR=localhost
export MASTER_PORT=29500
