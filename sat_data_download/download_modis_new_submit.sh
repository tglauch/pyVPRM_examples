#!/bin/bash
#SBATCH --account=mj0143
#SBATCH --job-name=modis_dl
#SBATCH --partition=compute
#SBATCH --output=logs/modis_%A_%a.out
#SBATCH --error=logs/modis_%A_%a.err
#SBATCH --time=04:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4        # threads per job
#SBATCH --mem=8G

source /work/mj0143/b301034/anaconda3/etc/profile.d/conda.sh 
conda activate pyVPRM 

OUTPUT_DIR=/work/mj0143/b301034/Scrapbook_Analysis/Projects/pyVPRM/pyVPRM_examples/sat_data_download/modis_parallel_dl/
TASKFILE=tasklist_tile_year.json 
IDX=$((SLURM_ARRAY_TASK_ID - 1))

TILE=$(jq -r ".[$IDX].tile" "$TASKFILE")
YEAR=$(jq -r ".[$IDX].year" "$TASKFILE")
PRODUCT=$(jq -r ".[$IDX].product" "$TASKFILE")

TOKEN_FILE=token.txt
TOKEN=$(cat "$TOKEN_FILE")

python download_modis_new_parallel.py \
    --tile "${TILE}" --year "${YEAR}" --token "${TOKEN}" \
    --output "${OUTPUT_DIR}" --product "${PRODUCT}" --workers ${SLURM_CPUS_PER_TASK}
