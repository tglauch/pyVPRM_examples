# pyVPRNN Example: DK-RCW Flux Partitioning

This example demonstrates the **pyVPRNN** flux-partitioning pipeline end to end, using the Danish deciduous forest site **DK-RCW** as a worked example. DK-RCW is a good test site for this purpose because of its relatively small, well-constrained footprint.

## 1. Prerequisites

Before running the pipeline, you'll need to collect three things:

**Land cover map**
Download the Copernicus Global Land Cover map and place it under `./data/`: https://zenodo.org/records/3939050/files/PROBAV_LC100_global_v3.0.1_2019-nrt_Discrete-Classification-map_EPSG-4326.tif?download=1

**ERA5 access token**
Request a token for the [Destination Earth platform](https://platform.destine.eu/) (used to query ERA5 meteorological data), and export it as an environment variable:
```bash
export EARTHDATAHUB_PAT="your-token-here"
```

**Flux tower data**
Download the DK-RCW data from the [FLUXNET Data Explorer](https://www.keenangroup.info/fluxnet-data-explorer/) and place it under `./siteData/`, e.g.: siteData/ICOS_DK-RCW_FLUXNET_2023-2024_v1.3_r1/ICOS_DK-RCW_*

Once all three are in place, update `config.yaml` to point at them (site ID, land cover path, and site data path).

## 2. Running the Pipeline

```bash
# 1. Generate the training dataset (satellite indices, footprints, meteorology)
python run_vprm_pipeline.py --config config.yaml

# 2. Train the model and produce a quick-look evaluation
python train_and_evaluate.py --config config.yaml
```

> **Note:** by default, `config.yaml` limits training to a few minutes so the pipeline runs quickly end to end as a smoke test. For a real training run, increase `training.max_runtime_hours`.

## 3. Compute Requirements

**Dataset generation** (`run_vprm_pipeline.py`) is both CPU- and memory-intensive:
- Set `compute.n_cpus` to a larger value to parallelize and speed things up.
- Generating hourly footprints on the satellite grid is memory-intensive — for larger sites, use an HPC node with **>128 GB RAM**.

**Training** (`train_and_evaluate.py`) benefits substantially from a GPU:
- Run on a node with an available GPU for reasonably fast neural network training.

## Related

- Main package: [pyVPRM](https://github.com/tglauch/pyVPRM)
- Method paper: [Glauch et al. (2025), *Geoscientific Model Development*](https://gmd.copernicus.org/articles/18/4713/2025/)
