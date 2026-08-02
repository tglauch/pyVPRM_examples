<figure>
<img width="100%" alt="github_logo 001" src="https://github.com/user-attachments/assets/1628353c-802d-4644-8dbc-0a327a72ab24" />
</figure> 

Example scripts demonstrating how to use [`pyVPRM`](https://github.com/tglauch/pyVPRM), covering everything from generating model inputs to full flux partitioning.

## Installation

First, install `pyVPRM` itself (see the [pyVPRM README](https://github.com/tglauch/pyVPRM) for details), then clone this repository:

```bash
git clone https://github.com/tglauch/pyVPRM_examples.git
```

## What's Inside

| Folder | Description |
|---|---|
| [`vprm_predictions`](./vprm_predictions) | Run VPRM predictions of GPP, respiration, and NEE |
| [`wrf_preprocessor`](./wrf_preprocessor) | Generate VPRM input files for use with WRF or ICON |
| [`pyVPRNN_partitioning`](./pyVPRNN_partitioning) | Partition observed NEE into GPP and respiration using the process-informed neural network approach (pyVPRNN) |

Each folder has its own README with setup instructions and requirements specific to that example — start there once you've picked the one relevant to your use case.

## Related

- Main package: [pyVPRM](https://github.com/tglauch/pyVPRM)
- Method paper: [Glauch et al. (2025), *Geoscientific Model Development*](https://gmd.copernicus.org/articles/18/4713/2025/)

Questions? Open an issue, or reach out to **theo.glauch@dlr.de**.

## Repository Size

This repository bundles some example data files to make the examples runnable more easily. As a result, cloning the full repository — including its git history — downloads >100 megabytes. 
