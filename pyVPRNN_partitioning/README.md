This examples show how the pyVPRNN flux partitioning works exemplarily for the Danish decidious forest site DK-RCW. 

To get things running you need:
- A version of the Copernicus Land Cover map from https://zenodo.org/records/3939050/files/PROBAV_LC100_global_v3.0.1_2019-nrt_Discrete-Classification-map_EPSG-4326.tif?download=1 and stor under ./data 
- A token for the destination Earth platform (https://platform.destine.eu/) to query ERA5 data exported to EARTHDATAHUB_PAT.
- Download flux tower data from https://www.keenangroup.info/fluxnet-data-explorer/ and store it under /siteData. For example siteData/ICOS_DK-RCW_FLUXNET_2023-2024_v1.3_r1/ICOS_DK-RCW_* (DK-RCW is a good test site because of it's relatively small footprint)

After data collection update the config.yaml accordingly and run via 

1. python run_vprm_pipeline.py --config config.yaml
