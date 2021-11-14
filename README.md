# HaverDst
Module to apply haversine distance (degrees to meters) in all target cells of geotiff

# TO DO
- Loop through windows of exracted buffer and iteratively add points to geodataframe
- Use multiprocessing or dask to loop through windows and process in parallel
- Implement function to deal with a mask rather than the whole global dataset
- Test xarray.rio.clip(from_disk=True) on big dataset

## If no mask
## Checks

### Get GDF of points for pixels in global == 0
1. Open raster and get profile (update NODATA and datatype) for output
2. For each block in raster:
	2.1 - if 0's in block:
		- Get bounds of block - get centroid and buffer by twice dist to top right corner
		- Extract global raster to buffer
		- Loop through windows of extract: If zeros AND ones in window append 0's points to 0 gdf and 1's to 1 gdf
											- add closest pt col to each dataframe and another for haversine dist
											- calculate haversine distance
											- multiply 1's gdf by -1 and merge gdf's
											- rasterise points

										else: double buffer and repeat



### Get GDF of points for pixels in global == 1

## if mask
