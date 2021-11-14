from pathlib import Path

import geopandas as gpd
import pytest
import rasterio
import datetime

from haversine_distance import DistanceToEdge, DistanceToFeatureEdge
from haversine_distance.errors import PathError, BlockSizesError

#BASE_DIR = Path(__file__).resolve().parent.joinpath('tests/data/bigger_data')
#RASTER = BASE_DIR.joinpath('BWA_buildings_128.tif') #raster with square block sizes
#OUT_RASTER = BASE_DIR.joinpath('BWA-dist-to-dask.tif')

#BASE_DIR = Path(__file__).resolve().parent.joinpath('tests/data')
#RASTER = BASE_DIR.joinpath('block.tif')
#OUT_RASTER = BASE_DIR.joinpath('ABW_DASK__.tif')

BASE_DIR = Path(__file__).resolve().parent.joinpath('tests/data/gtm')
RASTER = BASE_DIR.joinpath('gtm_1_0.tif')
OUT_RASTER = BASE_DIR.joinpath('gtm_dst_to_dask.tif')

if __name__ == "__main__":
	x = DistanceToFeatureEdge(RASTER, OUT_RASTER)
	x.calculate_dask(num_workers=3)
	#x.calculate()
