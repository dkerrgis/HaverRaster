"""
Tests on bigger data
"""

from pathlib import Path

import geopandas as gpd
import pytest
import rasterio


from haversine_distance import DistanceToEdge
from haversine_distance.errors import PathError, BlockSizesError

BASE_DIR = Path(__file__).resolve().parent.joinpath('data/bigger_data')

RASTER_GOOD_BLOCKS = BASE_DIR.joinpath('BWA_buildings.tif') #raster with square block sizes

OUT_RASTER = BASE_DIR.joinpath('dist-to.tif')
MASK_RASTER = BASE_DIR.joinpath('bwa_level0_100m_2000_2020.tif')

@pytest.fixture
def dist_good_without_mask_BIG():
	x = DistanceToEdge(RASTER_GOOD_BLOCKS, OUT_RASTER)
	yield x

def test_big_country(dist_good_without_mask_BIG):
	dist_good_without_mask_BIG._process_distance_to_edge()
	assert OUT_RASTER.exists()