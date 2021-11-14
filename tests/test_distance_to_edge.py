"""
Unit tests for helper functions in haversine_distance.haver_raster
"""

from pathlib import Path

import geopandas as gpd
import pytest
import rasterio
import rioxarray
from shapely.strtree import STRtree
import xarray

from haversine_distance import DistanceToFeatureEdge
from haversine_distance.errors import PathError, BlockSizesError

BASE_DIR = Path(__file__).resolve().parent.joinpath('data')

RASTER_GOOD_BLOCKS = BASE_DIR.joinpath('block.tif') #raster with square block sizes
#GEOPACKAGE_0 = BASE_DIR.joinpath('test_ABW_0.gpkg')
#GEOPACKAGE_1 = BASE_DIR.joinpath('test_ABW_1.gpkg')

OUT_RASTER = BASE_DIR.joinpath('dist-to-nearest-point.tif')
MASK_RASTER = BASE_DIR.joinpath('ABW_L0_mastergrid.tif')


@pytest.fixture
def dist():
	x = DistanceToFeatureEdge(RASTER_GOOD_BLOCKS, OUT_RASTER)
	yield x 

def test_class_instantiaiton(dist):
	assert isinstance(dist, DistanceToFeatureEdge)


def test_calculate(dist):
	dist.calculate()
	assert OUT_RASTER.exists()
	profile_expected = rasterio.open(RASTER_GOOD_BLOCKS).profile
	profile_got = rasterio.open(OUT_RASTER).profile
	assert profile_expected['height'] == profile_got['height']
	assert profile_expected['width'] == profile_got['width']
	assert profile_got['dtype'] == 'int32'

def test_calculate_dask(dist):
	dist.calculate_dask()
