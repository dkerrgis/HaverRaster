"""
Unit tests for helper functions in haversine_distance.haver_raster
"""

from pathlib import Path

import geopandas as gpd
import pytest
import rasterio


from haversine_distance import DistanceToEdge
from haversine_distance.errors import PathError, BlockSizesError

BASE_DIR = Path(__file__).resolve().parent.joinpath('data')

RASTER_BAD_BLOCKS = BASE_DIR.joinpath('cls_130.tif') #raster with non-square block sizes
RASTER_GOOD_BLOCKS = BASE_DIR.joinpath('block.tif') #raster with square block sizes

OUT_RASTER = BASE_DIR.joinpath('dist-to.tif')
MASK_RASTER = BASE_DIR.joinpath('ABW_L0_mastergrid.tif')


@pytest.fixture
def dist_good_with_mask():
	x = DistanceToEdge(RASTER_GOOD_BLOCKS, OUT_RASTER, MASK_RASTER)
	yield x

@pytest.fixture
def dist_good_without_mask():
	x = DistanceToEdge(RASTER_GOOD_BLOCKS, OUT_RASTER)
	yield x

@pytest.fixture
def dist_bad_tiles():
	x = DistanceToEdge(RASTER_BAD_BLOCKS, OUT_RASTER)
	yield x


def test_object_creation_with_mask(dist_good_with_mask):
	assert isinstance(dist_good_with_mask, DistanceToEdge)

def test_object_creation_with_no_mask(dist_good_without_mask):
	assert isinstance(dist_good_without_mask, DistanceToEdge)

def test_exception_raised_for_non_square_tiles():
	with pytest.raises(BlockSizesError) as excinfo:
		x = DistanceToEdge(RASTER_BAD_BLOCKS, OUT_RASTER)
	assert "gdal_translate" in str(excinfo.value)

def test_points_generator(dist_good_without_mask):
	gen = dist_good_without_mask.points_generator()
	gdf, window = next(gen)
	assert isinstance(gdf, gpd.geodataframe.GeoDataFrame)
	assert isinstance(window, rasterio.windows.Window)


def test_calc_distance(dist_good_without_mask):
	gen = dist_good_without_mask.points_generator()
	gdf, window = next(gen)
	gdf_dst = dist_good_without_mask.calc_distance(gdf)
	assert isinstance(gdf_dst, gpd.geodataframe.GeoDataFrame)
	assert 'dist_to' in gdf_dst.columns

def test_process_distance_to_edge(dist_good_without_mask):
	dist_good_without_mask._process_distance_to_edge()
	assert dist_good_without_mask.out_raster.exists()


