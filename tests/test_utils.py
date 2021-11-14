"""
Unit tests for helper functions in haversine_distance.utils
"""
from pathlib import Path

import geopandas as gpd
import numpy as np
import pytest
import rasterio
import rioxarray
import shapely 
from shapely.geometry import Point, Polygon
from shapely.strtree import STRtree
import xarray

from haversine_distance import check_paths, check_tile_sizes, get_in_out_profile, get_windows, get_points_from_pixels, haversine_distance, get_utm_zone, get_buffer_gdf, clip_using_gdf, dist_to_edge, save_points_to_raster, make_dataset_from_points, get_rtrees, get_rtrees_from_geopackage, distance_to_polygon_edge, get_corner_points, get_corner_buffers, merge_corner_buffers_and_rebuffer
from haversine_distance.errors import PathError

BASE_DIR = Path(__file__).resolve().parent.joinpath('data')

PATH_GOOD = Path(__file__).resolve()
STR_GOOD = str(PATH_GOOD)

RASTER_BAD_BLOCKS = BASE_DIR.joinpath('cls_130.tif') #raster with non-square block sizes
RASTER_GOOD_BLOCKS = BASE_DIR.joinpath('block.tif') #raster with square block sizes
POLYGONS_0 = BASE_DIR.joinpath('test_ABW_0.gpkg')
POLYGONS_1 = BASE_DIR.joinpath('test_ABW_1.gpkg')


###############  FIXTURES  ###############
#----------Workflow-------------------
@pytest.fixture
def pt_gdf():
    window_generator = get_windows(RASTER_GOOD_BLOCKS)
    data, window = next(window_generator)
    dataset = rioxarray.open_rasterio(RASTER_GOOD_BLOCKS)
    gdf = get_points_from_pixels(dataset, window)
    yield gdf

@pytest.fixture
def dst_pt(pt_gdf):
    gdf_buffer = get_buffer_gdf(pt_gdf, 2)
    dataset_global = rioxarray.open_rasterio(RASTER_GOOD_BLOCKS)
    dataset_clipped = clip_using_gdf(gdf_buffer, dataset_global)
    gdf_destination = get_points_from_pixels(dataset_clipped)
    gdf_distance = dist_to_edge(pt_gdf, gdf_destination)
    yield gdf_distance


###############  FIXTURES  ###############


###############  CHECKS  ###############
#-----------check_paths--------------------

def test_check_paths_returns_pathlib_obj():
    path = check_paths(PATH_GOOD)
    assert isinstance(path, Path)
    path_str = check_paths(STR_GOOD)
    assert isinstance(path, Path)

def test_wrong_path_type_raises_exception():
    wrong_path_type = 5
    with pytest.raises(PathError) as excinfo:
        path = check_paths(wrong_path_type)
    assert "Path should be pathlib.Path obj or str" in str(excinfo.value)	

#-----------check_paths--------------------

#----------check_tile_sizes--------------------
def test_non_square_tiles_returns_false_from_check_tile_sizes():
    square_blocks = check_tile_sizes(RASTER_BAD_BLOCKS)
    assert square_blocks == False

def test_square_tiles_returns_true_from_check_tile_sizes():
    square_blocks = check_tile_sizes(RASTER_GOOD_BLOCKS)
    assert square_blocks == True

#----------check_tile_sizes--------------------

#----------check_projection--------------------
def test_check_projection():
    pass
#----------check_projection--------------------

#----------check_resolution_matches_raster_and_mask--------------------
def test_check_resolution_matches():
    print('should resolution be the same???')
#----------check_resolution_matches_raster_and_mask--------------------

###############  CHECKS  ###############


###############  DISTANCE PROCESSING  ###############
def test_get_in_out_profile():
    in_profile, out_profile = get_in_out_profile(RASTER_GOOD_BLOCKS)
    for key, value in in_profile.items():
        if key == 'dtype':
            assert out_profile[key] == 'int32'
        elif key == 'nodata':
            assert out_profile[key] == -999999999
        else:
            assert value == out_profile[key]


def test_get_windows():
    window_generator = get_windows(RASTER_GOOD_BLOCKS)
    data, window = next(window_generator)
    assert isinstance(data, np.ndarray)
    assert window.col_off == 0 #First window in generator should be 0 offset
    assert window.row_off == 0 #First window in generator should be 0 offset


def test_get_points_from_pixels(pt_gdf):
    number_of_cells = 128 * 128 #block size in pixels
    assert len(pt_gdf) == number_of_cells

def test_haversine_distance():
    london = Point(-2.254337, 51.351073)
    seoul = Point(126.722237, 37.500354)
    expected_distance = 8959410
    distance = haversine_distance(london, seoul)
    assert abs(expected_distance - distance) < 100

def test_get_utm_zone():
    london = Point(-2.254337, 51.351073)
    seoul = Point(126.722237, 37.500354)
    epsg_london = 32630
    epsg_seoul = 32652
    assert get_utm_zone(london) == epsg_london
    assert get_utm_zone(seoul) == epsg_seoul

def test_get_buffer_gdf(pt_gdf):
    gdf = get_buffer_gdf(pt_gdf, 2)
    assert len(gdf) == 1
    
def test_clip_using_gdf(pt_gdf):
    gdf = get_buffer_gdf(pt_gdf, 2)
    dataset = rioxarray.open_rasterio(RASTER_GOOD_BLOCKS)
    dataset_clipped = clip_using_gdf(gdf, dataset)
    assert dataset_clipped.shape[1] < dataset.shape[1]

def test_get_rtrees(pt_gdf):
    gdf = get_buffer_gdf(pt_gdf, 2)
    dataset = rioxarray.open_rasterio(RASTER_GOOD_BLOCKS)
    dataset_clipped = clip_using_gdf(gdf, dataset)
    tree_0, tree_1 = get_rtrees(dataset_clipped)
    assert isinstance(tree_0, shapely.strtree.STRtree)
    assert isinstance(tree_1, shapely.strtree.STRtree)

def test_get_rtrees_from_geopackage():
    tree_0 = get_rtrees_from_geopackage(POLYGONS_0)
    assert isinstance(tree_0, shapely.strtree.STRtree)


def test_dist_to_edge(pt_gdf):
    gdf_buffer = get_buffer_gdf(pt_gdf, 2)
    dataset_global = rioxarray.open_rasterio(RASTER_GOOD_BLOCKS)
    dataset_clipped = clip_using_gdf(gdf_buffer, dataset_global)
    gdf_destination = get_points_from_pixels(dataset_clipped)
    gdf_distance = dist_to_edge(pt_gdf, gdf_destination)
    assert len(gdf_distance) == len(pt_gdf)
    assert list(gdf_distance.data.unique()) == [0, 1, 255]
    assert True

def test_distance_to_polygon_edge(pt_gdf):
    tree_0 = get_rtrees_from_geopackage(POLYGONS_0)
    tree_1 = get_rtrees_from_geopackage(POLYGONS_1)
    gdf_distance = distance_to_polygon_edge(pt_gdf, tree_0, tree_1)
    print('TRY SPEEDING UP BY PASSING BOTH 1/0 GDF TO GET CLOSEST POINT FUNCTION')
    assert 'dist_to' in gdf_distance.columns
    assert len(gdf_distance) == len(pt_gdf)
    assert list(gdf_distance.data.unique()) == [0, 1, 255]
    

def test_save_points_to_raster(dst_pt):
    src = rasterio.open(RASTER_GOOD_BLOCKS)
    profile = src.profile.copy()
    resolution = tuple(src.get_transform()[1:2] + src.get_transform()[5:])
    src.close()
    OUT_TILE = RASTER_GOOD_BLOCKS.parent.joinpath('test_dist_to_edge.tif')
    save_points_to_raster(dst_pt, OUT_TILE, resolution)
    src_out = rasterio.open(OUT_TILE)
    assert src_out.profile['nodata'] == -99999999
    assert src_out.profile['dtype'] == 'int32'
    assert abs(abs(src_out.get_transform()[1]) - abs(resolution[0])) < 0.000001

def test_make_dataset_from_points(dst_pt):
    src = rasterio.open(RASTER_GOOD_BLOCKS)
    profile = src.profile.copy()
    resolution = tuple(src.get_transform()[1:2] + src.get_transform()[5:])
    src.close()
    dataset = make_dataset_from_points(dst_pt, resolution)
    assert isinstance(dataset, xarray.Dataset)

def test_get_corner_points(pt_gdf):
    bounds = pt_gdf.total_bounds
    corners = get_corner_points(bounds)
    assert isinstance(corners, gpd.GeoDataFrame)
    assert len(corners) == 4


def test_get_corner_buffers(pt_gdf):
    dataset = rioxarray.open_rasterio(RASTER_GOOD_BLOCKS)
    gdf = get_corner_buffers(pt_gdf, dataset)
    assert isinstance(gdf, gpd.GeoDataFrame)
    assert len(gdf) == 4
    assert gdf.intersected.all() == True

def test_merge_corner_buffers_and_rebuffer(pt_gdf):
    dataset = rioxarray.open_rasterio(RASTER_GOOD_BLOCKS)
    gdf_corners = get_corner_buffers(pt_gdf, dataset)
    buffer = merge_corner_buffers_and_rebuffer(gdf_corners)
    dataset_clip = clip_using_gdf(buffer, dataset)
    assert 1 in dataset_clip
    assert 0 in dataset_clip
    assert len(buffer) == 1




###############  DISTANCE PROCESSING  ###############
