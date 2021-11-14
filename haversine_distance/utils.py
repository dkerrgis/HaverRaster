"""
Helper functions to be imported by classes
"""

# Author : David Kerr <d.kerr@soton.ac.uk>
# License : MIT
import json
from math import radians, cos, sin, asin, sqrt
from pathlib import Path
import warnings

from geocube.api.core import make_geocube
import geopandas as gpd
import numpy as np
import pandas as pd
import pyproj
import rasterio
from rasterio.features import shapes
import rioxarray
from scipy.spatial import cKDTree
from sklearn.neighbors import BallTree
import shapely
from shapely.geometry import Point, box
from shapely.ops import nearest_points, transform
from shapely.strtree import STRtree
import utm


from pandas.core.common import SettingWithCopyError
from inspect import currentframe, getframeinfo

pd.options.mode.chained_assignment = 'raise'

from .errors import PathError, GeoPackageError


def check_paths(path):
    """
    Function checks whether given path is valid pathlib.Path. If not it will try to convert a str to a pathlib.Path and return this

    Parameters:
    -----------
    path    :   Path/str
        String or Path object of path to be checked

    Returns:
    --------
    path    :   Path
        Path checked and converted

    """
    if isinstance(path, str):
        try:
            path = Path(path).resolve()
            return path
        except Exception:
            raise PathError("There is a problem with in_raster path. Path should be pathlib.Path obj or str") from None
    elif isinstance(path, Path):
        return path
    else:
        raise PathError("There is a problem with in_raster path. Path should be pathlib.Path obj or str") from None

def check_tile_sizes(raster):
    """
    Returns True if raster tiled block sizes are square, else returns False. Block sizes need to be square to efficiently extract pixels from a buffer around the block. If narrow rectangles, there will be too much area extracted around the block

    Parameters:
    ------------

    raster  : Path/str
        Path to raster to check

    Returns:
    ---------

    boolean :   boolean
        True if xsize == ysize and tiled == True; else False
    """
    src = rasterio.open(raster)
    profile = src.profile
    if {'blockxsize', 'blockysize', 'tiled'} <= profile.keys():
        if profile['blockxsize'] == profile['blockysize'] and profile['tiled'] == True:
            return True
    else:
        return False

def get_in_out_profile(raster):
    """
    Returns input raster's profile as well as updated profile for output distance raster

    Parameters:
    -----------

    raster  :   Path/str
        Path to input raster

    Returns:
    --------

    in_profile  :   dict
        Input raster's profile

    out_profile     :   dict
        Output raster's profile updated to:
            1. dtype = Int32
            2. nodata = -999999999
    """
    src = rasterio.open(raster)
    profile_in = src.profile.copy()
    profile_out = src.profile.copy()
    src.close()
    profile_out.update(dtype='int32', nodata=-999999999)
    return profile_in, profile_out

def get_windows(raster):
    """Iterator to yield windows in input raster
    
    Parameters:
    -----------

    raster  :   str/Path
        Path to input 1/0/nodata raster to be used for calculate distances to

    Returns:
    --------

    data    :   np.array
        Array of values in raster within current block window
    window  :   Object
        Window object representing block. This will be used for getting the bounds and getting coordinates of pixels

    """
    with rasterio.open(raster) as src:
        for ji, window in src.block_windows():
            data = src.read(window=window)
            yield data, window

def get_points_from_pixels(dataset, window=None, remove_nodata_before_converting=False):
    """
    Returns GeoDataFrame of pixels in raster window as points

    Parameters:
    ------------

    dataset  :   rioxarray dataset
        Rioxarray reference to input raster (rioxarray.open_rasterio(<raster>))

    window  :   rasterio.windows.Window object (Default=None)
        Window to be read and converted to points. If None, gdf will be processed on whole dataset

    Returns:
    --------

    gdf     :   gpd.GeoDataFrame
        Geopandas geodataframe of pixels as points


    """
    if window: #Raster dataset sent in blocks
        subset = dataset.rio.isel_window(window) #get pixels from raster
        subset.name = 'data'
        if remove_nodata_before_converting:
            subset = subset.where(subset != subset._FillValue).stack(z=['band', 'x', 'y'])
            subset = subset[subset.notnull()] #Remove nodata before converting to points
        df = subset.squeeze().to_dataframe().reset_index()
        geometry = gpd.points_from_xy(df.x, df.y)
        gdf = gpd.GeoDataFrame(df, crs=dataset.rio.crs, geometry=geometry)
    else:
        dataset.name = 'data'
        if remove_nodata_before_converting:
            dataset = dataset.where(dataset != dataset._FillValue).stack(z=['band', 'x', 'y'])
            dataset = dataset[dataset.notnull()] #Remove nodata before converting to points
        df = dataset.squeeze().to_dataframe().reset_index()
        geometry = gpd.points_from_xy(df.x, df.y)
        gdf = gpd.GeoDataFrame(df, crs=dataset.rio.crs, geometry=geometry)
    return gdf

def haversine_distance(src_pt, dst_pt):
    """
    Returns great circle distance in meters between 2 points

    Parameters:
    -----------
    src_pt  :   shapely.geometry.Point
        Point object of source point
    
    dst_pt  :   shapely.geometry.Point
        Pont object of destination Point

    Returns:
    --------
    distance    :    int
        Distance between src_pt and dst_pt in meters

    """
    # convert decimal degrees to radians 
    lon1, lat1, lon2, lat2 = map(radians, [src_pt.x, src_pt.y, dst_pt.x, dst_pt.y]) 
    dlon = lon2 - lon1 
    dlat = lat2 - lat1 
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a)) 
    r = 6371000 # Radius of earth in meters.
    return c * r

def get_utm_zone(point):
    """
    Returns EPSG code for UTM zone in which input point is located

    Parameters:
    ------------

    point   :   Shapely.geometry.Point
        Point object for point

    Returns:
    --------

    epsg   :    int
        EPSG code for UTM zone in which input point is located  

    """
    utm_x, utm_y, band, zone = utm.from_latlon(point.y, point.x)
    if point.y > 0:
        epsg = 32600 + band
    else:
        epsg = 32700 + band
    return epsg


def get_buffer_gdf(gdf, diagonal_multiples=2):
    """
    Returns circular-like buffer of centroid of gdf.bounds. Distance is multiples of distances between centroid and top right corner.

    **Note - some of the commands called in this function will raise a numpy exception (see https://github.com/Toblerity/Shapely/pull/1174). These have been silenced for now

    Parameters:
    -----------

    gdf     :   gpd.GeoDataFrame
        GeoDataFrame of points that should have a square shape for the purposes of this package

    diagonal_multiples  :   int (Default=2)
        Multiples of distance between gdf.bounds centroid and top right corner of gdf.bounds 

    Returns:
    --------

    gdf_buffer  :   gpd.GeoDataFrame
        GeoDataFrame of buffer from gdf.bounds centroid
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore") # Below commands raises numpy / shapely warning (see https://github.com/Toblerity/Shapely/pull/1174 for details)
        bounds = gdf.geometry.total_bounds
        bbox = box(bounds[0], bounds[1], bounds[2], bounds[3])
        geo = gpd.GeoDataFrame({'x': 0, 'geometry': bbox}, index=[0], crs="EPSG:4326") #Dataframe of bounding box of tile
        utm_epsg = get_utm_zone(geo.geometry.centroid[0])
        geo = geo.to_crs(f'EPSG:{utm_epsg}') #Needs to be in meters to calculate buffer
        corner = Point(geo.geometry.total_bounds[0], geo.geometry.total_bounds[1])
        centre = geo.centroid #Centre of the bounding box
        distance = centre.distance(corner) * diagonal_multiples #buffer distance from centroid to corner
        geo_buffer = gpd.GeoDataFrame({'x': 0, 'geometry': centre}).buffer(distance)
        geo_buffer = geo_buffer.to_crs('EPSG:4326') #Return back to WGS84
        return geo_buffer

def get_corner_buffers(gdf, dataset):
    """
    Buffers 4 corners of gdf's bounds, for each one extracts pixels from the global raster and if 1 and 0 in ALL buffers, they are dissolved together, and a buffer is made from the gdf's centroid to the corner of the dissolved buffers' bounding box.

    Parameters:
    ------------
    gdf :   gpd.GeoDataFrame
        Geodataframe of points of tile being processed

    dataset :   rioxarray.Dataset
        Global feature destination raster

    Returns:
    ---------
    gdf_buffer  :   gpd.GeoDataFrame
        Geodataframe of buffer that intersect closest pixels (This will NOT contain and pixels of the destination raster at this point)
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore") #Warning created on centroid calculation in wgs84. This doesn't have to be too accurate
        bounds = gdf.total_bounds
        corner_points = get_corner_points(bounds)
        bbox = box(bounds[0], bounds[1], bounds[2], bounds[3])
        geo = gpd.GeoDataFrame({'x': 0, 'geometry': bbox}, index=[0], crs="EPSG:4326") #Dataframe of bounding box of tile
        utm_epsg = get_utm_zone(geo.geometry.centroid[0])
        geo = geo.to_crs(f'EPSG:{utm_epsg}') #Needs to be in meters to calculate buffer
        corner = Point(geo.geometry.total_bounds[0], geo.geometry.total_bounds[1])
        centre = geo.centroid
        distance = centre.distance(corner)
        gdf_buffers = corner_points.apply(lambda row: intersect_points_with_dataset(row, utm_epsg, distance, dataset), axis=1)
        return gdf_buffers

def get_corner_points(bounds):
    """
    Returns list of 4 corners of geodataframe total points. Each element in list is a shapely.geometry Point

    Paramenters:
    -------------
    total_bounds    :   np.ndarray
        Array of 4 coordinates of gdf.total_bounds

    Returns:
    ---------
    gdf_corners   :   gpd.GeoDataFrame
        Geodataframe of shapely points representing corner points of geodataframe
    """
    uleft = Point([bounds[0], bounds[3]]) #Corner points
    uright = Point([bounds[2], bounds[3]]) #Corner points
    lleft = Point([bounds[0], bounds[1]]) #Corner points
    lright = Point(bounds[2], bounds[1]) #Corner points
    corner_points = [uleft, uright, lleft, lright]
    labels = points = ['Upper left', 'Upper right', 'Lower left', 'Lower right']
    gdf_corners = gpd.GeoDataFrame({'points': labels, 'geometry': corner_points}, crs="EPSG:4326")
    gdf_corners['intersected'] = False #Column will become true once corner buffers intersect pixels in global raster
    return gdf_corners

def intersect_points_with_dataset(row, utm_epsg, distance, dataset):
    """Iteratively buffers row's geometry until buffer intersects both 0 and 1 features in dataset
    
    Parameters:
    -----------
    row     :   gpd.GeoSeries
        Row from geodataframe

    utm_epsg  :     int
        UTM zone of row's point for use in projection

    distance    :   int
        seed distance in meters to buffer point. This will iteratively be increased if no intersectioun made with dataset's pixels

    dataset :   rioxarray.DataArray
        Global raster from which features will be intersected
    """
    wgs84 = pyproj.CRS('EPSG:4326')
    utm = pyproj.CRS(f'EPSG:{utm_epsg}')
    project = pyproj.Transformer.from_crs(wgs84, utm, always_xy=True).transform
    utm_point = transform(project, row.geometry)
    pixels_0_and_1_present = False
    buffer_multiple = 2
    while not pixels_0_and_1_present:
        try:
            utm_buff = utm_point.buffer(distance)
            project = pyproj.Transformer.from_crs(utm, wgs84, always_xy=True).transform
            wgs_buff = transform(project, utm_buff)
            clip = dataset.rio.clip([wgs_buff], 'EPSG:4326', from_disk=True)
            if (1 in clip) and (0 in clip):
                pixels_0_and_1_present = True
            else:
                distance = distance * buffer_multiple
        except MemoryError as e:
            raise e('Memory exceeded when trying to find closest feature to point')
    row.geometry = wgs_buff
    row.intersected = True
    return row


def merge_corner_buffers_and_rebuffer(gdf_corners):
    """
    Merge/dissolves buffers created in corners of tiles, then rebuffers from tile's centroid to the distance of the dissolved buffers' bounding box corner

    Parameters:
    ------------

    gdf_corners :   gpd.GeoDataFrame
        Geodataframe of tile's corners' buffereed to intersect closest feature pixels. 

    Returns:
    --------
    
    extraction_buffer   :   gpd.GeoDataFrame
        Geodataframe of buffer from tile's centroid to corner of dissolved buffers bounding box.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore") #Warning created on centroid calculation in wgs84. This doesn't have to be too accurate
        gdf_corners['dissolve'] = 1
        gdf_merge = gdf_corners.dissolve(by='dissolve')
        bounds = gdf_merge.total_bounds
        bbox = box(bounds[0], bounds[1], bounds[2], bounds[3])
        geo = gpd.GeoDataFrame({'x': 0, 'geometry': bbox}, index=[0], crs="EPSG:4326") #Dataframe of bounding box of tile
        utm_epsg = get_utm_zone(geo.geometry.centroid[0])
        geo = geo.to_crs(f'EPSG:{utm_epsg}') #Needs to be in meters to calculate buffer
        corner = Point(geo.geometry.total_bounds[0], geo.geometry.total_bounds[1])
        centre = geo.centroid #Centre of the bounding box
        distance = centre.distance(corner)
        geo_buffer = gpd.GeoDataFrame({'x': 0, 'geometry': centre}).buffer(distance)
        extraction_buffer = geo_buffer.to_crs('EPSG:4326')
        return extraction_buffer




def clip_using_gdf(gdf, dataset):
    """
    Returns data from raster dataset that intersects gdf.geometry

    Parameters:
    -----------

    gdf     :   gpd.GeoDataFrame
        Single polygon geodataframe to use to clip dataset raster

    dataset :   rioxarray.DataArray
        Global dataset of read by rioxarray for data to be clipped 


    Returns:
    --------

    dataset_clip    :   rioxarray.DataArray
        Dataset of input dataset clipped by input gdf
    """
    dataset_clip = dataset.rio.clip(gdf.geometry, gdf.crs, from_disk=True)
    return dataset_clip


def get_rtrees(raster):
    """
    Returns the Sort-Tile-Recursive (STR) tree of the polygons created from vectorising each value (0 and 1) in the input raster. An STRtree is returned for each value as a list.

    Parameters:
    -----------
    raster  :   rioxarray.DataArray
        Dataset of feature raster tile

    Returns:
    --------
    tree_0  :   shapely.strtree.STRtree
        Sort-Tile-Recursive (STR) tree of polygons created for 0-values in input raster

    tree_1  :   shapely.strtree.STRtree
        Sort-Tile-Recursive (STR) tree of polygons created for 1-values in input raster
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore") # Below commands raises numpy / shapely warning (see https://github.com/Toblerity/Shapely/pull/1174 for details)
        mask = None
        nodata = raster.rio.nodata
        image = raster.data
        results = (
            {'properties': {'class': v}, 'geometry': s}
        for i, (s, v) 
        in enumerate(
            shapes(image, mask=mask, transform=raster.rio.transform())))
        geoms = list(results)
        gdf = gpd.GeoDataFrame.from_features(geoms, crs='EPSG:4326')   
        tree_0 = STRtree(gdf.geometry.loc[gdf['class'] == 0].to_list())
        tree_1 = STRtree(gdf.geometry.loc[gdf['class'] == 1].to_list())
        return tree_0, tree_1


def get_rtrees_from_geopackage(geopackage, layer=None):
    """
    Returns rtree created from polygons in geopackage (MUST BE ONLY 1 OR 0 VALUES, NOT BOTH)

    Parameters:
    -----------
    geopackage  :   Path/str
        Path to input geopackage

    layer   : None/str
        Layer within geopackage from which rtrees should be created

    Returns:
    --------
    tree    :   shapely.strtree.STRtree
        Sort-Tile-Recursive (STR) tree of polygons in geopackage

    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        gdf = gpd.read_file(geopackage, layer=None)
        if not 'DN' in gdf.columns:
            raise GeoPackageError("Please use a geopackage with a field name 'DN' denoting the polygon values")
        if not len(gdf.DN.unique()) == 1:
            raise GeoPackageError("Please use a geopackage with only 1 polygon value (0 OR 1)")
        tree = STRtree(gdf.geometry.to_list())
    return tree



def dist_to_edge(gdf_tile, gdf_global):
    """
    Joins geometries of both dataframes based on value 1/0 in gdf_tile points closest to 0/1 in gdf_global. Then calculates the distance in meters (int) between the point and it's nearest opposite (valued) neighbour.

    Parameters:
    ------------

    gdf_tile    :   gpd.GeoDataFrame
        Point geodataframe of source pixels in raster (both 0 - external and 1 internal source pixels)

    gdf_global  :   gpd.GeoDataFrame
        Point geodataframe of destination pixels in raster (both 0 - external and 1 internal source pixels)

    Returns:
    ---------

    gdf_distance_to_edge    :   gpd.GeoDataFrame
        Point geodataframe of source points with distances to nearest opposite (value) neighbour - Internal distances are converted to negative (0 --> 1 dist and 1 --> 0 * -1)

    """
    gdf_tile_0 = gdf_tile.loc[gdf_tile['data'] == 0] #Remove nodata points
    gdf_tile_1 = gdf_tile.loc[gdf_tile.loc['data'] == 1] #Remove nodata points
    gdf_tile_ND = gdf_tile.loc[gdf_tile.data == 255, :].copy() #Keep nodata to join back after calculation
    gdf_tile_ND.loc[:, 'dist_to'] = -99999999
    cols = ['data', 'dist_to', 'geometry']
    gdf_tile_ND = gdf_tile_ND[cols]
    gdf_global_0 = gdf_global.loc[gdf_global['data'] == 0] #Remove nodata points
    gdf_global_1 = gdf_global.loc[gdf_global['data'] == 1] #Remove nodata points
    gdf_tile_0 = ckdnearest(gdf_tile_0, gdf_global_1)
    gdf_tile_1 = ckdnearest(gdf_tile_1, gdf_global_0)
    if gdf_tile_0.empty:
        gdf = gdf_tile_1.copy()
    elif gdf_tile_1.empty:
        gdf = gdf_tile_0.copy()
    else:
        gdf = gpd.GeoDataFrame(pd.concat([gdf_tile_0, gdf_tile_1]))
    gdf['dist_to'] = gdf.apply(lambda row: haversine_distance(row.geometry, row.geometry_dst), axis=1)
    gdf = gdf[cols].copy()
    gdf.loc[gdf.data == 1, 'dist_to'] = gdf.loc[gdf.data == 1, 'dist_to'].apply(lambda x : x * -1)
    gdf = gpd.GeoDataFrame(pd.concat([gdf, gdf_tile_ND]))
    return gdf

def _get_closest_point(geom, tree):
    """
    Returns the closest point between the input geom and the closest polygon in the input rtree

    Parameters:
    -----------
    geom    :   shapely.geometry.Point
        Point representing raster pixels

    tree    :   shapely.strtree.STRtree
        Sort-Tile-Recursive (STR) tree of polygons

    Returns:
    ---------
    geom    :   shapely.geometry.Point
        Point on closest polygon closest to input geom
    """
    nearest_poly = tree.nearest(geom)
    p1, p2 = nearest_points(nearest_poly, geom)
    return p1 

def _get_distance_to_closest_point(source, destination, k_neighbours=1):
    """
    Find closest point to all source points from destiantion points. Returns point and distance

    This function is adapted from get_nearest() at https://automating-gis-processes.github.io/site/notebooks/L3/nearest-neighbor-faster.html

    Parameters:
    -----------
    source  :   np.array
        Source points converted to radians

    destination :   np.array
        Destination points converted to radians

    Returns:
    --------
    closest_pt  :   np.array
        Closest point from destination to respective source points as radians

    distance    :   np.array
        Distance to closest point for each point in source array
    """
    ball_tree = BallTree(destination, leaf_size=15, metric='haversine')
    distances, indices = ball_tree.query(source, k=k_neighbours)
    distances = distances.transpose()
    indices = indices.transpose()
    closest_distance = distances[0]
    return closest_distance

def distance_to_nearest_neighbour(src_gdf, dst_gdf):
    """
    Calculates distance of points in src_gdf to nearest neighbour of opposite value in dst_gdf. Geodataframe with distances appended are returned

    Parameters:
    ------------
    src_gdf :   gpd.GeoDataFrame
        Point geodataframe to which distances to nearest neighbour should be calculated

    dst_gdf :   gpd.GeoDataFrame
        Point geodataframe of desitantion points to be used to find nearest neighours

    return_dst :    boolean
        Append (or not) distance to geodataframe
    """
    try:
        right = dst_gdf.copy().reset_index(drop=True)
        src_radians = np.array(src_gdf['geometry'].apply(lambda geom: (geom.y * np.pi / 180, geom.x * np.pi / 180)).to_list())
        dst_radians = np.array(right['geometry'].apply(lambda geom: (geom.y * np.pi / 180, geom.x * np.pi / 180)).to_list())

        distance = _get_distance_to_closest_point(src_radians, dst_radians, k_neighbours=1)
        gdf_dst = src_gdf.reset_index(drop=True)
        earth_radius = 6371000  # meters
        gdf_dst['dist_to'] = distance * earth_radius
    except Exception as e:
        print(e)
        gdf_dst = None
    return gdf_dst



def distance_to_polygon_edge(gdf_tile, tree_0, tree_1):
    """
    Calculates distance from points in gdf_tile to distance to closest polygon in both trees. Points valued 0 will be measured to polygons valued 1 and vice versa

    Parameters:
    -----------
    gdf_tile    :   gpd.GeoDataFrame
        Points of raster pixels in geodataframe

    tree_0  :   shapely.strtree.STRtree
        Sort-Tile-Recursive (STR) tree of polygons
    """
    gdf_tile_0 = gdf_tile.loc[gdf_tile['data'] == 0].copy() #Remove nodata points
    gdf_tile_1 = gdf_tile.loc[gdf_tile['data'] == 1].copy() #Remove nodata points
    gdf_tile_ND = gdf_tile.loc[gdf_tile.data == 255, :].copy() #Keep nodata to join back after calculation
    gdf_tile_ND.loc[:, 'dist_to'] = -99999999
    cols = ['data', 'dist_to', 'geometry']
    gdf_tile_ND = gdf_tile_ND[cols]
    gdf_tile_0['geometry_dst'] = gdf_tile_0.apply(lambda row: _get_closest_point(row.geometry, tree_1), axis=1)
    gdf_tile_1['geometry_dst'] = gdf_tile_1.apply(lambda row: _get_closest_point(row.geometry, tree_0), axis=1)
    if gdf_tile_0.empty:
        gdf = gdf_tile_1.copy()
    elif gdf_tile_1.empty:
        gdf = gdf_tile_0.copy()
    else:
        gdf = gpd.GeoDataFrame(pd.concat([gdf_tile_0, gdf_tile_1]))
    gdf.loc[:, 'dist_to'] = gdf.apply(lambda row: haversine_distance(row.geometry, row.geometry_dst), axis=1)
    gdf = gdf[cols]
    gdf.loc[gdf.data == 1, 'dist_to'] = gdf.loc[gdf.data == 1, 'dist_to'].apply(lambda x : x * -1)
    if not gdf_tile_ND.empty:
        gdf = gpd.GeoDataFrame(pd.concat([gdf, gdf_tile_ND]))
    return gdf


def ckdnearest(gdf_tile, gdf_global):
    """Returns nearest neighbours between 1 gdf and another
    Function derived from posting @ https://gis.stackexchange.com/questions/222315/geopandas-find-nearest-point-in-other-dataframe with thanks to user JHuw
    
    Parameters:
    -----------

    gdf_tile    :   gpd.GeoDataFrame
        Point geodataframe of source points (limited to single value - 0 OR 1)

    gdf_global  :   gpd.GeoDataFrame
        Point geodataframe of destination points (limited to single 'data' value opposite to that of gdf_tile (0 OR 1))

    Returns:
    --------

    gdf     :   gpd.GeoDataFrame
        gdf_tile points with additional column of closest point from gdf_global

    """
    gdf_global = gdf_global.rename(columns={'geometry': 'geometry_dst', 'data': 'data_dst'})
    try:
        nA = np.array(list(gdf_tile.geometry.apply(lambda x: (x.x, x.y))))
        nB = np.array(list(gdf_global.geometry_dst.apply(lambda x: (x.x, x.y))))
        btree = cKDTree(nB)
        dist, idx = btree.query(nA, k=1)
        gdB_nearest = gdf_global.iloc[idx].reset_index(drop=True)
        gdf = pd.concat(
            [
                gdf_tile.reset_index(drop=True),
                gdB_nearest,
                pd.Series(dist, name='dist')
            ], 
            axis=1)
    except ValueError:
        gdf = gpd.GeoDataFrame({'y': [], 'x': [], 'band': [], 'spatial_ref': [], 'data': [], 'geometry': [], 'data_dst': [], 'geometry_dst': [], 'dist': []})
    return gdf

def save_points_to_raster(gdf, outname, resolution, nodata=-99999999):
    """
    Saves input GeoDataFrame points to raster as outname


    Parameters:
    ------------

    gdf     :   gpd.GeoDataFrame
        Geodataframe with points geometry and corresponding 'dist_to' column

    outname :   str/Path
        Path to output raster

    resolution   :  tuple
        2 Tuple of resolution (y should be negative, i.e. (0.83, -0.83))

    nodata  :   int
        NoData value in output


    Returns:
    ---------
    None

    """

    x, y = resolution
    grid = make_geocube(
        vector_data=gdf,
        measurements=['dist_to'],
        resolution=(x, y),
        fill=nodata
    )
    grid['dist_to'] = grid.dist_to.astype(np.int32)
    grid.dist_to.rio.to_raster(outname)


def make_dataset_from_points(gdf, resolution, subset, nodata=-99999999):
    """
    Returns input points geodataframe as rioxarray raster dataset

    Parameters:
    -----------

    gdf     :   gpd.GeoDataFrame
        Geodataframe with points geometry and corresponding 'dist_to' column

    resolution   :  tuple
        2 Tuple of resolution (y should be negative, i.e. (0.83, -0.83))

    subset  :   rioxarray.DataArray
        Data array of window to be processed to be used as rasterise extent

    nodata  :   int
        NoData value in output


    Returns:
    ---------
    dataset     :   rioxarray.Dataset
        Rasterised points gdf

    """    
    x, y = resolution
    grid = make_geocube(
        vector_data=gdf,
        measurements=['dist_to'],
        #resolution=(x, y),
        like=subset,
        fill=np.nan
    ).fillna(nodata)
    grid['dist_to'] = grid.dist_to.astype(np.int32)
    return grid


def make_dataset_from_nodata_points(gdf, resolution, nodata=-99999999):
    """
    Returns input points geodataframe as rioxarray raster dataset

    Parameters:
    -----------

    gdf     :   gpd.GeoDataFrame
        Geodataframe with points geometry and corresponding 'dist_to' column

    resolution   :  tuple
        2 Tuple of resolution (y should be negative, i.e. (0.83, -0.83))

    nodata  :   int
        NoData value in output


    Returns:
    ---------
    dataset     :   rioxarray.Dataset
        Rasterised points gdf

    """    
    x, y = resolution
    grid = make_geocube(
        vector_data=gdf,
        measurements=['data'],
        resolution=(x, y),
        fill=np.nan
    ).fillna(nodata)
    grid['data'] = grid.data.astype(np.int32)
    return grid


