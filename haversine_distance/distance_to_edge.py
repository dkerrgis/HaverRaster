# Author: David Kerr <d.kerr@soton.ac.uk>
# License: MIT

"""

Module with class to calculate the distance to edge (internal and external) from every source pixel to its closest destination polygon (closest point on polygon).

"""
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from rasterio.merge import merge
import rioxarray
from rioxarray.merge import merge_arrays, merge_datasets
import xarray


from haversine_distance.errors import BlockSizesError
from haversine_distance.utils import check_paths, check_tile_sizes, get_windows, get_points_from_pixels, get_buffer_gdf, clip_using_gdf, dist_to_edge, save_points_to_raster, make_dataset_from_points, get_rtrees, get_rtrees_from_geopackage, distance_to_polygon_edge, make_dataset_from_nodata_points, distance_to_nearest_neighbour, get_corner_buffers, get_corner_points, intersect_points_with_dataset, merge_corner_buffers_and_rebuffer

class DistanceToFeatureEdge:

    def __init__(self, in_raster, out_raster, buffer_method='corners', mask=None):
        """
        Class intantiation

        Parameters:
        ------------
        in_raster   :   Path/str
            Path to binary input raster (0 values -> External land pixels to caculate distances; 1 values -> Pixels defined as feature to calculate distances to; NA/NoData values -> 'sea' pixels that will not be calculated (if these disances are needed, then 'sea' pixels should be left as 0))
            Raster should be in EPSG:4326 projection

        out_raster  :    Path/str
            Path to output raster representing internal and external distances to the edge of the features defined in in_raster

        buffer_method  :    str ('corners'/'centroid')
            Method in which to find closest 0 and 1 pixels in global destination raster. 'corners' (Default) can take longer as 4 buffers are required for each tile. 'centroid' is faster but can result in edge effects due to 'closest' pixels being found as a false positive. See README. 

        mask    :   Path/str/None
            Path to mask raster if distances relative to global (in_raster) raster features are only desired within smaller extent. Raster should have 2 unique values only, one being nodata


        Returns:
        --------
        None
        """
        self.in_raster = check_paths(in_raster)
        if not check_tile_sizes(in_raster):
            raise BlockSizesError(f"Tile sizes should be square for this program. Please remake {self.in_raster.name} as a tiled raster with square blocks (i.e. >>gdal_translate -a_srs EPSG:4326 -co COMPRESS=LZW -co TILED=YES -co BLOCKXSIZE=512 BLOCKYSIZE=512 {str(self.in_raster)} <OUTPUT_NAME.tif> ")
        self.out_raster = check_paths(out_raster)
        self.buffer_method = buffer_method
        if not self.buffer_method in ['corners', 'centroid']:
            raise Exception("Buffer method shoud be 'corners' or 'centroid'.")
        self.dataset = rioxarray.open_rasterio(self.in_raster) # open global raster


    def points_generator(self):
        """
        Generator yields each block window of self.in_raster as points in geodataframe

        Parameters:
        -----------
        
        None

        Yields:
        --------
        
        gdf_pt  :   gpd.GeoDataFrame OR np.array
            GeoDataframe of point OR np.array of nodata values if no valid values in array


        """
        nodata = self.dataset._FillValue
        for data, window in get_windows(self.in_raster):
            if not np.all(data == nodata):
                gdf_pt = get_points_from_pixels(self.dataset, window)
            else:
                gdf_pt = data #This is not a geodataframe
            yield gdf_pt, window

    def get_rtrees_from_buffer(self, tile):
        """
        Iteratively buffers tile's bounds and clips raster dataset using buffer until both 0 AND 1 values are found in the clip. Polygons are then built from the 2 values and rtrees are built and returned

        Parameters:
        -----------
        tile    :   gpd.GeoDataFrame
            Point geodataframe representing raster tile


        Returns:
        ---------
        rtree_0     :   shapely.strtree.STRtree
            STR tree of geometries for polygons/features valued at 0

        rtree_1     :   shapely.strtree.STRtree
            STR tree of geometries for polygons/features valued at 1
        """
        pixels_0_and_1_present = False
        buffer_multiple = 2
        while not pixels_0_and_1_present:
            print(f'{buffer_multiple} buffer')
            try:
                buffer = get_buffer_gdf(tile, diagonal_multiples=buffer_multiple)
                clip = clip_using_gdf(buffer, self.dataset)
                if (1 in clip) and (0 in clip):                
                    pixels_0_and_1_present = True
                    rtree_0, rtree_1 = get_rtrees(clip)
                else:
                    buffer_multiple = buffer_multiple * 2
            except MemoryError as e:
                    raise e('Memory exceeded when trying to find closest feature to point')
        return rtree_0, rtree_1

    def get_destination_points_centroid_buffer(self, tile):
        """
        Returns geodataframe of points in buffer from tile. Will only return points that are valued 1 and 0. If not present, buffer will keep increasing

        Parameters:
        -----------
        tile    :   gpd.GeodataFrame
            Point geodataframe of raster tile (Source points)


        Returns:
        --------
        gdf_dst     :   gpd.GeoDataFrame
            Point geodataframe of destination points
        """
        pixels_0_and_1_present = False
        buffer_multiple = 2
        ############################
        while not pixels_0_and_1_present:
            try:
                buffer = get_buffer_gdf(tile, diagonal_multiples=buffer_multiple)
                clip = clip_using_gdf(buffer, self.dataset)
                if (1 in clip) and (0 in clip):                
                    pixels_0_and_1_present = True
                    gdf_dst = get_points_from_pixels(clip, window=None, remove_nodata_before_converting=True)
                else:
                    buffer_multiple = buffer_multiple * 5
            except MemoryError as e:
                    raise e('Memory exceeded when trying to find closest feature to point')
        return gdf_dst


    def get_destination_points_corner_buffer(self, tile):
        """
        Returns geodataframe of points in buffer from tile. Will only return points that are valued 1 and 0. Buffer is calculated be initially buffering from the 4 corners or the tile until 0/1 pixels are found. Corner buffers are then merged and a new buffer is made based on distance from tile centroid to corner buffers' bounding box corner

        Parameters:
        -----------
        tile    :   gpd.GeodataFrame
            Point geodataframe of raster tile (Source points)


        Returns:
        --------
        gdf_dst     :   gpd.GeoDataFrame
            Point geodataframe of destination points
        """
        try:
            gdf_corners = get_corner_buffers(tile, self.dataset)
            buffer = merge_corner_buffers_and_rebuffer(gdf_corners)
            dataset_clip = clip_using_gdf(buffer, self.dataset)
            gdf_dst = get_points_from_pixels(dataset_clip, window=None, remove_nodata_before_converting=True)
        except MemoryError as e:
            raise e('Memory exceeded when trying to find closest feature to point')
        return gdf_dst


    def calculate_distance(self, gdf_src, gdf_dst):
        """
        Returns gdf with 'dist_to' column appended with distance to closest feature

        Parameters:
        -----------

        self    :   Instantiated class

        gdf     :   gpd.GeoDataFrame
            Point geodataframe representing raster's pixels

        """
        gdf_src_nodata = gdf_src[gdf_src['data'] == self.dataset._FillValue]
        gdf_src_0 = gdf_src.loc[gdf_src['data'] == 0]
        gdf_src_1 = gdf_src.loc[gdf_src['data'] == 1]
        gdf_dst_0 = gdf_dst[gdf_dst['data'] == 0]
        gdf_dst_1 = gdf_dst[gdf_dst['data'] == 1]
        gdf_distance_0 = None 
        gdf_distance_1 = None
        if not gdf_src_0.empty:
            gdf_distance_0 = distance_to_nearest_neighbour(gdf_src_0, gdf_dst_1)
        if not gdf_src_1.empty:
            gdf_distance_1 = distance_to_nearest_neighbour(gdf_src_1, gdf_dst_0)
        if not gdf_distance_1 is None:
            if not gdf_distance_1.empty:
                gdf_distance_1.dist_to = gdf_distance_1.dist_to * -1
        if (gdf_distance_0 is not None) and (gdf_distance_1 is None):
            if not gdf_distance_0.empty:
                gdf_distance = gpd.GeoDataFrame(pd.concat([gdf_distance_0, gdf_src_nodata]))
        elif (gdf_distance_1 is not None) and (gdf_distance_0 is None):
            if not gdf_distance_1.empty:
                gpd.GeoDataFrame(pd.concat([gdf_distance_1, gdf_src_nodata]))
        else:
            gdf_distance = gpd.GeoDataFrame(pd.concat([gdf_distance_0, gdf_distance_1, gdf_src_nodata]))
        gdf_distance.loc[gdf_distance.data == 255, 'dist_to'] = -99999999
        return gdf_distance


    def calculate(self):
        """
        Wrapper function to calulate distance to edge and rasterise output

        Parameters:
        -----------

        self    :   DistanceToFeatureEdge
            Instantiated object

        Returns:
        ---------

        None
        """
        src = rasterio.open(self.in_raster)
        resolution = tuple(src.get_transform()[5:] + src.get_transform()[1:2])
        profile = src.profile.copy()
        original_nodata = profile['nodata']
        src.close()
        profile.update({
            "driver": "GTiff",
            "count": 1,
            "dtype": 'int32',
            "nodata": -99999999
            })
        index = 0
        with rasterio.open(self.out_raster, 'w', **profile) as dst:
            for tile, window in self.points_generator():
                if isinstance(tile, np.ndarray):
                    data = tile.astype(np.int32)
                    data[data == original_nodata] = dst.nodata
                    data = data[0]    
                else:
                    if self.buffer_method == 'centroid':
                        gdf_destination = self.get_destination_points_centroid_buffer(tile)
                    elif self.buffer_method == 'corners':
                        gdf_destination = self.get_destination_points_corner_buffer(tile)
                    gdf_dist = self.calculate_distance(tile, gdf_destination)
                    subset = self.dataset.rio.isel_window(window)                   
                    dataset_window = make_dataset_from_points(gdf_dist, resolution, subset)
                    data = dataset_window.dist_to.values
                    #tile.to_file(Path(__file__).resolve().parent.parent.joinpath(f'rubbish/tile_{index}.shp'))
                    #gdf_dist.to_file(Path(__file__).resolve().parent.parent.joinpath(f'rubbish/dist_{index}.shp'))
                dst.write(data, 1,  window=window)
                index += 1


    def calculate_dask(self, num_workers=4):
        """
        Process calculations and rasterisation using multiple processes using dask

        Parameters:
        -----------
        num_workers :   int
            Number of parallel processes to use

        Returns:
        ----------
        None

        """
        import dask
        from dask.distributed import Client, LocalCluster, as_completed
        from dask import delayed
        cluster = LocalCluster(n_workers=num_workers,
         threads_per_worker=1,
          processes=True,
          memory_limit="5GB")
        with Client(cluster) as client:

            src = rasterio.open(self.in_raster)
            resolution = tuple(src.get_transform()[5:] + src.get_transform()[1:2])
            profile = src.profile.copy()
            original_nodata = profile['nodata']
            src.close()
            profile.update({
                "driver": "GTiff",
                "count": 1,
                "dtype": 'int32',
                "nodata": -99999999
                })
            index = 0
            futures = []

            def process_(window, index):
                _tile = get_points_from_pixels(self.dataset, window)
                if self.buffer_method == 'centroid':
                    gdf_destination = self.get_destination_points_centroid_buffer(_tile)
                elif self.buffer_method == 'corners':
                    gdf_destination = self.get_destination_points_corner_buffer(_tile)
                #gdf_destination = self.get_destination_points(_tile)
                gdf_dist = self.calculate_distance(_tile, gdf_destination)
                subset = self.dataset.rio.isel_window(window)                   
                dataset_window = make_dataset_from_points(gdf_dist, resolution, subset)                    
                #dataset_window = make_dataset_from_points(gdf_dist, resolution)
                data = dataset_window.dist_to.values
                data_to_return = {'data': data, 'window': window, 'index': index}
                return data_to_return

            def process_nodata(data, window, index):
                data = data.astype(np.int32)
                data[data == original_nodata] = profile['nodata']
                data_to_return = {'data': data, 'window': window, 'index': index}
                return data_to_return

            with rasterio.open(self.out_raster, 'w', **profile) as dst, rasterio.open(self.in_raster) as src:
                for ij, window in src.block_windows():
                    data = src.read(1, window=window)
                    if np.all(data == original_nodata):
                        future = client.submit(process_nodata, data, window, index)
                        futures.append(future)
                    else:
                        future = client.submit(process_, window, index)
                        futures.append(future)
                    index += 1
                completed = as_completed(futures)
                for i in completed:
                    dst.write(i.result()['data'], 1,  window=i.result()['window'])
                    print(i.result()['index'])






