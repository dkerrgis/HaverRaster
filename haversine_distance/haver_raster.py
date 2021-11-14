# pylint: disable=line-too-long
"""

Module with class to calculate the distance to edge (internal and external) from every source pixel to its closest destination pixel.

"""

# Author: David Kerr <d.kerr@soton.ac.uk>
# License: MIT

from pathlib import Path

import geopandas as gpd
import numpy as np
import rasterio
from rasterio.merge import merge
import rioxarray
from rioxarray.merge import merge_arrays, merge_datasets


from haversine_distance.errors import BlockSizesError
from haversine_distance.utils import check_paths, check_tile_sizes, get_windows, get_points_from_pixels, get_buffer_gdf, clip_using_gdf, dist_to_edge, save_points_to_raster, make_dataset_from_points


class DistanceToEdge:
    """
    Class to calculate the distance to edge (internal and external) from every source pixel to its closest destination pixel.

    Input rasters (binary and mask) are:
    1. Binary raster that will be used to define source and destination pixels (will be inverted to define internal distance to edge) -- This can be a 'global' raster with a greater extent than that of the mask (2) to allow for accurate selection of source pixels' closest destination 'edge' pixels;
    2. Mask grid to define extent of calculations (i.e. 'global' raster could define region, 'mask' raster could define country);
    3. Mask and binary raster should be in geographic coordinate system (WGS84 / EPSG:4326)

    Ouput raster:
    1. Pixel values are number of meters to closest edge of pixels defined in binary input raster (negative internal values in pixels overlaying the pixels within those defined in the binary raster);
    2. Will be in geographic coordinate system (WGS84 / EPSG:4326)
    """
    def __init__(self, in_raster, out_raster, mask_raster=None):
        """
        Class intantiation

        Parameters:
        ------------
        in_raster   :   Path/str
            Path to binary input raster (0 values -> External land pixels to caculate distances; 1 values -> Pixels defined as feature to calculate distances to; NA/NoData values -> 'sea' pixels that will not be calculated (if these disances are needed, then 'sea' pixels should be left as 0))
            Raster should be in EPSG:4326 projection

        out_raster  :    Path/str
            Path to output raster representing internal and external distances to the edge of the features defined in in_raster

        mask_raster :   Path/str or None (default)
            Path to raster defining mask for processing extent and mask (only pixels in in_raster that overlay mask will have distances calculated in the out_raster. The remainder will have NoData value) - Setting mask is suggested to improve performance

        Returns:
        --------
        None
        """
        self.in_raster = check_paths(in_raster)
        if not check_tile_sizes(in_raster):
            raise BlockSizesError(f"Tile sizes should be square for this program. Please remake {self.in_raster.name} as a tiled raster with square blocks (i.e. >>gdal_translate -a_srs EPSG:4326 -co COMPRESS=LZW -co TILED=YES -co BLOCKXSIZE=512 BLOCKYSIZE=512 {str(self.in_raster)} <OUTPUT_NAME.tif> ")
        self.out_raster = check_paths(out_raster)
        if mask_raster is None:
            self.mask_raster = mask_raster
        else:
            self.mask_raster = check_paths(mask_raster)
            if not check_tile_sizes(mask_raster):
                raise BlockSizesError(f"Tile sizes should be square for this program. Please remake {self.mask_raster.name} as a tiled raster with square blocks (i.e. >>gdal_translate -a_srs EPSG:4326 -co COMPRESS=LZW -co TILED=YES -co BLOCKXSIZE=512 BLOCKYSIZE=512 {str(self.mask_raster)} <OUTPUT_NAME.tif> ")
        self.dataset = rioxarray.open_rasterio(self.in_raster)


    def points_generator(self):
        """
        Generator yields each block window of self.in_raster as points in geodataframe

        Parameters:
        -----------
        
        None

        Yields:
        --------
        
        gdf_pt  :   gpd.GeoDataFrame

        """
        nodata = self.dataset._FillValue
        for data, window in get_windows(self.in_raster):
            #if not data.all() == nodata:
            if not np.all(data == nodata):
                gdf_pt = get_points_from_pixels(self.dataset, window)
                yield gdf_pt, window

    def calc_distance(self, gdf_pt):
        """
        Iteratively increases buffer around gdf_pt until both 0 AND 1 values are present in buffer extracted from self.in_raster (i.e. If all pixes in buffer extract are nodata, buffer is increased until both values are found).

        Parameters:
        ------------

        gdf_pt  :   gpd.GeoDataFrame
            Point geodataframe representing 0 and 1 pixel values for tile in self.in_raster

        Returns:
        ---------

        gdf_dst_pt  :   gpd.GeoDataFrame
            Geodataframe of points in extract from self.in_raster where 0 and 1's are present.
        """
        buffer = 2 # start by buffering tile centroid by double distance to it's top right corner
        values_in_extract = False #start by assuming no valid (0 AND 1) values in extract
        while not values_in_extract:
            gdf_buffer = get_buffer_gdf(gdf_pt, buffer)
            dataset_clipped = clip_using_gdf(gdf_buffer, self.dataset)
            if (1 in dataset_clipped) and (0 in dataset_clipped):
                try:
                    gdf_destination = get_points_from_pixels(dataset_clipped)
                    gdf_dst_pt = dist_to_edge(gdf_pt, gdf_destination)
                    values_in_extract = True
                    return gdf_dst_pt
                except MemoryError as e:
                    raise e('Extract is too large for memory')
            else:
                buffer += 2

    def _process_distance_to_edge(self):
        """
        Function combines all functions to go through workflow and mosaic and save output

        Parameters:
        ------------
        None
        
        Returns:
        --------
        None
        """
        src = rasterio.open(self.in_raster)
        resolution = tuple(src.get_transform()[5:] + src.get_transform()[1:2])
        profile = src.profile.copy()
        src.close()
        profile.update({
            "driver": "GTiff",
            "count": 1,
            "dtype": 'int32',
            "nodata": -99999999
            })
        tiles_to_mosaic = []
        index = 0
        with rasterio.open(self.out_raster, 'w', **profile) as dst:
            for tile, window in self.points_generator():
                gdf_dist = self.calc_distance(tile)
                out_tile = self.out_raster.parent.joinpath(f'tmp_{self.out_raster.stem}/tmp_{index}.tif')
                if not out_tile.parent.exists():
                    out_tile.parent.mkdir(parents=True)
                save_points_to_raster(gdf_dist, out_tile, resolution)
                dataset_window = make_dataset_from_points(gdf_dist, resolution)
                data = dataset_window.dist_to
                dst.write(data, 1, window=window)
                index += 1
                print(index)

    def _process(self, tile, window):            
        nodata = self.dataset._FillValue
        #for data, window in get_windows(self.in_raster):
        data = 'x'
        if not data.all() == nodata:
            gdf_pt = get_points_from_pixels(self.dataset, window)
            yield gdf_pt, window

    def _process_distance_to_edge_dask(self, num_processes=4):        
        """
        Function combines all functions to go through workflow and mosaic and save output utilising dask multiprocessing to process each tile on a separate cpu

        Parameters:
        ------------
        num_processes : (int)
            Number of processes (Default = 4)
        
        Returns:
        --------
        None
        """        
        import dask
        from dask import delayed
        src = rasterio.open(self.in_raster)
        resolution = tuple(src.get_transform()[5:] + src.get_transform()[1:2])
        profile = src.profile.copy()
        src.close()
        profile.update({
            "driver": "GTiff",
            "count": 1,
            "dtype": 'int32',
            "nodata": -99999999
            })
        tiles_to_mosaic = []
        index = 0

        with rasterio. open(self.in_raster) as src, rasterio.open(self.out_raster, 'w', **profile) as dst:
            for tile, window in self.points_generator():
                gdf_dist = self.calc_distance(tile)
                out_tile = self.out_raster.parent.joinpath(f'tmp_{self.out_raster.stem}/tmp_{index}.tif')
                if not out_tile.parent.exists():
                    out_tile.parent.mkdir(parents=True)
                save_points_to_raster(gdf_dist, out_tile, resolution)
                dataset_window = make_dataset_from_points(gdf_dist, resolution)
                data = dataset_window.dist_to
                dst.write(data, 1, window=window)
                index += 1


