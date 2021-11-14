from pathlib import Path

import pytest

from haversine_distance import DistanceToEdge

BASE_DIR = Path(__file__).resolve().parent.joinpath('tests/data')
GLOBAL_BINARY_RASTER = BASE_DIR.joinpath('block.tif')
OUT_RASTER = BASE_DIR.joinpath('example_dst.tif')



def main():
	dist = DistanceToEdge()


if __name__ == "__main__":
	main()