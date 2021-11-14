"""
Error class definitions
"""

# Author : David Kerr <d.kerr@soton.ac.uk>
# License : MIT

class PathError(Exception):
	pass 

class BlockSizesError(Exception):
	pass

class GeoPackageError(Exception):
	pass