#Import libraries
import pandas as pd
import geopandas as gpd
import pandas as pd
import os
import shapely
import numpy as np
#import psycopg2
import networkx as nx
import multiprocessing as mp
import pandana as pdna
import h5py
#import shapefile
import pyproj
import matplotlib.pyplot as plt
from pandana import Network
#from mpl_toolkits.basemap import Basemap
from shapely import ops
from shapely import wkt
from shapely.geometry import Polygon, LineString, Point, box
from pandana import Network
from geoalchemy2 import Geometry, WKTElement
from sqlalchemy import *
from shapely.geometry import *
from fiona.crs import from_epsg
from mpl_toolkits.axes_grid1 import make_axes_locatable

#%matplotlib inline

#pd.options.display.max_rows = 120

#pd.set_option('display.max_rows', 500)
#pd.set_option('display.max_columns', 500)
#pd.set_option('display.width', 1000)

# reading pednet file
pednet = gpd.read_file("zip://data/pednet.zip")
#pednet.head(2)
print(pednet.crs)
crs = {'init': 'epsg:4326'}
pednet = gpd.GeoDataFrame(pednet, crs=crs, geometry='geometry')
pednet = pednet.to_crs({'init': 'epsg:2019'})
pednet = pednet[['OBJECTID', 'road_type', 'sdwlk_code', 'sdwlk_desc', 'crosswalk', 'cwalk_type', 'px', 'px_type','geometry']]

ax = pednet.plot(figsize=(12, 12),color='blue', markersize =1)
ax.set(xlim=(311400, 311600), ylim=(4834000, 4834200))
plt.show()