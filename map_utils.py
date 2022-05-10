import json
import math
import pandas as pd
import geopandas as gpd
import pandas as pd
import osmnx as ox
import shapely
import numpy as np
from shapely.geometry import Polygon, LineString, Point, box
from sqlalchemy import *
from shapely.geometry import *
import os


def get_nias(file_path="data/neighbourhood-improvement-areas-wgs84/NEIGHBOURHOOD_IMPROVEMENT_AREA_WGS84.shp"):
    nia = gpd.read_file(file_path)
    nia.columns = map(str.lower, nia.columns)
    nia = nia[['area_id', 'area_s_cd', 'area_name', 'geometry']]
    print("nias file crs is: " + str(nia.crs))
    crs = {'init': 'epsg:4326'}
    nia = gpd.GeoDataFrame(nia, crs=crs, geometry='geometry')
    #nia = nia.to_crs({'init': 'epsg:2019'})
    return nia


def get_CTs_boundary(CTs, use, file_path="data/lct_000b16a_e/lct_000b16a_e.shp"):
    boundary = gpd.read_file(file_path)
    print("CT file crs is: "+ str(boundary.crs))
    crs = {'init': 'epsg:3347'}
    boundary = gpd.GeoDataFrame(boundary, crs=crs, geometry='geometry')
    boundary = boundary[boundary["CMANAME"] == "Toronto"]
    if use=="ox":
        boundary = boundary.to_crs({'init': 'epsg:4326'})
    elif use=="pednet":
        boundary = boundary.to_crs({'init': 'epsg:2019'})
    else:
        print("NEED TO SPECIFY CPRS !!!")
        return

    exclude = ['0806.00', '0400.22', '0803.04', '0803.03', '0801.02', '0412.02', '0576.40', '0531.02', '0528.41',
               '0528.35', '0510.00', '0530.02', '0531.02', '0525.02', '0527.04', '0500.02']
    if CTs is not None:
        boundary = boundary[boundary["CTNAME"].isin(CTs)]
    boundary = boundary[~boundary["CTNAME"].isin(exclude)]

    return boundary

def query_ox(polygons,tags):
    # might want to sava this to disk
    frames=[]
    for polygon in polygons:
        frames.append(ox.geometries.geometries_from_polygon(polygon, tags))  #['unique_id', 'osmid', 'element_type', 'building', 'geometry']
    result = pd.concat(frames,ignore_index=True)

    return result.to_crs({'init': 'epsg:2019'})

def centroid(item, value):

    if isinstance(item,shapely.geometry.polygon.Polygon):
        x = item.centroid.x
        y = item.centroid.y
    elif isinstance(item,shapely.geometry.point.Point):
        x = item.x
        y = item.y
    if value=="x":
        return x
    if value == "y":
        return y

def nia_filename(nias):
    NIAs=sorted(nias)
    NIAs_name = ''
    for i in range(len(NIAs)):
        NIAs_name+=str(NIAs[i])
        if i != (len(NIAs)-1):
            NIAs_name += "_"
    return NIAs_name

def road_points(root,outputfile="./preprocessing/road_end_points.txt",prec=2):

    # original copy in test.py
    # pednet_path="zip://data/pednet.zip"
    root="/Users/weimin/Documents/MASC/walkability_data"
    pednet_path = os.path.join(root,"pednet.zip")

    # reading pednet file
    pednet = gpd.read_file(pednet_path)
    print(pednet.crs)
    crs = {'init': 'epsg:4326'}
    pednet = gpd.GeoDataFrame(pednet, crs=crs, geometry='geometry')
    pednet = pednet.to_crs({'init': 'epsg:2019'})
    pednet = pednet[
        ['OBJECTID', 'road_type', 'sdwlk_code', 'sdwlk_desc', 'crosswalk', 'cwalk_type', 'px', 'px_type', 'geometry']]

    d = {'roadID': [], 'x': [], 'y': []}

    for i in range(len(pednet)):
        if i % 500 == 0:
            print("processing", i)
        x_list, y_list = pednet.iloc[i]["geometry"].coords.xy
        # get the two end points of each road segment
        d['x'].append(np.round(x_list[0], prec))
        d['y'].append(np.round(y_list[0], prec))
        d['roadID'].append(i + 1)
        d['x'].append(np.round(x_list[-1], prec))
        d['y'].append(np.round(y_list[-1], prec))
        d['roadID'].append(i + 1)

    with open(outputfile, 'w') as file:
        file.write(json.dumps(d))
    return

def road_nia_mapping(data_root, preprocessing_folder, outputfile):
    # code reference: https://github.com/gcc-dav-official-github/dav_cot_walkability/blob/master/code/TTC%20Walkability%20Tutorial.ipynb
    nia = gpd.read_file(os.path.join(data_root,"neighbourhood-improvement-areas-wgs84/NEIGHBOURHOOD_IMPROVEMENT_AREA_WGS84.shp"))
    nia.columns = map(str.lower, nia.columns)
    nia = nia[['area_id', 'area_s_cd', 'area_name', 'geometry']]

    # reprojecting epsg 4386 (wgs84) to epsg 2019 (mtm nad 27)
    crs = {'init': 'epsg:4326'}
    nia = gpd.GeoDataFrame(nia, crs=crs, geometry='geometry')
    nia = nia.to_crs({'init': 'epsg:2019'})

    nia_d = {'niaID': [], 'x_p': [], 'y_p': [], 'roadID': []}

    with open(os.path.join(preprocessing_folder,"road_end_point.txt"), 'r') as f:
        D = json.load(f)
    roadID = D['roadID']
    x_list = D['x']
    y_list = D['y']


    # assign road segments (end points) to census tract
    for row in range(len(nia)):
        nia_id = int(nia.iloc[0]["area_s_cd"]) # nia name
        poly = nia.iloc[row]['geometry']  # nia_boundary

        for j in range(len(x_list)):
            p = Point(x_list[j], y_list[j])
            if p.within(poly) == True:
                nia_d['niaID'].append(nia_id)
                nia_d['x_p'].append(x_list[j])
                nia_d['y_p'].append(y_list[j])
                nia_d['roadID'].append(roadID[j])

    with open(outputfile, 'w') as file:
        file.write(json.dumps(nia_d))
    return


def road_CT_mapping(CT_boundary_path="data/lct_000b16a_e/lct_000b16a_e.shp",
                    end_points_path="./preprocessing/pednet_points/end_points.txt",
                    save_path='./preprocessing/pednet_points/road_CT_mapping.txt'):
    boundary = gpd.read_file(CT_boundary_path)
    print(boundary.crs)
    crs = {'init': 'epsg:3347'}
    boundary = gpd.GeoDataFrame(boundary, crs=crs, geometry='geometry')
    boundary = boundary[boundary["CMANAME"] == "Toronto"]
    boundary = boundary.to_crs({'init': 'epsg:2019'})

    ct_d = {'CTNAME': [], 'x_p': [], 'y_p': [], 'roadID': []}

    with open(end_points_path, 'r') as f:
        D = json.load(f)
    roadID = D['roadID']
    x_list = D['x']
    y_list = D['y']

    all_CTs = list(boundary['CTNAME'].unique())

    # CMAUID=535 for all of these

    # assign road segments (end points) to census tract
    for row in range(len(boundary)):
        name = boundary.iloc[row]['CTNAME']  # census tract name
        poly = boundary.iloc[row]['geometry']  # census tract boundary
        print(row, name)
        for j in range(len(x_list)):
            p = Point(x_list[j], y_list[j])
            if p.within(poly) == True:
                ct_d['CTNAME'].append(name)
                ct_d['x_p'].append(x_list[j])
                ct_d['y_p'].append(y_list[j])
                ct_d['roadID'].append(roadID[j])

    with open(save_path, 'w') as file:
        file.write(json.dumps(ct_d))


def ct_nia_mapping(nia_path):
    df = pd.read_excel(nia_path)
    D={}
    for i in range(len(df)):
        id = int(df.iloc[i][0])
        ct = str(df.iloc[i][2])
        if not id in D.keys():
            D[id]={"name":df.iloc[i][1],"CTs":[ct[-6:-2]+'.'+ct[-2:]]}
        else:
            D[id]["CTs"].append(ct[-6:-2]+'.'+ct[-2:])
    return D

def map_back_allocate(allocations,df_to):
    allocated_nodes = [df_to.iloc[j]["node_ids"] for j in allocations]
    allocated_df = df_to.iloc[allocations]
    return allocated_nodes, allocated_df

def map_back_assign(assignments, df_from, df_to, dict):
    # assignments
    i_s = []
    j_s = []
    d_s = []
    i_id = []
    j_id = []
    a_id = []
    for (i, j, a) in assignments:
        i_s.append(i)
        j_s.append(j)
        d_s.append(dict[(i, j)])
        i_id.append(df_from.iloc[i]["node_ids"])
        j_id.append(df_to.iloc[j]["node_ids"])
        a_id.append(a)
    assign_D = {
        "i": i_s,
        "j": j_s,
        "i_id": i_id,
        "j_id": j_id,
        "a_id": a_id,
        "d_s": d_s,
    }
    return assign_D

if __name__ == "__main__":
    D=ct_nia_mapping()
    b=1


