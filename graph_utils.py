import pandas as pd
import geopandas as gpd
import pandas as pd
import os
import shapely
import numpy as np
import networkx as nx
import pandana as pdna
from shapely.ops import nearest_points
import h5py
from shapely.geometry import Polygon, LineString, Point, box
from sqlalchemy import *
from shapely.geometry import *
import json

preprocessing_folder="preprocessing_pickle4"

def load_pednet(data_root):
    pednet = gpd.read_file(os.path.join(data_root,"pednet.zip"))
    print(pednet.crs)
    crs = {'init': 'epsg:4326'}
    pednet = gpd.GeoDataFrame(pednet, crs=crs, geometry='geometry')
    pednet = pednet.to_crs({'init': 'epsg:2019'})
    pednet = pednet[
        ['OBJECTID', 'road_type', 'sdwlk_code', 'sdwlk_desc', 'crosswalk', 'cwalk_type', 'px', 'px_type', 'geometry']]
    return pednet

# from https://github.com/gcc-dav-official-github/dav_cot_walkability/blob/master/code/TTC%20Walkability%20Tutorial.ipynb
def create_graph(gdf, precision=3):

    '''Create a networkx given a GeoDataFrame of lines. Every line will
    correspond to two directional graph edges, one forward, one reverse. The
    original line row and direction will be stored in each edge. Every node
    will be where endpoints meet (determined by being very close together) and
    will store a clockwise ordering of incoming edges.
    '''

    G = nx.Graph()

    def make_node(coord, precision):
        return tuple(np.round(coord, precision))

    # Edges are stored as (from, to, data), where from and to are nodes.
    def add_edges(row, G):
        geometry = row.geometry
        coords = list(geometry.coords)
        geom_r = LineString(coords[::-1])
        coords_r = geom_r.coords
        start = make_node(coords[0], precision)
        end = make_node(coords[-1], precision)
        # Add forward edge
        fwd_attr = {}
        for k, v in row.items():
            fwd_attr[k] = v
        fwd_attr['forward'] = 1
        # fwd_attr['geometry']=  geometry
        fwd_attr['length'] = geometry.length

        fwd_attr['visited'] = 0

        G.add_edge(start, end, **fwd_attr)

    gdf.apply(add_edges, axis=1, args=[G])

    return G

# from https://github.com/gcc-dav-official-github/dav_cot_walkability/blob/master/code/TTC%20Walkability%20Tutorial.ipynb
def creat_pandana_net(G, save_path, save=True): #probably will not use

    # create a pandana net
    # get network "from" and "to" from nodes
    edges = nx.to_pandas_edgelist(G, 'from', 'to')
    to = edges['to'].tolist()
    fr = edges['from'].tolist()
    fr = list(set(fr))
    to = list(set(to))
    to.extend(fr)
    nodes = list(set(to))
    nodes = pd.DataFrame(nodes)
    nodes.columns = ['x', 'y']
    nodes['xy'] = nodes.apply(lambda z: (z.x, z.y), axis=1)

    # Assigning node ids to to_node and from_node

    nodes['id'] = nodes.index
    edges['to_node'] = edges['to'].map(nodes.set_index('xy').id)
    edges['from_node'] = edges['from'].map(nodes.set_index('xy').id)

    # creating pandana network

    transit_ped_net = pdna.Network(nodes["x"],
                                   nodes["y"],
                                   edges["from_node"],
                                   edges["to_node"],
                                   pd.DataFrame([edges['length']]).T,
                                   twoway=True)

    # saving walkability file is optional. It can be used in the next steps if you don't have transit_ped_net in memory
    if save==True:
        transit_ped_net.save_hdf5(save_path)
    return transit_ped_net

# adapted from https://github.com/gcc-dav-official-github/dav_cot_walkability/blob/master/code/TTC%20Walkability%20Tutorial.ipynb
def get_pandana_net(G, save_path):
    if not os.path.exists(save_path):
        # create a pandana net
        # get network "from" and "to" from nodes
        edges = nx.to_pandas_edgelist(G, 'from', 'to')
        to = edges['to'].tolist()
        fr = edges['from'].tolist()
        fr = list(set(fr))
        to = list(set(to))
        to.extend(fr)
        nodes = list(set(to))
        nodes = pd.DataFrame(nodes)
        nodes.columns = ['x', 'y']
        nodes['xy'] = nodes.apply(lambda z: (z.x, z.y), axis=1)

        # Assigning node ids to to_node and from_node

        nodes['id'] = nodes.index
        edges['to_node'] = edges['to'].map(nodes.set_index('xy').id)
        edges['from_node'] = edges['from'].map(nodes.set_index('xy').id)

        # creating pandana network

        transit_ped_net = pdna.Network(nodes["x"],
                                       nodes["y"],
                                       edges["from_node"],
                                       edges["to_node"],
                                       pd.DataFrame([edges['length']]).T,
                                       twoway=True)
        transit_ped_net.save_hdf5(save_path)
    else:
        transit_ped_net = pdna.Network.from_hdf5(save_path)
    return transit_ped_net

def pednet_CTs(pednet,CTs,mapping=os.path.join(preprocessing_folder,'pednet_points/road_CT_mapping.txt')):
    with open(mapping, 'r') as f:
        D = json.load(f)

    df_road=pd.DataFrame.from_dict(D)

    df_road=df_road[df_road["CTNAME"].isin(CTs)]
    pednet_ct = pednet[pednet['OBJECTID'].isin(list(df_road["roadID"].values))]

    pednet_ct=pednet_ct.reset_index()

    return pednet_ct

def pednet_NIA(pednet,nia,preprocessing_folder):
    mapping=os.path.join(preprocessing_folder,"road_nia_mapping.txt")
    with open(mapping, 'r') as f:
        D = json.load(f)
    df_road=pd.DataFrame.from_dict(D)
    df_road = df_road[int(nia.iloc[0]["area_s_cd"])==nia]
    pednet_nia = pednet[pednet['OBJECTID'].isin(list(df_road["roadID"].values))]
    pednet_nia=pednet_nia.reset_index()

    return pednet_nia

def nodes_census(pednet,ct,mapping=os.path.join(preprocessing_folder,'pednet_points/road_CT_mapping.txt')):
    with open(mapping, 'r') as f:
        D = json.load(f)
    CTs = D['CTNAME']
    x_p = D['x_p']
    y_p = D['y_p']
    roadID = D['roadID']

    roads_ct = []
    nodes_ct = []
    for i in range(len(CTs)):
        if CTs[i] == ct:
            roads_ct.append(roadID[i])
            nodes_ct.append(Point(x_p[i], y_p[i]))
    # simplification: take one end of the road as nodes
    nodes_ct = nodes_ct[::2]
    #pednet_ct = pednet[pednet['OBJECTID'].isin(roads_ct)]
    #nodes_ct_df = pd.DataFrame(nodes_ct, columns=['geometry'])
    #nodes_ct_df_g = gpd.GeoDataFrame(nodes_ct_df)
    #print(len(nodes_ct))
    #ax_2 = pednet_ct.plot(figsize=(15, 15), color='blue', markersize=1)
    #nodes_ct_df_g.plot(ax=ax_2, color='red')
    #plt.show()
    return nodes_ct

def nodes_from_pandana_net(transit_ped_net):
    nodes_df = transit_ped_net.nodes_df
    gdf = gpd.GeoDataFrame(
        nodes_df, geometry=gpd.points_from_xy(nodes_df.x, nodes_df.y))
    #return gdf, nodes_df.x, nodes_df.y
    return gdf

def nearest_panana_net(item, nodes):

    pts=nodes.geometry.unary_union

    if isinstance(item, shapely.geometry.polygon.Polygon):
        point = item.centroid
    elif isinstance(item, shapely.geometry.point.Point):
        point = item
    else:
        print("Unkown origin type !!!")
        return "unknown"

    return np.where(nodes.geometry == nearest_points(point, pts)[1])[0][0]

def get_SP(transit_ped_net,save_path):
    '''
    return a matrix with pre-computed SPs
    '''

    if not os.path.exists(save_path):
        print("starting computing SP")
        gdf = nodes_from_pandana_net(transit_ped_net)
        num_nodes = len(gdf)
        mat = np.zeros((num_nodes, num_nodes))
        for i in range(num_nodes):
            for j in range(num_nodes):
                mat[i,j]=transit_ped_net.shortest_path_length(i, j)
        print("finish computing SP")
        np.savetxt(save_path, mat)
    else:
        mat=np.loadtxt(save_path)
    return mat


if __name__ == "__main__":
    # creating network graph

    pednet = gpd.read_file("zip://data/pednet.zip")
    # pednet.head(2)
    print(pednet.crs)
    crs = {'init': 'epsg:4326'}
    pednet = gpd.GeoDataFrame(pednet, crs=crs, geometry='geometry')
    pednet = pednet.to_crs({'init': 'epsg:2019'})
    pednet = pednet[
        ['OBJECTID', 'road_type', 'sdwlk_code', 'sdwlk_desc', 'crosswalk', 'cwalk_type', 'px', 'px_type', 'geometry']]

    CT='0363.07'
    pednet_ct=pednet_census(pednet,CT)

    G = create_graph(pednet_ct,precision=2)
    #G2=create_graph(pednet)
    transit_ped_net=creat_pandana_net(G,name=CT)


