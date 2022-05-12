from graph_utils import *
from map_utils import *
import matplotlib.pyplot as plt
from pathlib import Path
import networkx as nx
import geopandas as gpd

# def of tags: https://wiki.openstreetmap.org/wiki/Map_features#Others
#POI['geometry'] some are points, some are polygon

data_root="/Users/weimin/Documents/MASC/walkability_data"
preprocessing_folder="./preprocessing"
Path(preprocessing_folder).mkdir(parents=True, exist_ok=True)
Path(os.path.join(preprocessing_folder,'saved_nets')).mkdir(parents=True, exist_ok=True)

# get end points of roads
road_points(data_root,outputfile=os.path.join(preprocessing_folder,"road_end_point.txt"))

# road and nia mapping
#road_nia_mapping(data_root, preprocessing_folder, os.path.join(preprocessing_folder,"road_nia_mapping.txt"))

tag_parking={"amenity":"parking"}
tag_residential={"building":["apartments","bungalow","cabin","detached","dormitory","farm","ger","hotel","house","houseboat","residential","semidetached_house","static_caravan","terrace"]}
tag_grocery={"shop":"supermarket"}
tag_school={"amenity":"school"}
tag_mall={"shop":"mall"}
tag_coffee={"shop":"coffee"}
tag_restaurant={"amenity":"restaurant"}

pednet = load_pednet(data_root)

D_NIA = ct_nia_mapping(os.path.join(data_root,"neighbourhood-improvement-areas-wgs84/processed_TSNS 2020 NIA Census Tracts.xlsx"))

for nia_id in D_NIA.keys():
    print("processing nia",nia_id)

    prec=2
    pednet_nia = pednet_NIA(pednet,nia_id,preprocessing_folder)
    G = create_graph(pednet_nia, precision=prec)
    net_filename="NIA_%s_prec_%s.hd5" % (nia_id, prec)
    transit_ped_net=get_pandana_net(G,os.path.join(os.path.join(preprocessing_folder,'saved_nets'),net_filename))


    residentials_df = query_ox(get_NIAs_boundary(nia_id, "ox", data_root).geometry.values, tag_residential)
    malls_df=query_ox(get_NIAs_boundary(nia_id, "ox").geometry.values,tag_mall)
    parking_df=query_ox(get_NIAs_boundary(nia_id, "ox").geometry.values,tag_parking)
    grocery_df=query_ox(get_NIAs_boundary(nia_id, "ox").geometry.values,tag_grocery)
    school_df=query_ox(get_NIAs_boundary(nia_id, "ox").geometry.values,tag_school)

    # TODO: continue from here

    all_dfs=[residentials_df, parking_df, supermarket_df, school_df]

    for df in all_dfs:
        if len(df)>0:
            df['x']=df['geometry'].apply(centroid,args=('x',))
            df['y']=df['geometry'].apply(centroid,args=('y',))
            x, y = df.x, df.y
            df["node_ids"] = transit_ped_net.get_node_ids(x, y)

    ax = pednet_ct.plot(figsize=(15, 15), color='blue', markersize=1)
    residentials_df.plot(ax=ax,color='green', markersize=80)
    parking_df.plot(ax=ax,color='gray', markersize=80)
    supermarket_df.plot(ax=ax,color='purple', markersize=80)
    school_df.plot(ax=ax,color='yellow', markersize=80)


    df_filename = "NIA_%s_%s.pkl" % (nia_id, "parking")

    parking_df.to_pickle(os.path.join(os.path.join(preprocessing_folder,'saved_dfs'), df_filename))
    df_filename = "NIA_%s_%s.pkl" % (nia_id, "residential")
    residentials_df.to_pickle(os.path.join(os.path.join(preprocessing_folder,'saved_dfs'), df_filename))
    df_filename = "NIA_%s_%s.pkl" % (nia_id, "supermarket")
    supermarket_df.to_pickle(os.path.join(os.path.join(preprocessing_folder,'saved_dfs'), df_filename))

    SP_filename = "NIA_%s_prec_%s.txt" % (nia_id, prec)
    D = get_SP(transit_ped_net, save_path=os.path.join(os.path.join(preprocessing_folder,'saved_SPs'), SP_filename))

    # TODO: check max value, do some clean-up

    visual_path=os.path.join(preprocessing_folder,'visual')
    plt.savefig(os.path.join(visual_path,"nia_%s.png" % (nia_id)))
