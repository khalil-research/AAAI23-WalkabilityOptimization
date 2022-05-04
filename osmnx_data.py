#import osmnx as ox
from graph_utils import *
from map_utils import *
import matplotlib.pyplot as plt
import networkx as nx
import geopandas as gpd

# def of tags: https://wiki.openstreetmap.org/wiki/Map_features#Others
#POI['geometry'] some are points, some are polygon

preprocessing_folder="preprocessing_pickle4"
PARKING_TAG={"amenity":"parking"}
RESIDENTIAL_TAG={"building":["apartments","bungalow","cabin","detached","dormitory","farm","ger","hotel","house","houseboat","residential","semidetached_house","static_caravan","terrace"]}
SUPERMARKET_TAG={"shop":"supermarket"}
SCHOOL_TAG={"amenity":"school"}

pednet = load_pednet("zip://data/pednet.zip")
#D_NIA = ct_nia_mapping("./data/neighbourhood-improvement-areas-wgs84/processed_TSNS 2020 NIA Census Tracts.xlsx")
D_NIA = ct_nia_mapping("./data/neighbourhood-improvement-areas-wgs84/pieces.xlsx")
for nia_id in D_NIA.keys():
#for nia_id in [1000]:

    print(nia_id)
    if int(nia_id)>1119:

        CT=D_NIA[nia_id]['CTs']

        pednet_ct = pednet_CTs(pednet, CT)
        prec=2
        G = create_graph(pednet_ct, precision=prec)
        net_filename="NIA_%s_prec_%s.hd5" % (nia_id, prec)
        transit_ped_net=get_pandana_net(G,os.path.join(os.path.join(preprocessing_folder,'saved_nets'),net_filename))

        residentials_df=query_ox(get_CTs_boundary(CT, "ox").geometry.values,RESIDENTIAL_TAG)
        parking_df=query_ox(get_CTs_boundary(CT, "ox").geometry.values,PARKING_TAG)
        supermarket_df=query_ox(get_CTs_boundary(CT, "ox").geometry.values,SUPERMARKET_TAG)
        school_df=query_ox(get_CTs_boundary(CT, "ox").geometry.values,SCHOOL_TAG)

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

        visual_path=os.path.join(preprocessing_folder,'visual')
        plt.savefig(os.path.join(visual_path,"nia_%s.png" % (nia_id)))
