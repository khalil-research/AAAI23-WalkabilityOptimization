from graph_utils import *
from map_utils import *
#from MIP_models import *
from CP_models import *
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import argparse
import os
from pathlib import Path


parser = argparse.ArgumentParser(description='Enter model name:grb_PWL,scratch_1')
parser.add_argument("model", help="model",
                    type=str)
parser.add_argument("nias", help="nias to run",
                    type=str)
args = parser.parse_args()


D_NIA=ct_nia_mapping()
EXISTING_AMENITIES=True
WALK_SCORE = True
#NIA_LIST=[3,25,28,44,55]
#NIA_LIST=D_NIA.keys()
max_k = 3
min_k = 2

with open(args.nias) as f:
    NIA_LIST = [[int(x) for x in line.split(',')]for line in f][0]
models_folder = "models"
results_folder = "results"
preprocessing_folder="preprocessing_pickle4"

allocated_folder = os.path.join(results_folder,os.path.join("map_back",args.model))
visual_folder = os.path.join(results_folder,os.path.join("visualization",args.model))
csv_folder = os.path.join(results_folder,"csvs")
Path(allocated_folder).mkdir(parents=True, exist_ok=True)
Path(visual_folder).mkdir(parents=True, exist_ok=True)
Path(csv_folder).mkdir(parents=True, exist_ok=True)
Path(models_folder).mkdir(exist_ok=True)

if WALK_SCORE:
    csv_name = args.model+'_'+args.nias.split('/')[1].split('.')[0]+'.csv'
else:
    csv_name = "min_dist.csv"
summary_df_filename = os.path.join(csv_folder, csv_name)


if __name__ == "__main__":

    pednet = load_pednet("zip://data/pednet.zip")
    nia_id_L = []
    nia_name_L = []
    Objective_L = []
    k_L = []
    solving_time_L = []
    m_list = []
    n_list = []
    total_num_nodes = []
    allocated_id = []
    if EXISTING_AMENITIES:
        cur_amenities=[]

    for nia_id in NIA_LIST:
        print("NIA ",nia_id)
        pednet_ct = pednet_CTs(pednet, D_NIA[nia_id]['CTs'])
        prec = 2
        G = create_graph(pednet_ct, precision=prec)
        net_filename = "NIA_%s_prec_%s.hd5" % (nia_id, prec)
        transit_ped_net = get_pandana_net(G, os.path.join(os.path.join(preprocessing_folder,'saved_nets'), net_filename))
        # all nodes in CT
        all_nodes = nodes_from_pandana_net(transit_ped_net)

        df_filename = "NIA_%s_%s.pkl" % (nia_id, "parking")
        parking_df = pd.read_pickle(os.path.join(os.path.join(preprocessing_folder,'saved_dfs'), df_filename))
        df_filename = "NIA_%s_%s.pkl" % (nia_id, "residential")
        residentials_df = pd.read_pickle(os.path.join(os.path.join(preprocessing_folder,'saved_dfs'), df_filename))
        df_filename = "NIA_%s_%s.pkl" % (nia_id, "supermarket")
        supermarkets_df = pd.read_pickle(os.path.join(os.path.join(preprocessing_folder,'saved_dfs'), df_filename))

        SP_filename = "NIA_%s_prec_%s.txt" % (nia_id, prec)
        D = get_SP(transit_ped_net, save_path=os.path.join(os.path.join(preprocessing_folder,'saved_SPs'), SP_filename))

        # print("k=0")
        #
        # if len(supermarkets_df)>0:
        #     assignments, distances, obj_value, solving_time, assigned_nodes=cur_assignments(residentials_df,supermarkets_df, D)
        #     scores=dist_to_score(distances, L_a, L_f_a)
        #     cur_score=np.mean(scores)
        #
        #     assigned_f_name = "NIA_%s_k_%s.csv" % (nia_id, 0)
        #     assigned_df = pd.DataFrame(assigned_nodes)
        #     assigned_df.to_csv(os.path.join(allocated_folder, assigned_f_name), index=False)
        # else:
        #     cur_score=0
        #     solving_time=0
        #
        # nia_id_L.append(nia_id)
        # nia_name_L.append(D_NIA[nia_id]['name'])
        # k_L.append(0)
        # Objective_L.append(cur_score)
        # solving_time_L.append(solving_time)
        # n_list.append(len(residentials_df))
        # m_list.append(len(parking_df))
        # total_num_nodes.append(len(all_nodes))
        # allocated_id.append('none')
        # if EXISTING_AMENITIES:
        #     cur_amenities.append(len(supermarkets_df))


        for k in range(min_k,max_k):
            print("k=",k)
            # baseline
            if EXISTING_AMENITIES:
                if WALK_SCORE:
                    if args.model=='cp1':
                        obj_value, solving_time, allocated_nodes, allocated_df, assigned_nodes, m = max_score_cp1(residentials_df, parking_df,
                                                                                           supermarkets_df, D, k)
                    elif args.model=='scratch_1':
                        obj_value, solving_time, allocated_nodes, allocated_df, assigned_nodes, m = max_walk_score_scratch_1(residentials_df, parking_df,
                                                                                           supermarkets_df, D, k)
                    else:
                        print("choose model name")

                    # save allocated results for mapping
                    allocated_f_name="NIA_%s_k_%s.pkl" % (nia_id,k)
                    assigned_f_name="NIA_%s_k_%s.csv" % (nia_id,k)
                    model_f_name = "NIA_%s_k_%s.lp" % (nia_id, k)
                    #allocated_nodes.to_pickle(os.path.join(allocated_folder,os.path.join("allocated_nodes") )
                    allocated_df.to_pickle(os.path.join(allocated_folder,allocated_f_name))
                    assigned_df = pd.DataFrame(assigned_nodes)
                    assigned_df.to_csv(os.path.join(allocated_folder,assigned_f_name), index=False)
                    #m.write(os.path.join(models_folder,model_f_name))
                else:
                    obj_value, solving_time, allocated_nodes, allocated_df = k_center2(residentials_df, parking_df,
                                                                                supermarkets_df, D, k)

            else:
                obj_value, solving_time, allocated_nodes, allocated_df = k_center(residentials_df,parking_df,D,k)

            nia_id_L.append(nia_id)
            nia_name_L.append(D_NIA[nia_id]['name'])
            k_L.append(k)
            Objective_L.append(obj_value)
            solving_time_L.append(solving_time)
            n_list.append(len(residentials_df))
            m_list.append(len(parking_df))
            total_num_nodes.append(len(all_nodes))
            allocated_id.append(allocated_nodes)
            if EXISTING_AMENITIES:
                cur_amenities.append(len(supermarkets_df))

            ax = pednet_ct.plot(figsize=(15, 15), color='blue', markersize=1)
            res = residentials_df.plot(ax=ax,color='green', markersize=80, label='Residential location')
            parking = parking_df.plot(ax=ax,color='gray', markersize=80, label='Parking lot') #,fontsize=20
            if EXISTING_AMENITIES:
                if len(supermarkets_df)>0:
                    supermarkets=supermarkets_df.plot(ax=ax, color='gold', markersize=120, label='Existing amenity')
            nodes_df=all_nodes.iloc[allocated_nodes]

            # plot before allocation
            # plt.title("Neighbourhood %s" % (D_NIA[nia_id]['name']))
            # plt.savefig(os.path.join(results_folder, "nia_%s_k_%s_before.png" % (nia_id, k)))

            # plot all nodes
            #all_nodes.plot(ax=ax, color='yellow', markersize=40)

            nodes_df.plot(ax=ax, color='red')
            allocated=allocated_df.plot(ax=ax, color='red', markersize=80, label='Allocated amenity')

            red_patch = mpatches.Patch(color='red', label='Allocated amenity')
            green_patch = mpatches.Patch(color='green', label='Residential location')
            gray_patch = mpatches.Patch(color='gray', label='Parking lot')
            gold_patch = mpatches.Patch(color='gold', label='Existing amenity')

            if EXISTING_AMENITIES and (len(supermarkets_df)>0):
                plt.legend(handles=[red_patch,green_patch,gray_patch,gold_patch])
            else:
                plt.legend(handles=[red_patch,green_patch,gray_patch])


            if k==1:
                plt.title("%s with %s Amenity Allocated" % (D_NIA[nia_id]['name'], k))
            else:
                plt.title("%s with %s Amenities Allocated" % (D_NIA[nia_id]['name'], k))

            if WALK_SCORE:
                fig_name = "nia_%s_k_%s_after.png" % (nia_id,k)
            else:
                fig_name = "min_dist_nia_%s_k_%s_after.png" % (nia_id,k)

            plt.savefig(os.path.join(visual_folder,fig_name))


        results_D={
                "nia_id":nia_id_L,
                "nia_name":nia_name_L,
                "k":k_L,
                "Objective":Objective_L,
                "solving_time":solving_time_L,
                "n":n_list,
                "m":m_list,
                "tot_nodes":total_num_nodes,
                "allocated_id":allocated_id
        }
        if EXISTING_AMENITIES:
            results_D["cur_amenities"]=cur_amenities
        summary_df = pd.DataFrame(results_D)
        summary_df.to_csv(summary_df_filename,index=False)


