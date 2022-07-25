from graph_utils import *
from map_utils import *
import model_latest
#from CP_models import *
#from MIP_models import *
from model_latest import opt_single, cur_assignment_single, opt_multiple, opt_single_depth, cur_assignment_single_depth, weights_array, dist_to_score, L_a, L_f_a, opt_multiple_depth, weights_array_multi, choice_weights, opt_single_CP, opt_multiple_CP,opt_single_depth_CP, opt_multiple_depth_CP
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import argparse
import os
from pathlib import Path
import numpy as np
import pickle
from greedy import greedy_multiple_depth, greedy_multiple, greedy_single, greedy_single_depth, get_nearest

data_root = "/Users/weimin/Documents/MASC/walkability_data"
D_NIA = ct_nia_mapping(os.path.join(data_root,"neighbourhood-improvement-areas-wgs84/processed_TSNS 2020 NIA Census Tracts.xlsx"))

models_folder = "models"
results_folder = "results"
results_folder = "saved_results"
plot_folder = "results_plot"
data_root = "/Users/weimin/Documents/MASC/walkability_data"
processed_folder= "processed_results"
preprocessing_folder = "./preprocessing"

model_name = "OptMultipleDepth_False_0"
sol_folder = os.path.join(results_folder,os.path.join("sol",model_name))

net_save_path = os.path.join(preprocessing_folder, 'saved_nets')
df_save_path = os.path.join(preprocessing_folder, 'saved_dfs')
sp_save_path = os.path.join(preprocessing_folder, 'saved_SPs')

nia=43
k_name = 2

myhatch = 'O'
myscale = 4


if __name__ == "__main__":


    pednet = load_pednet(data_root)
    nia_id_L = []
    nia_name_L = []
    obj_L = []
    solving_time_L = []
    num_residents_L = []
    num_allocations_L = []
    status_L = []



    pednet_nia = pednet_NIA(pednet, nia, preprocessing_folder)


    # # load net
    prec = 2

    # load dfs
    all_strs = ['residential', 'mall', 'parking', 'grocery', 'school', 'coffee', 'restaurant']
    colors = ['g', 'lightcoral', 'grey', 'red', 'yellow', 'brown', 'orange']
    df_filenames = ["NIA_%s_%s.pkl" % (nia, str) for str in all_strs]
    all_dfs = [pd.read_pickle(os.path.join(df_save_path, df_filename)) for df_filename in df_filenames]
    residentials_df, malls_df, parking_df, grocery_df, school_df, coffee_df, restaurant_df = all_dfs

    # load SP
    SP_filename = "NIA_%s_prec_%s.txt" % (nia, prec)
    D = np.loadtxt(os.path.join(sp_save_path, SP_filename))

    allocated_f_name = os.path.join(sol_folder, "allocation_NIA_%s_%s,%s,%s.pkl" % (nia, k_name,k_name,k_name))
    #pd.DataFrame.from_dict(allocated_D).to_csv(allocated_f_name)
    with open(allocated_f_name, 'rb') as f:
        sol = pickle.load(f)

    ax = pednet_nia.plot(figsize=(15, 15), color='blue', markersize=1)
    res = residentials_df.plot(ax=ax,color='green', markersize=80, label='Residential location')
    parking = parking_df.plot(ax=ax,color='gray', markersize=80, label='Parking lot') #,fontsize=20
    grocery = grocery_df.plot(ax=ax,color='red', markersize=80, label='Existing grocery')
    res = restaurant_df.plot(ax=ax, color='orange', markersize=80, label='Existing restaurant')
    school = school_df.plot(ax=ax, color='yellow', markersize=80, label='Existing school')


    allocated_grocery = parking_df.iloc[sol["allocate_row_id_grocery"]]
    allocated_grocery2=allocated_grocery.copy()
    allocated_grocery2["geometry"] = allocated_grocery["geometry"].scale(myscale,myscale, origin='center')
    allocated_grocery2.plot(ax=ax, edgecolor='black',facecolor='red', hatch=myhatch,markersize=80, label='Allocated grocery')

    allocated_res = parking_df.iloc[sol["allocate_row_id_restaurant"]]
    allocated_res2 = allocated_res.copy()
    allocated_res2["geometry"] = allocated_res2["geometry"].scale(myscale, myscale, origin='center')
    allocated_res2.plot(ax=ax,  edgecolor='black',facecolor='orange', hatch=myhatch, markersize=80, label='Allocated restaurant')

    allocated_school = parking_df.iloc[sol["allocate_row_id_school"]]
    allocated_school2 = allocated_school.copy()
    allocated_school2["geometry"] = allocated_school2["geometry"].scale(myscale, myscale, origin='center')
    allocated_school2.plot(ax=ax,  edgecolor='black',facecolor='yellow', hatch=myhatch, markersize=80, label='Allocated school')

    green_patch = mpatches.Patch(color='green', label='Residential location')
    gray_patch = mpatches.Patch(color='gray', label='Parking lot')
    red_patch = mpatches.Patch(color='red', label='Existing grocery')
    orange_patch = mpatches.Patch(color='orange', label='Existing restaurant')
    yellow_patch = mpatches.Patch(color='yellow', label='Existing school')

    red_patch2 = mpatches.Patch(facecolor='red',  edgecolor='black',hatch=myhatch, label='Allocated grocery')
    orange_patch2 = mpatches.Patch(facecolor='orange', edgecolor='black', hatch=myhatch, label='Allocated restaurant')
    yellow_patch2 = mpatches.Patch(facecolor='yellow',  edgecolor='black',hatch=myhatch, label='Allocated school')

    #last_patch = mpatches.Patch(color=colors[all_strs.index(args.amenity)], label='Existing amenity')

    plt.legend(handles=[gray_patch,green_patch,red_patch,orange_patch,yellow_patch,red_patch2,orange_patch2,yellow_patch2])


    plt.title("neighbourhood: %s" % (D_NIA[nia]['name']))

    fig_name = "nia_%s_%s_allocation.png" % (nia, k_name)


    plt.savefig(os.path.join(plot_folder,fig_name))




