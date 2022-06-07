import glob
import pandas as pd
import os
from map_utils import ct_nia_mapping
import matplotlib.pyplot as plt
import numpy as np
import json

def shifted_geo_mean(L, s=1):
    a = np.array(L)
    shifted = a+1
    return (shifted.prod())**(1.0/len(a)) - s

def get_results_df(results_folder, model_name,amenity_name=None):

    L = []
    if amenity_name:
        all_files = glob.glob(os.path.join(results_folder,"summary",model_name) + "/*" + amenity_name + ".csv")
    else:
        all_files = glob.glob(os.path.join(results_folder, "summary", model_name) + "/*" + ".csv")
    for filename in all_files:
        df = pd.read_csv(filename, index_col=None, header=0)
        L.append(df)
    final_df = pd.concat(L, axis=0, ignore_index=True)

    return final_df

def plot_time_vs_size(results_folder, plot_folder, models, display_names):

    for model_name in models:

        if "Depth" in model_name:
            amenities = ["restaurant"]
        else:
            amenities = ["grocery","restaurant","school"]
        for amenity in amenities:
            plt.clf()
            results_df = get_results_df(results_folder, model_name, amenity)
            size = results_df.groupby("nia_id").mean()["num_res"] + results_df.groupby("nia_id").mean()["num_parking"]
            avg_time = results_df.groupby("nia_id")["solving_time"].apply(shifted_geo_mean)
            new_x, new_y = zip(*sorted(zip(size, avg_time)))

            plt.plot(new_x, new_y, '--o', label=model_name+amenity)
            plt.legend(prop={'size': 6})
            plt.xlabel("|M|+|N|")
            plt.ylabel("avg solving time (s) ")
            plt.savefig(os.path.join(plot_folder,  "time"+amenity + "_"+ model_name + ".png"))
    return


def plot_obj_vs_k(results_df, plot_folder, amenity_name, model_name, display_name):

    data_root = "/Users/weimin/Documents/MASC/walkability_data"
    D_NIA = ct_nia_mapping(
        os.path.join(data_root, "neighbourhood-improvement-areas-wgs84/processed_TSNS 2020 NIA Census Tracts.xlsx"))

    if 'Depth' in model_name:
        for ind in [0,1]:
            plt.clf()
            plt.figure(figsize=(15, 10))
            for nia in list(D_NIA.keys()):
                nia_df = results_df[results_df["nia_id"] == nia]
                nia_df = nia_df.sort_values(by=['k'])
                k_list = nia_df["k"]
                dist_list = [L[ind] for L in [json.loads(x) for x in nia_df["dist_obj"]]]
                dist_list = [None if item == 0 else item for item in dist_list]

                plt.plot(k_list, dist_list, '--o', label=D_NIA[nia]['name'])
                plt.legend(prop={'size': 6})
                plt.xlabel("k")
                plt.ylabel("dist (m)")
            plt.savefig(os.path.join(plot_folder, amenity_name + "_" + display_name + ("_k_vs_dist_%s.png" % (ind+1))))

    else:
        plt.clf()
        plt.figure(figsize=(15, 10))
        for nia in list(D_NIA.keys()):
            nia_df=results_df[results_df["nia_id"]==nia]
            nia_df=nia_df.sort_values(by=['k'])
            k_list = nia_df["k"]
            dist_list = nia_df["dist_obj"]

            plt.plot(k_list, dist_list, '--o', label=D_NIA[nia]['name'])
            plt.legend(prop={'size': 6})
            plt.xlabel("k")
            plt.ylabel("dist (m)")
        plt.savefig(os.path.join(plot_folder,  amenity_name + "_"+ display_name + "_k_vs_dist.png"))

    plt.clf()
    plt.figure(figsize=(15, 10))

    for nia in list(D_NIA.keys()):
        nia_df = results_df[results_df["nia_id"] == nia]
        if len(nia_df)<10:
            if len(nia_df[nia_df['k']==0])>0:
                print("missing values?")
                print(model_name,amenity,nia,len(nia_df))
        nia_df = nia_df.sort_values(by=['k'])
        k_list = nia_df["k"]
        score_list = nia_df["obj"]

        plt.plot(k_list, score_list, '--o', label=D_NIA[nia]['name'])
        plt.legend(prop={'size': 6})
        plt.xlabel("k")
        plt.ylabel("score")
    plt.savefig(os.path.join(plot_folder, amenity_name + "_"+ display_name + "_k_vs_score.png"))

    return


if __name__ == "__main__":

    results_folder = "saved_results"
    plot_folder = "results_plot"
    # for model_name in ["OptSingle_False_0", "OptSingleDepth_False_0"]:
    #     if model_name=="OptSingle_False_0":
    #         amenity_L=["restaurant", "grocery", "school"]
    #     elif model_name=="OptSingleDepth_False_0":
    #         amenity_L = ["restaurant"]
    #     for amenity in amenity_L:
    #         results_df = get_results_df(results_folder, model_name, amenity)
    #         plot_obj_vs_k(results_df, plot_folder,amenity,model_name, model_name.split("_")[0])

    plot_time_vs_size(results_folder, plot_folder, ["OptSingle_False_0", "OptSingleDepth_False_0"],["OptSingle", "OptSingleDepth"])
