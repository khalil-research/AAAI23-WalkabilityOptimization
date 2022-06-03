import glob
import pandas as pd
import os
from map_utils import ct_nia_mapping
import matplotlib.pyplot as plt

results_folder="results"

def get_results_df(results_folder, model_name, amenity):

    L = []
    all_files = glob.glob(os.path.join(results_folder,"summary",model_name) + "/*" + amenity + ".csv")
    for filename in all_files:
        df = pd.read_csv(filename, index_col=None, header=0)
        L.append(df)
    final_df = pd.concat(L, axis=0, ignore_index=True)
    return final_df

def plot_time_vs_size(results_df):
    return


def plot_obj_vs_k(results_df, save_folder, model_name):
    plt.clf()
    data_root = "/Users/weimin/Documents/MASC/walkability_data"
    D_NIA = ct_nia_mapping(
        os.path.join(data_root, "neighbourhood-improvement-areas-wgs84/processed_TSNS 2020 NIA Census Tracts.xlsx"))
    for nia in list(D_NIA.keys()):
        nia_df=results_df[results_df["nia_id"]==nia]
        nia_df=nia_df.sort_values(by=['k'])
        k_list = nia_df["k"]
        dist_list = nia_df["dist_obj"]

        plt.plot(k_list, dist_list, '--o', label=D_NIA[nia]['name'])
        plt.legend(prop={'size': 6})
        plt.xlabel("k")
        plt.ylabel("dist (m)")
        plt.savefig(os.path.join(save_folder, model_name + "_k_vs_dist.png"))

    plt.clf()

    for nia in list(D_NIA.keys()):
        nia_df = results_df[results_df["nia_id"] == nia]
        if len(nia_df)<10:
            print(amenity,nia,len(nia_df))
        nia_df = nia_df.sort_values(by=['k'])
        k_list = nia_df["k"]
        score_list = nia_df["obj"]

        plt.plot(k_list, score_list, '--o', label=D_NIA[nia]['name'])
        plt.legend(prop={'size': 6})
        plt.xlabel("k")
        plt.ylabel("score")
        plt.savefig(os.path.join(save_folder, model_name + "_k_vs_score.png"))
        # plt.clf()
        # plt.plot(k_list, score_list, '--o')
        # plt.savefig(os.path.join(save_folder, model_name + "_k_vs_score.png"))
    return


if __name__ == "__main__":
    model_name="OptSingle"
    for amenity in ["restaurant", "grocery", "school"]:
        results_df = get_results_df(results_folder, model_name, amenity)
        plot_obj_vs_k(results_df, "results_plot", model_name+"_"+amenity)
    a=1
    #TODO: why does some curve start at k=1 instead of k=0???