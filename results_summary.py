import glob
import pandas as pd
import os
from map_utils import ct_nia_mapping
import matplotlib.pyplot as plt
import numpy as np
import json
from model_latest import opt_single, cur_assignment_single, opt_multiple, opt_single_depth, cur_assignment_single_depth, weights_array, dist_to_score, L_a, L_f_a, weights_array_multi, choice_weights

def shifted_geo_mean(L, s=1):
    a = np.array(L)
    shifted = a+s
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

def plot_time_vs_size_multiple(results_folder, plot_folder, models, display_names, save_name):

    data_root = "/Users/weimin/Documents/MASC/walkability_data"
    D_NIA = ct_nia_mapping(
        os.path.join(data_root, "neighbourhood-improvement-areas-wgs84/processed_TSNS 2020 NIA Census Tracts.xlsx"))
    plt.clf()


    for i in range(len(models)):

        model_name = models[i]
        display_name = display_names[i]

        results_df = get_results_df(results_folder, model_name)
        results_df = results_df[(results_df["k_L_grocery"]>0) & (results_df["k_L_restaurant"]>0) & (results_df["k_L_school"]>0)]
        for nia in list(D_NIA.keys()):
            nia_df = results_df[results_df["nia_id"] == nia]
            if len(nia_df) < 9:
                    print("missing for",nia)
        size = results_df.groupby("nia_id").mean()["num_res"] + results_df.groupby("nia_id").mean()["num_parking"]
        avg_time = results_df.groupby("nia_id")["solving_time"].apply(shifted_geo_mean)
        new_x, new_y = zip(*sorted(zip(size, avg_time)))

        plt.plot(new_x, new_y, '--o', label=display_name)
        plt.legend(prop={'size': 6})

    plt.xlabel("|M|+|N|")
    plt.ylabel("Shifted geo mean")
    plt.title(save_name)
    plt.savefig(os.path.join(plot_folder,  save_name))
    return

def plot_time_vs_size_single(results_folder, plot_folder, models, display_names, save_name):
    plt.clf()

    for i in range(len(models)):

        model_name = models[i]
        display_name = display_names[i]

        if "Depth" in model_name:
            amenities = ["restaurant"]
        else:
            amenities = ["grocery","restaurant","school"]

        for amenity in amenities:

            results_df = get_results_df(results_folder, model_name, amenity)
            results_df=results_df[results_df["k"]>0]
            size = results_df.groupby("nia_id").mean()["num_res"] + results_df.groupby("nia_id").mean()["num_parking"]
            avg_time = results_df.groupby("nia_id")["solving_time"].apply(shifted_geo_mean)
            new_x, new_y = zip(*sorted(zip(size, avg_time)))

            plt.plot(new_x, new_y, '--o', label=display_name + "-"+ amenity)
            plt.legend(prop={'size': 6})

    plt.xlabel("|M|+|N|")
    plt.ylabel("Shifted geo mean")
    plt.title(save_name)
    plt.savefig(os.path.join(plot_folder,  save_name))
    return


def plot_obj_vs_k(results_df, plot_folder, amenity_name, model_name, display_name):

    data_root = "/Users/weimin/Documents/MASC/walkability_data"
    D_NIA = ct_nia_mapping(os.path.join(data_root, "neighbourhood-improvement-areas-wgs84/processed_TSNS 2020 NIA Census Tracts.xlsx"))

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
                print(model_name,amenity_name,nia,len(nia_df))
        nia_df = nia_df.sort_values(by=['k'])
        k_list = nia_df["k"]
        score_list = nia_df["obj"]

        plt.plot(k_list, score_list, '--o', label=D_NIA[nia]['name'])
        plt.legend(prop={'size': 6})
        plt.xlabel("k")
        plt.ylabel("score")
    plt.savefig(os.path.join(plot_folder, amenity_name + "_"+ display_name + "_k_vs_score.png"))

    return


def all_instances_obj(data_root, results_folder,processed_folder):
    D_NIA = ct_nia_mapping(os.path.join(data_root, "neighbourhood-improvement-areas-wgs84/processed_TSNS 2020 NIA Census Tracts.xlsx"))

    # opt single
    print("single case")
    amenities = ["grocery", "restaurant", "school"]
    for amenity in amenities:
        L_nia = []
        L_mip_obj = []
        L_cp_obj = []
        L_greedy_obj = []
        L_k = []

        for nia in list(D_NIA.keys()):
            print("nia:",nia)
            for k in range(1,10):
                filename = "assignment_NIA_%s_%s_%s.csv" % (nia,k,amenity)

                L_nia.append(nia)
                L_k.append(k)
                #  MIP
                if os.path.exists(os.path.join(results_folder,"sol","OptSingle_False_0",filename)):
                    mip_df = pd.read_csv(os.path.join(results_folder,"sol","OptSingle_False_0",filename), index_col=None, header=0)
                    scores = dist_to_score(np.array(mip_df["dist"]), L_a, L_f_a)
                    score_obj = np.mean(scores)
                    L_mip_obj.append(score_obj)
                else:
                    L_mip_obj.append(None)
                # CP
                if os.path.exists(os.path.join(results_folder, "sol", "OptSingleCP_False_0", filename)):
                    cp_df = pd.read_csv(os.path.join(results_folder, "sol", "OptSingleCP_False_0", filename),index_col=None, header=0)
                    scores = dist_to_score(np.array(cp_df["dist"]), L_a, L_f_a)
                    score_obj = np.mean(scores)
                    L_cp_obj.append(score_obj)
                else:
                    L_cp_obj.append(None)

                # Greedy
                if os.path.exists(os.path.join(results_folder, "summary", "GreedySingle_False_0", "NIA_%s_%s_%s.csv" % (nia,k,amenity))):
                    greedy_df = pd.read_csv(os.path.join(results_folder, "summary", "GreedySingle_False_0", "NIA_%s_%s_%s.csv" % (nia,k,amenity)), index_col=None,header=0)
                    L_greedy_obj.append(greedy_df["obj"].values[0])
                else:
                    L_greedy_obj.append(None)

        results_D = { "nia":L_nia, "k": L_k ,"mip":L_mip_obj,"cp":L_cp_obj,"greedy":L_greedy_obj,}
        df_filename = os.path.join(processed_folder, "single_%s.csv" % (amenity))
        summary_df = pd.DataFrame(results_D)
        summary_df["best"] = summary_df[["mip", "cp", "greedy"]].max(axis=1)
        summary_df.to_csv(df_filename,index=False)

    # opt single depth
    print("single depth case")
    amenities = ["restaurant"]
    for amenity in amenities:
        L_nia = []
        L_mip_obj = []
        L_cp_obj = []
        L_greedy_obj = []
        L_k = []

        for nia in list(D_NIA.keys()):
            print("nia:", nia)
            for k in range(1,10):
                filename = "assignment_NIA_%s_%s_%s.csv" % (nia, k, amenity)

                L_nia.append(nia)
                L_k.append(k)
                #  MIP
                if os.path.exists(os.path.join(results_folder, "sol", "OptSingleDepth_False_0", filename)):
                    mip_df = pd.read_csv(os.path.join(results_folder, "sol", "OptSingleDepth_False_0", filename),index_col=None, header=0)
                    choices_dist = [mip_df[str(c)+"_dist"] if (str(c)+"_dist") in mip_df.columns else L_a[-2] for c in range(10)]
                    choices_dist = np.array(choices_dist)
                    weighted_choices = np.dot(np.array(choice_weights), choices_dist)
                    scores = dist_to_score(np.array(weighted_choices), L_a, L_f_a)
                    score_obj = np.mean(scores)
                    L_mip_obj.append(score_obj)
                else:
                    L_mip_obj.append(None)
                    # CP
                if os.path.exists(os.path.join(results_folder, "sol", "OptSingleDepthCP_False_0", filename)):
                    cp_df = pd.read_csv(os.path.join(results_folder, "sol", "OptSingleDepthCP_False_0", filename),index_col=None, header=0)
                    choices_dist = [cp_df[str(c) + "_dist"] if (str(c) + "_dist") in cp_df.columns else L_a[-2] for c in range(10)]
                    choices_dist = np.array(choices_dist)
                    weighted_choices = np.dot(np.array(choice_weights), choices_dist)
                    scores = dist_to_score(np.array(weighted_choices), L_a, L_f_a)
                    score_obj = np.mean(scores)
                    L_cp_obj.append(score_obj)
                else:
                    L_cp_obj.append(None)

                # Greedy
                if os.path.exists(os.path.join(results_folder, "summary", "GreedySingleDepth_False_0", "NIA_%s_%s_%s.csv" % (nia,k,amenity))):
                    greedy_df = pd.read_csv(os.path.join(results_folder, "summary", "GreedySingleDepth_False_0", "NIA_%s_%s_%s.csv" % (nia,k,amenity)),index_col=None, header=0)
                    L_greedy_obj.append(greedy_df["obj"].values[0])
                else:
                    L_greedy_obj.append(None)

        results_D = {"nia": L_nia, "k": L_k, "mip": L_mip_obj, "cp": L_cp_obj, "greedy": L_greedy_obj }
        df_filename = os.path.join(processed_folder, "single_depth_%s.csv" % (amenity))
        summary_df = pd.DataFrame(results_D)
        summary_df["best"] = summary_df[["mip", "cp", "greedy"]].max(axis=1)
        summary_df.to_csv(df_filename, index=False)

    # opt multiple
    print("multiple case")
    L_nia = []
    L_mip_obj = []
    L_cp_obj = []
    L_greedy_obj = []
    L_k = []

    for nia in list(D_NIA.keys()):
        print("nia:", nia)
        for k in range(1,10):
            filename = "assignment_NIA_%s_%s,%s,%s.csv" % (nia,k,k,k)

            L_nia.append(nia)
            L_k.append(k)
            #  MIP
            if os.path.exists(os.path.join(results_folder, "sol", "OptMultiple_False_0", filename)):
                mip_df = pd.read_csv(os.path.join(results_folder, "sol", "OptMultiple_False_0", filename),index_col=None, header=0)
                dist_grocery = mip_df["dist_grocery"]
                dist_restaurant = mip_df["dist_restaurant"]
                dist_school = mip_df["dist_school"]
                multiple_dist = np.array([dist_grocery, dist_restaurant, dist_school])
                weighted_dist = np.dot(np.array(weights_array), multiple_dist)
                scores = dist_to_score(np.array(weighted_dist), L_a, L_f_a)
                score_obj = np.mean(scores)
                L_mip_obj.append(score_obj)
            else:
                L_mip_obj.append(None)
            # CP
            if os.path.exists(os.path.join(results_folder, "sol", "OptMultipleCP_False_0", filename)):
                cp_df = pd.read_csv(os.path.join(results_folder, "sol", "OptMultipleCP_False_0", filename),index_col=None, header=0)
                dist_grocery = cp_df["dist_grocery"]
                dist_restaurant = cp_df["dist_restaurant"]
                dist_school = cp_df["dist_school"]
                multiple_dist = np.array([dist_grocery, dist_restaurant, dist_school])
                weighted_dist = np.dot(np.array(weights_array), multiple_dist)
                scores = dist_to_score(np.array(weighted_dist), L_a, L_f_a)
                score_obj = np.mean(scores)
                L_cp_obj.append(score_obj)
            else:
                L_cp_obj.append(None)

            # Greedy
            if os.path.exists(os.path.join(results_folder, "summary", "GreedyMultiple_False_0", "NIA_%s_%s,%s,%s.csv" % (nia,k,k,k))):
                greedy_df = pd.read_csv(os.path.join(results_folder, "summary", "GreedyMultiple_False_0", "NIA_%s_%s,%s,%s.csv" % (nia,k,k,k)),index_col=None, header=0)
                L_greedy_obj.append(greedy_df["obj"].values[0])
            else:
                L_greedy_obj.append(None)

        results_D = {"nia": L_nia, "k": L_k, "mip": L_mip_obj, "cp": L_cp_obj, "greedy": L_greedy_obj}
        df_filename = os.path.join(processed_folder, "multiple.csv" )
        summary_df = pd.DataFrame(results_D)
        summary_df["best"] = summary_df[["mip", "cp", "greedy"]].max(axis=1)
        summary_df.to_csv(df_filename, index=False)

    # opt multiple depth
    print("multiple depth case")
    L_nia = []
    L_mip_obj = []
    L_cp_obj = []
    L_greedy_obj = []
    L_k = []

    for nia in list(D_NIA.keys()):
        print("nia:", nia)
        for k in range(1,10):
            filename = "assignment_NIA_%s_%s,%s,%s.csv" % (nia, k, k, k)

            L_nia.append(nia)
            L_k.append(k)
            #  MIP
            if os.path.exists(os.path.join(results_folder, "sol", "OptMultipleDepth_False_0", filename)):
                mip_df = pd.read_csv(os.path.join(results_folder, "sol", "OptMultipleDepth_False_0", filename),index_col=None, header=0)
                dist_grocery = mip_df["dist_grocery"]
                choices_dist = [mip_df[str(c) + "_dist_restaurant"] if (str(c) + "_dist_restaurant") in mip_df.columns else L_a[-2] for c in range(10)]
                dist_school = mip_df["dist_school"]
                multiple_dist = np.array([dist_grocery] + choices_dist + [dist_school])
                weighted_dist = np.dot(np.array(weights_array_multi), multiple_dist)
                scores = dist_to_score(np.array(weighted_dist), L_a, L_f_a)
                score_obj = np.mean(scores)
                L_mip_obj.append(score_obj)
            else:
                L_mip_obj.append(None)
            # CP
            if os.path.exists(os.path.join(results_folder, "sol", "OptMultipleDepthCP_False_0", filename)):
                cp_df = pd.read_csv(os.path.join(results_folder, "sol", "OptMultipleDepthCP_False_0", filename), index_col=None, header=0)
                dist_grocery = cp_df["dist_grocery"]
                choices_dist = [mip_df[str(c) + "_dist_restaurant"] if (str(c) + "_dist_restaurant") in mip_df.columns else L_a[-2] for c in range(10)]
                dist_school = cp_df["dist_school"]
                multiple_dist = np.array([dist_grocery] + choices_dist + [dist_school])
                weighted_dist = np.dot(np.array(weights_array_multi), multiple_dist)
                scores = dist_to_score(np.array(weighted_dist), L_a, L_f_a)
                score_obj = np.mean(scores)
                L_cp_obj.append(score_obj)
            else:
                L_cp_obj.append(None)

            # Greedy
            if os.path.exists(os.path.join(results_folder, "summary", "GreedyMultipleDepth_False_0", "NIA_%s_%s,%s,%s.csv" % (nia,k,k,k))):
                greedy_df = pd.read_csv(os.path.join(results_folder, "summary", "GreedyMultipleDepth_False_0", "NIA_%s_%s,%s,%s.csv" % (nia,k,k,k)),index_col=None, header=0)
                L_greedy_obj.append(greedy_df["obj"].values[0])
            else:
                L_greedy_obj.append(None)

        results_D = {"nia": L_nia, "k": L_k, "mip": L_mip_obj, "cp": L_cp_obj, "greedy": L_greedy_obj}
        df_filename = os.path.join(processed_folder, "multiple_depth.csv")
        summary_df = pd.DataFrame(results_D)
        summary_df["best"] = summary_df[["mip", "cp","greedy"]].max(axis=1)
        summary_df.to_csv(df_filename, index=False)

    return

def plot_quality(processed_folder):

    data_root = "/Users/weimin/Documents/MASC/walkability_data"
    D_NIA = ct_nia_mapping(os.path.join(data_root, "neighbourhood-improvement-areas-wgs84/processed_TSNS 2020 NIA Census Tracts.xlsx"))
    plt.clf()

    # single case

    for amenity in ["grocery","restaurant","school"]:
        L_size = []
        L_gap_mip = []
        L_gap_cp = []
        L_gap_greedy = []
        all_obj_df = pd.read_csv(os.path.join(processed_folder, "single_" + amenity+".csv"), index_col=None, header=0)
        for nia in list(D_NIA.keys()):
            results_df = get_results_df(results_folder, "OptSingle_False_0", amenity)
            results_df = results_df[results_df["nia_id"] == nia]
            obj_df = all_obj_df[all_obj_df["nia"]==nia]

            L_size.append(results_df["num_res"].mean() + results_df["num_parking"].mean())
            L_gap_mip.append(np.mean(obj_df['best']-obj_df['mip']))
            L_gap_cp.append(np.mean(obj_df['best'] - obj_df['cp']))
            L_gap_greedy.append(np.mean(obj_df['best'] - obj_df['greedy']))

        new_x, new_gap_mip = zip(*sorted(zip(L_size, L_gap_mip)))
        new_x, new_gap_cp = zip(*sorted(zip(L_size, L_gap_cp)))
        new_x, new_gap_greedy = zip(*sorted(zip(L_size, L_gap_greedy)))

        plt.plot(new_x, new_gap_mip, '--o', label="MILP" + "-"+ amenity)
        plt.plot(new_x, new_gap_cp, '--o', label="CP" + "-" + amenity)
        plt.plot(new_x, new_gap_greedy, '--o', label="Greedy" + "-" + amenity)
        plt.legend(prop={'size': 6})

        plt.xlabel("|M|+|N|")
        plt.ylabel("Relative error")
        plt.title("solution quality - single amenity case")
        plt.savefig(os.path.join(plot_folder, "quality", "single"))

    plt.clf()

    for amenity in ["restaurant"]:
        L_size = []
        L_gap_mip = []
        L_gap_cp = []
        L_gap_greedy = []
        all_obj_df = pd.read_csv(os.path.join(processed_folder, "single_depth_" + amenity+".csv"), index_col=None, header=0)
        for nia in list(D_NIA.keys()):
            results_df = get_results_df(results_folder, "OptSingleDepth_False_0", amenity)
            results_df = results_df[results_df["nia_id"] == nia]
            obj_df = all_obj_df[all_obj_df["nia"]==nia]

            L_size.append(results_df["num_res"].mean() + results_df["num_parking"].mean())
            L_gap_mip.append(np.mean(obj_df['best']-obj_df['mip']))
            L_gap_cp.append(np.mean(obj_df['best'] - obj_df['cp']))
            L_gap_greedy.append(np.mean(obj_df['best'] - obj_df['greedy']))

        new_x, new_gap_mip = zip(*sorted(zip(L_size, L_gap_mip)))
        new_x, new_gap_cp = zip(*sorted(zip(L_size, L_gap_cp)))
        new_x, new_gap_greedy = zip(*sorted(zip(L_size, L_gap_greedy)))

        plt.plot(new_x, new_gap_mip, '--o', label="MILP" + "-"+ amenity)
        plt.plot(new_x, new_gap_cp, '--o', label="CP" + "-" + amenity)
        plt.plot(new_x, new_gap_greedy, '--o', label="Greedy" + "-" + amenity)
        plt.legend(prop={'size': 6})

        plt.xlabel("|M|+|N|")
        plt.ylabel("Relative error")
        plt.title("solution quality - single amenity with depth of choice")
        plt.savefig(os.path.join(plot_folder, "quality", "single_depth"))


    # multiple amenity case
    plt.clf()

    L_size = []
    L_gap_mip = []
    L_gap_cp = []
    L_gap_greedy = []
    all_obj_df = pd.read_csv(os.path.join(processed_folder, "multiple.csv"), index_col=None, header=0)
    for nia in list(D_NIA.keys()):
        results_df = get_results_df(results_folder, "OptMultiple_False_0")
        results_df = results_df[results_df["nia_id"] == nia]
        obj_df = all_obj_df[all_obj_df["nia"] == nia]

        L_size.append(results_df["num_res"].mean() + results_df["num_parking"].mean())
        L_gap_mip.append(np.mean(obj_df['best'] - obj_df['mip']))
        L_gap_cp.append(np.mean(obj_df['best'] - obj_df['cp']))
        L_gap_greedy.append(np.mean(obj_df['best'] - obj_df['greedy']))

    new_x, new_gap_mip = zip(*sorted(zip(L_size, L_gap_mip)))
    new_x, new_gap_cp = zip(*sorted(zip(L_size, L_gap_cp)))
    new_x, new_gap_greedy = zip(*sorted(zip(L_size, L_gap_greedy)))

    plt.plot(new_x, new_gap_mip, '--o', label="MILP" + "-" + amenity)
    plt.plot(new_x, new_gap_cp, '--o', label="CP" + "-" + amenity)
    plt.plot(new_x, new_gap_greedy, '--o', label="Greedy" + "-" + amenity)
    plt.legend(prop={'size': 6})

    plt.xlabel("|M|+|N|")
    plt.ylabel("Relative error")
    plt.title("solution quality - multiple amenity case")
    plt.savefig(os.path.join(plot_folder, "quality", "multiple"))

    # multiple amenity case with depth of choice
    plt.clf()
    L_size = []
    L_gap_mip = []
    L_gap_cp = []
    L_gap_greedy = []
    all_obj_df = pd.read_csv(os.path.join(processed_folder, "multiple_depth.csv"), index_col=None, header=0)
    for nia in list(D_NIA.keys()):
        results_df = get_results_df(results_folder, "OptMultipleDepth_False_0")
        results_df = results_df[results_df["nia_id"] == nia]
        obj_df = all_obj_df[all_obj_df["nia"] == nia]

        L_size.append(results_df["num_res"].mean() + results_df["num_parking"].mean())
        L_gap_mip.append(np.mean(obj_df['best'] - obj_df['mip']))
        L_gap_cp.append(np.mean(obj_df['best'] - obj_df['cp']))
        L_gap_greedy.append(np.mean(obj_df['best'] - obj_df['greedy']))

    new_x, new_gap_mip = zip(*sorted(zip(L_size, L_gap_mip)))
    new_x, new_gap_cp = zip(*sorted(zip(L_size, L_gap_cp)))
    new_x, new_gap_greedy = zip(*sorted(zip(L_size, L_gap_greedy)))

    plt.plot(new_x, new_gap_mip, '--o', label="MILP" + "-" + amenity)
    plt.plot(new_x, new_gap_cp, '--o', label="CP" + "-" + amenity)
    plt.plot(new_x, new_gap_greedy, '--o', label="Greedy" + "-" + amenity)
    plt.legend(prop={'size': 6})

    plt.xlabel("|M|+|N|")
    plt.ylabel("Relative error")
    plt.title("solution quality - multiple amenity with depth of choice")
    plt.savefig(os.path.join(plot_folder, "quality", "multiple_depth"))

    return


if __name__ == "__main__":

    results_folder = "saved_results"
    plot_folder = "results_plot"
    data_root = "/Users/weimin/Documents/MASC/walkability_data"
    # for model_name in ["OptSingleCP_False_0"]:
    #     if model_name in ["OptSingle_False_0","OptSingleCP_False_0"]:
    #         amenity_L=["restaurant", "grocery", "school"]
    #     elif model_name=="OptSingleDepth_False_0":
    #         amenity_L = ["restaurant"]
    #     for amenity in amenity_L:
    #         results_df = get_results_df(results_folder, model_name, amenity)
    #         plot_obj_vs_k(results_df, plot_folder,amenity,model_name, model_name.split("_")[0])


    # plot solving time (shifted geometric mean)
    plot_time_vs_size_single(results_folder, os.path.join(plot_folder,"time"), ["OptSingle_False_0", "OptSingleCP_False_0","GreedySingle_False_0"],
                             ["MILP",  "CP", "Greedy"],"Single amenity case without depth: shifted geo mean - input size")
    plot_time_vs_size_single(results_folder, os.path.join(plot_folder, "time"), ["OptSingleDepth_False_0", "OptSingleDepthCP_False_0","GreedySingleDepth_False_0"],
                             ["MILP", "CP", "Greedy"], "Single amenity case with depth of choice: shifted geo mean - input size")
    plot_time_vs_size_multiple(results_folder, os.path.join(plot_folder,"time"),["OptMultiple_False_0","OptMultipleCP_False_0","GreedyMultiple_False_0"],
                             ["MILP","CP", "Greedy"], "Multiple amenity case without depth: shifted geo mean - input size")
    plot_time_vs_size_multiple(results_folder, os.path.join(plot_folder, "time"), ["OptMultipleDepth_False_0", "OptMultipleDepthCP_False_0","GreedyMultipleDepth_False_0"],
                              ["MILP", "CP", "Greedy"], "Multiple amenity case with depth of choice: shifted geo mean - input size")

    #all_instances_obj(data_root, results_folder, "processed_results")
    # # plot quality
    # plot_quality("processed_results")
