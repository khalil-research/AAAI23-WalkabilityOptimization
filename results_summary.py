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
        L_mip_status = []
        L_cp_status = []

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
                    #L_mip_status.append(mip_df["model_status"])
                else:
                    L_mip_obj.append(None)
                    #L_mip_status.append("null")
                # CP
                if os.path.exists(os.path.join(results_folder, "sol", "OptSingleCP_False_0", filename)):
                    cp_df = pd.read_csv(os.path.join(results_folder, "sol", "OptSingleCP_False_0", filename),index_col=None, header=0)
                    scores = dist_to_score(np.array(cp_df["dist"]), L_a, L_f_a)
                    score_obj = np.mean(scores)
                    L_cp_obj.append(score_obj)
                    #L_cp_status.append(cp_df["model_status"])
                else:
                    L_cp_obj.append(None)
                    #L_cp_status.append("null")

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
        L_mip_status = []
        L_cp_status = []

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
                    #L_mip_status.append(mip_df["model_status"])
                else:
                    L_mip_obj.append(None)
                    #L_mip_status.append("null")
                    # CP
                if os.path.exists(os.path.join(results_folder, "sol", "OptSingleDepthCP_False_0", filename)):
                    cp_df = pd.read_csv(os.path.join(results_folder, "sol", "OptSingleDepthCP_False_0", filename),index_col=None, header=0)
                    choices_dist = [cp_df[str(c) + "_dist"] if (str(c) + "_dist") in cp_df.columns else L_a[-2] for c in range(10)]
                    choices_dist = np.array(choices_dist)
                    weighted_choices = np.dot(np.array(choice_weights), choices_dist)
                    scores = dist_to_score(np.array(weighted_choices), L_a, L_f_a)
                    score_obj = np.mean(scores)
                    L_cp_obj.append(score_obj)
                    #L_cp_status.append(cp_df["model_status"])
                else:
                    L_cp_obj.append(None)
                    #L_cp_status.append("null")

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
    L_mip_status = []
    L_cp_status = []

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
                #L_mip_status.append(mip_df["model_status"])
            else:
                L_mip_obj.append(None)
                #L_mip_status.append("null")
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
                #L_cp_status.append(cp_df["model_status"])
            else:
                L_cp_obj.append(None)
                #L_cp_status.append("null")

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
    L_mip_status = []
    L_cp_status = []

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
                #L_mip_status.append(mip_df["model_status"])
            else:
                L_mip_obj.append(None)
                #L_mip_status.append("null")
            # CP
            if os.path.exists(os.path.join(results_folder, "sol", "OptMultipleDepthCP_False_0", filename)):
                cp_df = pd.read_csv(os.path.join(results_folder, "sol", "OptMultipleDepthCP_False_0", filename), index_col=None, header=0)
                dist_grocery = cp_df["dist_grocery"]
                choices_dist = [cp_df[str(c) + "_dist_restaurant"] if (str(c) + "_dist_restaurant") in mip_df.columns else L_a[-2] for c in range(10)]
                dist_school = cp_df["dist_school"]
                multiple_dist = np.array([dist_grocery] + choices_dist + [dist_school])
                weighted_dist = np.dot(np.array(weights_array_multi), multiple_dist)
                scores = dist_to_score(np.array(weighted_dist), L_a, L_f_a)
                score_obj = np.mean(scores)
                L_cp_obj.append(score_obj)
                #L_cp_status.append(cp_df["model_status"])
            else:
                L_cp_obj.append(None)
                #L_cp_status.append("null")

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


def single_aggregate_obj(data_root, results_folder,processed_folder):

    # NOTE: THIS IS A RELATXATION, AS IT ALLOWS COMPETING RESOURCES TO BE TOGETHER AND DISREGARDS THE CAPACITY CONSTRAINTS
    D_NIA = ct_nia_mapping(os.path.join(data_root, "neighbourhood-improvement-areas-wgs84/processed_TSNS 2020 NIA Census Tracts.xlsx"))

    # no depth
    print("single aggregate, no depth")

    L_nia = []
    L_mip_obj = []
    L_cp_obj = []
    L_greedy_obj = []
    L_k = []

    for nia in list(D_NIA.keys()):
        print("nia:",nia)
        for k in range(1,10):
            L_nia.append(nia)
            L_k.append(k)

            amenities = ["grocery", "restaurant", "school"]

            #  MIP
            all_type_dist=[]
            for amenity in amenities:
                filename = "assignment_NIA_%s_%s_%s.csv" % (nia,k,amenity)
                mip_df = pd.read_csv(os.path.join(results_folder,"sol","OptSingle_False_0",filename), index_col=None, header=0)
                all_type_dist.append(mip_df["dist"])
            multiple_dist = np.array(all_type_dist)
            weighted_dist = np.dot(np.array(weights_array), multiple_dist)
            scores = dist_to_score(np.array(weighted_dist), L_a, L_f_a)
            score_obj = np.mean(scores)
            L_mip_obj.append(score_obj)

            # CP
            all_type_dist = []
            for amenity in amenities:
                filename = "assignment_NIA_%s_%s_%s.csv" % (nia, k, amenity)
                cp_df = pd.read_csv(os.path.join(results_folder, "sol", "OptSingleCP_False_0", filename),index_col=None, header=0)
                all_type_dist.append(cp_df["dist"])
            multiple_dist = np.array(all_type_dist)
            weighted_dist = np.dot(np.array(weights_array), multiple_dist)
            scores = dist_to_score(np.array(weighted_dist), L_a, L_f_a)
            score_obj = np.mean(scores)
            L_cp_obj.append(score_obj)

            # Greedy
            all_type_dist = []
            for amenity in amenities:
                filename = "assignment_NIA_%s_%s_%s.csv" % (nia, k, amenity)
                greedy_df = pd.read_csv(os.path.join(results_folder, "sol", "GreedySingle_False_0", filename), index_col=None,header=0)
                all_type_dist.append(greedy_df["dist"])
            multiple_dist = np.array(all_type_dist)
            weighted_dist = np.dot(np.array(weights_array), multiple_dist)
            scores = dist_to_score(np.array(weighted_dist), L_a, L_f_a)
            score_obj = np.mean(scores)
            L_greedy_obj.append(score_obj)


        results_D = { "nia":L_nia, "k": L_k ,"mip":L_mip_obj,"cp":L_cp_obj,"greedy":L_greedy_obj}
        df_filename = os.path.join(processed_folder, "single_aggregate_no_depth.csv")
        summary_df = pd.DataFrame(results_D)
        summary_df["best"] = summary_df[["mip", "cp", "greedy"]].max(axis=1)
        summary_df.to_csv(df_filename,index=False)


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
        L_gap_mip.append(np.mean((obj_df['best'] - obj_df['mip'])/obj_df['best']))
        L_gap_cp.append(np.mean((obj_df['best'] - obj_df['cp'])/obj_df['best']))
        L_gap_greedy.append(np.mean((obj_df['best'] - obj_df['greedy'])/obj_df['best']))

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


def network_charac(data_root, results_folder,processed_folder):


    D_NIA = ct_nia_mapping(os.path.join(data_root, "neighbourhood-improvement-areas-wgs84/processed_TSNS 2020 NIA Census Tracts.xlsx"))

    L_nia = []

    L_m = []
    L_n = []

    L_num_grocery = []
    L_num_res = []
    L_num_school = []

    for nia in list(D_NIA.keys()):
        print("nia:",nia)

        filename = "NIA_%s_%s,%s,%s.csv" % (nia,1,1,1)
        L_nia.append(nia)

        #  MIP

        mip_df = pd.read_csv(os.path.join(results_folder,"summary","OptMultiple_False_0",filename), index_col=None, header=0)

        L_m.append(mip_df["num_parking"][0])
        L_n.append(mip_df["num_res"][0])
        L_num_grocery.append(mip_df["num_existing_L_grocery"][0])
        L_num_res.append(mip_df["num_existing_L_restaurant"][0])
        L_num_school.append(mip_df["num_existing_L_school"][0])

    results_D = { "nia":L_nia, "n": L_n ,"m":L_m,"grocery":L_num_grocery,"res":L_num_res,"school":L_num_school}
    df_filename = os.path.join(processed_folder, "nia_summary.csv")
    summary_df = pd.DataFrame(results_D)
    summary_df["m+n"]=summary_df["m"]+summary_df["n"]
    summary_df.to_csv(df_filename,index=False)


    # m_split = 60
    # n_split = 200

    # print("m>=80,n>=200",len(summary_df[(summary_df["m"] >= m_split) & (summary_df["n"]>=n_split)]))
    # print("m>=80,n<200", len(summary_df[(summary_df["m"] >= m_split) & (summary_df["n"] < n_split)]))
    # print("m<80,n>=200", len(summary_df[(summary_df["m"] < m_split) & (summary_df["n"] >= n_split)]))
    # print("m<80,n<00", len(summary_df[(summary_df["m"] < m_split) & (summary_df["n"] < n_split)]))

    # plt.scatter(L_m, L_n, s=np.array(L_num_grocery)**2, alpha=0.5)
    # plt.show()

    return

def plot_time_by_group_multiple(results_folder, plot_folder, models, display_names, save_name):
    group_gap = 1
    model_gap = 0.15
    plt.rcParams["figure.figsize"] = (4, 3)

    all_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']

    data_root = "/Users/weimin/Documents/MASC/walkability_data"
    D_NIA = ct_nia_mapping(
        os.path.join(data_root, "neighbourhood-improvement-areas-wgs84/processed_TSNS 2020 NIA Census Tracts.xlsx"))
    plt.clf()

    group_thres = [0,200,400,600,2000]

    def define_box_properties(plot_name, color_code, label):
        for k, v in plot_name.items():
            plt.setp(plot_name.get(k), color=color_code)

        for element in ['boxes', 'whiskers', 'fliers', 'means', 'medians', 'caps']:
            plt.setp(plot_name[element], color=color_code)
        plt.setp(plot_name["fliers"], markeredgecolor=color_code)
        # for element in ['medians']:
        #     plt.setp(plot_name[element], color='black')

        # for patch in plot_name['boxes']:
        #     patch.set(facecolor=color_code)

        # use plot function to draw a small line to name the legend.
        #plt.plot([], c=color_code, label=label)
        plt.plot([], label=label)
        plt.legend()


    for i in range(len(models)):

        model_name = models[i]
        display_name = display_names[i]

        results_df = get_results_df(results_folder, model_name)
        results_df = results_df[(results_df["k_L_grocery"]>0) & (results_df["k_L_restaurant"]>0) & (results_df["k_L_school"]>0)]
        results_df["m+n"]=results_df["num_res"]+results_df["num_parking"]

        # for nia in list(D_NIA.keys()):
        #     nia_df = results_df[results_df["nia_id"] == nia]
        #     if len(nia_df) < 9:
        #         print("missing for",nia)

        group_array=[]
        avg_time = results_df.groupby("nia_id")["solving_time"].apply(shifted_geo_mean)
        for p in range(len(group_thres)-1):

            group_df = results_df[(results_df["m+n"] >= group_thres[p]) & (results_df["m+n"] < group_thres[p+1])]
            group_array.append(avg_time[group_df['nia_id'].unique()])

        #model_plot = plt.boxplot(group_array,positions=np.array(np.arange(len(group_array))) * 2.0 + i*0.35, widths=0.3,patch_artist=True)
        model_plot = plt.boxplot(group_array, positions=np.array(np.arange(len(group_array))) * group_gap + i * 0.2,widths=model_gap,medianprops = dict(linewidth=3.5))

        define_box_properties(model_plot, all_colors[i], display_name)

    # set the x label values
    ticks = ['[0,200)', '[200,400)', '[400,600)]', '[[600,inf)]']
    plt.xticks(np.arange(0, len(ticks) * group_gap, group_gap), ticks)

    plt.xlabel("|M|+|N|")
    plt.ylabel("Shifted geo mean")

    plt.title(save_name)
    plt.savefig(os.path.join(plot_folder,  save_name))
    return


def quality_table_by_group_multiple(processed_folder):

    summary_filename = os.path.join(processed_folder, "nia_summary.csv")
    summary_df = pd.read_csv(summary_filename)

    num_nia = []
    num_ins = []

    L_MILP_no_depth = []
    L_CP_no_depth = []
    L_greedy_no_depth = []

    L_MILP_depth = []
    L_CP_depth = []
    L_greedy_depth = []

    group_thres = [0, 200, 400, 600, 2000]
    groups_name = ['[0,200)', '[200,400)', '[400,600)]', '[[600,inf)]']

    for p in range(len(group_thres) - 1):
        group_df = summary_df[(summary_df["m+n"] >= group_thres[p]) & (summary_df["m+n"] < group_thres[p + 1])]
        num_nia.append(len(group_df))
        num_ins.append(len(group_df)*9)

    # no depth

    all_obj_df = pd.read_csv(os.path.join(processed_folder, "multiple.csv"), index_col=None, header=0)
    for p in range(len(group_thres) - 1):
        group_df = summary_df[(summary_df["m+n"] >= group_thres[p]) & (summary_df["m+n"] < group_thres[p + 1])]
        obj_group_df = all_obj_df[all_obj_df["nia"].isin(group_df['nia'])]
        L_MILP_no_depth.append(np.mean((obj_group_df['best'] - obj_group_df['mip']) / obj_group_df['best']))
        L_CP_no_depth.append(np.mean((obj_group_df['best'] - obj_group_df['cp']) / obj_group_df['best']))
        L_greedy_no_depth.append(np.mean((obj_group_df['best'] - obj_group_df['greedy']) / obj_group_df['best']))

    # depth of choice

    all_obj_df = pd.read_csv(os.path.join(processed_folder, "multiple_depth.csv"), index_col=None, header=0)
    for p in range(len(group_thres) - 1):
        group_df = summary_df[(summary_df["m+n"] >= group_thres[p]) & (summary_df["m+n"] < group_thres[p + 1])]
        obj_group_df = all_obj_df[all_obj_df["nia"].isin(group_df['nia'])]
        L_MILP_depth.append(np.mean((obj_group_df['best'] - obj_group_df['mip']) / obj_group_df['best']))
        L_CP_depth.append(np.mean((obj_group_df['best'] - obj_group_df['cp']) / obj_group_df['best']))
        L_greedy_depth.append(np.mean((obj_group_df['best'] - obj_group_df['greedy']) / obj_group_df['best']))

    output_D = { "|M|+|N|":groups_name,
                 "MILP":L_MILP_no_depth, "CP": L_CP_no_depth ,"Greedy":L_greedy_no_depth,
                 "MILP_d":L_MILP_depth, "CP_d": L_CP_depth ,"Greedy_d":L_greedy_depth}
    df_filename = os.path.join(processed_folder, "quality_by_group_multiple.csv")

    output_df = pd.DataFrame(output_D)

    output_df.loc[:, "MILP"] = output_df["MILP"].map('{:.4f}'.format)
    output_df.loc[:, "CP"] = output_df["CP"].map('{:.4f}'.format)
    output_df.loc[:, "Greedy"] = output_df["Greedy"].map('{:.4f}'.format)
    output_df.loc[:, "MILP_d"] = output_df["MILP_d"].map('{:.4f}'.format)
    output_df.loc[:, "CP_d"] = output_df["CP_d"].map('{:.4f}'.format)
    output_df.loc[:, "Greedy_d"] = output_df["Greedy_d"].map('{:.4f}'.format)



    output_df.to_csv(df_filename,index=False)

    return


def quality_table_by_k_multiple(processed_folder):

    summary_filename = os.path.join(processed_folder, "nia_summary.csv")
    summary_df = pd.read_csv(summary_filename)

    num_nia = []
    num_ins = []

    L_MILP_no_depth = []
    L_CP_no_depth = []
    L_greedy_no_depth = []

    L_MILP_depth = []
    L_CP_depth = []
    L_greedy_depth = []

    groups_name = list(range(1,10))


    # no depth

    all_obj_df = pd.read_csv(os.path.join(processed_folder, "multiple.csv"), index_col=None, header=0)
    for k in range(1,10):
        obj_group_df = all_obj_df[all_obj_df['k']==k]
        L_MILP_no_depth.append(np.mean((obj_group_df['best'] - obj_group_df['mip']) / obj_group_df['best']))
        L_CP_no_depth.append(np.mean((obj_group_df['best'] - obj_group_df['cp']) / obj_group_df['best']))
        L_greedy_no_depth.append(np.mean((obj_group_df['best'] - obj_group_df['greedy']) / obj_group_df['best']))

    # depth of choice

    all_obj_df = pd.read_csv(os.path.join(processed_folder, "multiple_depth.csv"), index_col=None, header=0)
    for k in range(1,10):
        obj_group_df = all_obj_df[all_obj_df['k']==k]
        L_MILP_depth.append(np.mean((obj_group_df['best'] - obj_group_df['mip']) / obj_group_df['best']))
        L_CP_depth.append(np.mean((obj_group_df['best'] - obj_group_df['cp']) / obj_group_df['best']))
        L_greedy_depth.append(np.mean((obj_group_df['best'] - obj_group_df['greedy']) / obj_group_df['best']))

    output_D = { "k":groups_name,
                 "MILP":L_MILP_no_depth, "CP": L_CP_no_depth ,"Greedy":L_greedy_no_depth,
                 "MILP_d":L_MILP_depth, "CP_d": L_CP_depth ,"Greedy_d":L_greedy_depth}
    df_filename = os.path.join(processed_folder, "quality_by_k_multiple.csv")
    output_df = pd.DataFrame(output_D)
    output_df.to_csv(df_filename,index=False)

    return

def quality_table_by_group_single(processed_folder):

    summary_filename = os.path.join(processed_folder, "nia_summary.csv")
    summary_df = pd.read_csv(summary_filename)

    num_nia = []
    num_ins = []

    L_MILP_no_depth = []
    L_CP_no_depth = []
    L_greedy_no_depth = []

    L_MILP_depth = []
    L_CP_depth = []
    L_greedy_depth = []

    group_thres = [0, 200, 400, 600, 2000]
    groups_name = ['[0,200)', '[200,400)', '[400,600)]', '[[600,inf)]']

    for p in range(len(group_thres) - 1):
        group_df = summary_df[(summary_df["m+n"] >= group_thres[p]) & (summary_df["m+n"] < group_thres[p + 1])]
        num_nia.append(len(group_df))
        num_ins.append(len(group_df)*9)

    # no depth

    all_obj_df = pd.read_csv(os.path.join(processed_folder, "multiple.csv"), index_col=None, header=0)
    for p in range(len(group_thres) - 1):
        group_df = summary_df[(summary_df["m+n"] >= group_thres[p]) & (summary_df["m+n"] < group_thres[p + 1])]
        obj_group_df = all_obj_df[all_obj_df["nia"].isin(group_df['nia'])]
        L_MILP_no_depth.append(100*np.mean((obj_group_df['best'] - obj_group_df['mip']) / obj_group_df['best']))
        L_CP_no_depth.append(100*np.mean((obj_group_df['best'] - obj_group_df['cp']) / obj_group_df['best']))
        L_greedy_no_depth.append(100*np.mean((obj_group_df['best'] - obj_group_df['greedy']) / obj_group_df['best']))

    # depth of choice

    all_obj_df = pd.read_csv(os.path.join(processed_folder, "multiple_depth.csv"), index_col=None, header=0)
    for p in range(len(group_thres) - 1):
        group_df = summary_df[(summary_df["m+n"] >= group_thres[p]) & (summary_df["m+n"] < group_thres[p + 1])]
        obj_group_df = all_obj_df[all_obj_df["nia"].isin(group_df['nia'])]
        L_MILP_depth.append(100*np.mean((obj_group_df['best'] - obj_group_df['mip']) / obj_group_df['best']))
        L_CP_depth.append(100*np.mean((obj_group_df['best'] - obj_group_df['cp']) / obj_group_df['best']))
        L_greedy_depth.append(100*np.mean((obj_group_df['best'] - obj_group_df['greedy']) / obj_group_df['best']))

    output_D = { "|M|+|N|":groups_name,
                 "MILP":L_MILP_no_depth, "CP": L_CP_no_depth ,"Greedy":L_greedy_no_depth,
                 "MILP_d":L_MILP_depth, "CP_d": L_CP_depth ,"Greedy_d":L_greedy_depth}
    df_filename = os.path.join(processed_folder, "quality_by_group_multiple.csv")
    output_df = pd.DataFrame(output_D)

    output_df.loc[:, "MILP"] = output_df["MILP"].map('{:.4f}'.format)
    output_df.loc[:, "CP"] = output_df["CP"].map('{:.4f}'.format)
    output_df.loc[:, "Greedy"] = output_df["Greedy"].map('{:.4f}'.format)
    output_df.loc[:, "MILP_d"] = output_df["MILP_d"].map('{:.4f}'.format)
    output_df.loc[:, "CP_d"] = output_df["CP_d"].map('{:.4f}'.format)
    output_df.loc[:, "Greedy_d"] = output_df["Greedy_d"].map('{:.4f}'.format)

    output_df.to_csv(df_filename,index=False)

    return output_df


def opt_feas_multiple(results_folder, processed_folder):

    output_D = {}

    models = ["OptMultiple_False_0","OptMultipleCP_False_0","OptMultipleDepth_False_0", "OptMultipleDepthCP_False_0"]
    display_names = ["MILP","CP","MILP,depth","CP,depth"]

    group_thres = [0,200,400,600,2000]

    for i in range(len(models)):

        model_name = models[i]
        display_name = display_names[i]

        output_D[display_name+" opt"]=[]
        output_D[display_name + " feas"] = []

        results_df = get_results_df(results_folder, model_name)
        results_df = results_df[(results_df["k_L_grocery"]>0) & (results_df["k_L_restaurant"]>0) & (results_df["k_L_school"]>0)]
        results_df["m+n"]=results_df["num_res"]+results_df["num_parking"]

        for p in range(len(group_thres)-1):

            group_df = results_df[(results_df["m+n"] >= group_thres[p]) & (results_df["m+n"] < group_thres[p+1])]
            num_feas = len(group_df)
            if "CP" in model_name:
                num_opt = len(group_df[group_df["model_status"] == "Optimal"])
            else:
                num_opt = len(group_df[group_df["model_status"].astype(int) == 2])
            output_D[display_name + " opt"].append(num_opt)
            output_D[display_name + " feas"].append(num_feas)

    df_filename = os.path.join(processed_folder, "num_ins_by_group.csv")
    output_df = pd.DataFrame(output_D)
    output_df.to_csv(df_filename, index=False)

    return output_df


def hist_distances(data_root, results_folder,processed_folder):
    D_NIA = ct_nia_mapping(os.path.join(data_root, "neighbourhood-improvement-areas-wgs84/processed_TSNS 2020 NIA Census Tracts.xlsx"))

    # opt multiple depth
    print("multiple depth case")
    L_nia = []
    L_mip_obj = []
    L_cp_obj = []
    L_greedy_obj = []
    L_k = []
    L_mip_status = []
    L_cp_status = []

    dist_grocery_before = []
    dist_res_1_before = []
    dist_res_2_before = []

    dist_grocery_after = []
    dist_res_1_after = []
    dist_res_2_after = []

    k = 4
    use = 'mip'

    for nia in list(D_NIA.keys()):
        print("nia:", nia)

        filename = "assignment_NIA_%s_%s,%s,%s.csv" % (nia, k, k, k)

        L_nia.append(nia)
        if use == 'mip':
            #  MIP
            if os.path.exists(os.path.join(results_folder, "sol", "OptMultipleDepth_False_0", filename)):
                mip_df = pd.read_csv(os.path.join(results_folder, "sol", "OptMultipleDepth_False_0", filename),index_col=None, header=0)
                dist_grocery = mip_df["dist_grocery"]
                choices_dist = [mip_df[str(c) + "_dist_restaurant"] if (str(c) + "_dist_restaurant") in mip_df.columns else L_a[-2] for c in range(10)]
                dist_school = mip_df["dist_school"]

            else:
                print("????????")

            dist_grocery_after += list(dist_grocery)
            dist_res_1_after += mip_df["0_dist_restaurant"]
            dist_res_2_after += mip_df["1_dist_restaurant"]

        else:
            # CP
            if os.path.exists(os.path.join(results_folder, "sol", "OptMultipleDepthCP_False_0", filename)):
                cp_df = pd.read_csv(os.path.join(results_folder, "sol", "OptMultipleDepthCP_False_0", filename), index_col=None, header=0)
                dist_grocery = cp_df["dist_grocery"]
                choices_dist = [cp_df[str(c) + "_dist_restaurant"] if (str(c) + "_dist_restaurant") in mip_df.columns else L_a[-2] for c in range(10)]
                dist_school = cp_df["dist_school"]

            else:
                print("????????")

            dist_grocery_after += list(dist_grocery)
            dist_res_1_after += cp_df["0_dist_restaurant"]
            dist_res_2_after += cp_df["1_dist_restaurant"]

    k = 0


    for nia in list(D_NIA.keys()):
        print("nia:", nia)

        filename = "assignment_NIA_%s_%s,%s,%s.csv" % (nia, k, k, k)

        if os.path.exists(os.path.join(results_folder, "sol", "GreedyMultipleDepth_False_0", filename)):
            greedy_df = pd.read_csv(os.path.join(results_folder, "sol", "GreedyMultipleDepth_False_0", filename),
                                 index_col=None, header=0)
            dist_grocery = greedy_df["dist_grocery"]
            dist_school = greedy_df["dist_school"]
        else:
            print("????????")

        dist_grocery_before += list(dist_grocery)
        dist_res_1_before += greedy_df["0_dist_restaurant"]
        dist_res_2_before += greedy_df["1_dist_restaurant"]

    # make histogram with the before and after arrays


    return



if __name__ == "__main__":

    results_folder = "saved_results"
    plot_folder = "results_plot"
    data_root = "/Users/weimin/Documents/MASC/walkability_data"
    processed_folder= "processed_results"
    # for model_name in ["OptSingleCP_False_0"]:
    #     if model_name in ["OptSingle_False_0","OptSingleCP_False_0"]:
    #         amenity_L=["restaurant", "grocery", "school"]
    #     elif model_name=="OptSingleDepth_False_0":
    #         amenity_L = ["restaurant"]
    #     for amenity in amenity_L:
    #         results_df = get_results_df(results_folder, model_name, amenity)
    #         plot_obj_vs_k(results_df, plot_folder,amenity,model_name, model_name.split("_")[0])


    # plot solving time (shifted geometric mean)
    # plot_time_vs_size_single(results_folder, os.path.join(plot_folder,"time"), ["OptSingle_False_0", "OptSingleCP_False_0","GreedySingle_False_0"],
    #                          ["MILP",  "CP", "Greedy"],"Single amenity case without depth: shifted geo mean - input size")
    # plot_time_vs_size_single(results_folder, os.path.join(plot_folder, "time"), ["OptSingleDepth_False_0", "OptSingleDepthCP_False_0","GreedySingleDepth_False_0"],
    #                          ["MILP", "CP", "Greedy"], "Single amenity case with depth of choice: shifted geo mean - input size")
    # plot_time_vs_size_multiple(results_folder, os.path.join(plot_folder,"time"),["OptMultiple_False_0","OptMultipleCP_False_0","GreedyMultiple_False_0"],
    #                          ["MILP","CP", "Greedy"], "Multiple amenity case without depth: shifted geo mean - input size")
    # plot_time_vs_size_multiple(results_folder, os.path.join(plot_folder, "time"), ["OptMultipleDepth_False_0", "OptMultipleDepthCP_False_0","GreedyMultipleDepth_False_0"],
    #                           ["MILP", "CP", "Greedy"], "Multiple amenity case with depth of choice: shifted geo mean - input size")

    all_instances_obj(data_root, results_folder, "processed_results")
    # # plot quality
    # plot_quality("processed_results")

    # network_charac(data_root, results_folder, "processed_results")

    # boxplots

    # plot_time_by_group_multiple(results_folder, os.path.join(plot_folder,"time"),["OptMultiple_False_0","OptMultipleCP_False_0","GreedyMultiple_False_0"],
    #                           ["MILP","CP", "Greedy"], "boxplot multiple")
    # plot_time_by_group_multiple(results_folder, os.path.join(plot_folder, "time"),
    #                             ["OptMultipleDepth_False_0", "OptMultipleDepthCP_False_0","GreedyMultipleDepth_False_0"],
    #                             ["MILP", "CP", "Greedy"], "boxplot multiple depth")
    #
    # # quality table
    quality_table_by_k_multiple(processed_folder)
    quality_table_by_group_multiple(processed_folder)
    # opt_feas_multiple(results_folder, processed_folder)
