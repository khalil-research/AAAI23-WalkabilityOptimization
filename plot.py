import pandas as pd
import glob
import os
import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np
from CP_models import *
from matplotlib.ticker import MaxNLocator
from matplotlib import pyplot
import seaborn


basepath='important_results/optimal_ins/csvs'
fig_path='important_results/optimal_ins/figs'

basepath='results-10s-limit/csvs'
fig_path='results-10s-limit/figs'

thres=1

def get_num_instances():

    basepath1='results-1h-limit/csvs'
    basepath2 = 'results-big-1h-limit/csvs'
    model_names = ['CP_1', 'CP_1b', 'CP_2', 'CP_2b_no_x', 'MILP']
    all_dfs = load_dfs(basepath1, basepath2,model_names)
    df=all_dfs[4]
    df_large=df[(df['m'] + df['n']) > 600]
    df_small = df[(df['m'] + df['n']) <= 600]
    model=model_names[4]
    print(model)
    print("<=20")
    print(len(df.loc[df['m'] <= 20]))
    print(">20,<=40")
    print(len(df.loc[(df['m'] > 20) & (df['m'] <= 40)]))
    print(">40")
    print(len(df.loc[(df['m'] > 40)]))

    print("<=30")
    print(len(df.loc[df['m'] <= 30]))
    print(">30")
    print(len(df.loc[(df['m'] > 30)]))

    return

def get_num_optimal():
    L=[]
    #basepath1 = 'results-1h-limit/csvs'
    #basepath2 = 'results-big-1h-limit/csvs'
    basepath1 = 'results-1h-nolim/csvs'
    basepath2 = 'results-big-1h-nolim/csvs'
    model_names = ['CP_1', 'CP_1b', 'CP_2', 'CP_2b_no_x', 'MILP','MaxSAT']
    all_dfs = load_dfs(basepath1, basepath2, model_names)

    basepath1_2 = 'results-1h-limit/csvs'
    basepath2_2 = 'results-big-1h-limit/csvs'
    model_names = ['CP_1', 'CP_1b', 'CP_2', 'CP_2b_no_x', 'MILP','MaxSAT']
    all_dfs_2 = load_dfs(basepath1_2, basepath2_2, model_names)
    for i in range(len(all_dfs)):
        df=all_dfs[i]
        df2=all_dfs_2[i]
        model=model_names[i]
        print(model, "optimal")
        # print("<=20")
        # print(len(df.loc[df['m'] <= 20]))
        # print(">20,<=40")
        # print(len(df.loc[(df['m'] > 20) & (df['m'] <= 40)]))
        # print(">40")
        # print(len(df.loc[(df['m'] > 40)]))

        print("<=30")
        print(len(df.loc[df['m'] <= 30]))
        print(">30")
        print(len(df.loc[(df['m'] > 30)]))
        L.append([len(df.loc[(df['n']<=80) & (df['m']<=20)]),len(df.loc[(df2['n']<=80) & (df2['m']<=20)]),
                  len(df.loc[(df['n']>80) & (df['m']<=20)]),len(df2.loc[(df2['n']>80) & (df2['m']<=20)]),
                  len(df.loc[(df['n']<=80) & (df['m']>20)]),len(df2.loc[(df2['n']<=80) & (df2['m']>20)]),
                  len(df.loc[(df['n']>80) & (df['m']>20)]),len(df2.loc[(df2['n']>80) & (df2['m']>20)])])

    for i in range(len(all_dfs)):
        print([str])
        print(L[i])

    return


def load_dfs(basepath1, basepath2, names):

    all_dfs=[]

    for model in names:

        L = []
        all_files = glob.glob(os.path.join(basepath1,model) + "/*.csv")+glob.glob(os.path.join(basepath2,model) + "/*.csv")
        for filename in all_files:
            df = pd.read_csv(filename, index_col=None, header=0)
            L.append(df)
        data = pd.concat(L, axis=0, ignore_index=True)
        all_dfs.append(data)

    return all_dfs

def plot_quality(basepath, model_names,plot_save_name, folder):
    all_dfs = load_dfs(basepath, model_names)

    for i in range(len(all_dfs)):
        df=all_dfs[i]
        df=df[df['m']<15]
        model=model_names[i]
        obj_ = list(df["Objective"])
        obj=[]
        x = list(df['m'])
        for item in obj_:
            if not ((is_int(item)) or (is_float(item))):
                obj.append(float(item.split(",")[0][1:]))
            else:
                obj.append(item)

        sorted_x=sorted(x)
        sorted_time=[t for _, t in sorted(zip(x, obj))]
        plt.plot(sorted_x, sorted_time, '--o', label=model)
        # pure dots
        #plt.plot(sorted_x, sorted_time, 'o', label=model)
    plt.legend()
    plt.savefig(os.path.join(folder, plot_save_name + ".png"))
    return

def plot_time2(basepath, model_names,plot_save_name, folder):
    all_dfs = load_dfs(basepath, model_names)

    for i in range(len(all_dfs)):
        df=all_dfs[i]
        df=df[df['m']<15]
        model=model_names[i]
        time = list(df["solving_time"])
        x = list(df['m'])

        sorted_x=sorted(x)
        sorted_time=[t for _, t in sorted(zip(x, time))]
        plt.plot(sorted_x, sorted_time, '--o', label=model)
        # pure dots
        #plt.plot(sorted_x, sorted_time, 'o', label=model)
    plt.legend()
    plt.savefig(os.path.join(folder, plot_save_name + ".png"))
    return

def feas_bar(plot_save_name, folder):
    path1 = 'results-10s-limit/csvs'
    path2 = 'results-1h-limit/csvs'
    path3 = 'results-300s/csvs'
    path1_2 = 'results-big-10s-limit/csvs'
    path2_2 = 'results-big-1h-limit/csvs'
    path3_2 = 'results-big-300s-limit/csvs'
    model_names = ['CP_1', 'CP_1b', 'CP_2', 'CP_2b_no_x', 'MILP']
    all_dfs_10s = load_dfs(path1,path1_2, model_names)
    all_dfs_1h = load_dfs(path2,path2_2, model_names)
    all_dfs_300s = load_dfs(path3,path3_2, model_names)
    dfs_opt = load_dfs('results-1h-nolim/csvs','results-big-1h-nolim/csvs', model_names)

    D = dict(zip(list(dfs_opt[4]['nia_id']), list(dfs_opt[4]['Objective'])))
    D[1104]=80.3232540016098
    D[1105] = 22.158998045869
    D[22]= 67.215
    D[25]=62.1768340578029
    D[43]=68.6925
    D[138]=60.54262885188
    D[2]=56.4767189796151
    # here
    # and need to find max solution across everything
    # actually only need to manually add

    mre_10s=[]
    mre_1h=[]
    mre_300s=[]

    plt.rcParams["figure.figsize"] = (4, 4)

    for i in range(len(all_dfs_10s)):
        temp=[]
        df=all_dfs_10s[i]
        #df=df[df['n']>80]
        df = df[df['m'] >20]
        model=model_names[i]
        obj_ = list(df["Objective"])
        obj=[]
        x = list(df['m'])
        for item in obj_:
            if not ((is_int(item)) or (is_float(item))):
                obj.append(float(item.split(",")[0][1:]))
            else:
                obj.append(item)
        nia=list(df["nia_id"])
        for j in range(len(nia)):
            temp.append(abs(obj[j]-D[nia[j]])/D[nia[j]])
        mre_10s.append(np.mean(np.array(temp)))

    for i in range(len(all_dfs_1h)):
        temp = []
        df = all_dfs_1h[i]
        #df=df[df['n']>80]
        df = df[df['m'] > 20]
        model = model_names[i]
        obj_ = list(df["Objective"])
        obj = []
        x = list(df['m'])
        for item in obj_:
            if not ((is_int(item)) or (is_float(item))):
                obj.append(float(item.split(",")[0][1:]))
            else:
                obj.append(item)
        nia = list(df["nia_id"])
        for j in range(len(nia)):
            temp.append(abs(obj[j] - D[nia[j]])/D[nia[j]])
        mre_1h.append(np.mean(np.array(temp)))

    for i in range(len(all_dfs_300s)):
        temp = []
        df = all_dfs_300s[i]
        #df=df[df['n']>80]
        df = df[df['m'] > 20]
        model = model_names[i]
        obj_ = list(df["Objective"])
        obj = []
        x = list(df['m'])
        for item in obj_:
            if not ((is_int(item)) or (is_float(item))):
                obj.append(float(item.split(",")[0][1:]))
            else:
                obj.append(item)
        nia = list(df["nia_id"])
        for j in range(len(nia)):
            temp.append(abs(obj[j] - D[nia[j]])/D[nia[j]])
        mre_300s.append(np.mean(np.array(temp)))

    print(mre_1h,mre_10s)
    df_bar_left = pandas.DataFrame({
        'model': ['CP 1', 'CP 1b', 'CP 2', 'CP 2b', 'MILP'],
        '10s': mre_10s,
        '1h': mre_1h,
        '300s': mre_300s,
    })


    # MaxSAT in a different subplot

    model_names3 = ['MaxSAT']
    df = load_dfs('results-1h-nolim/csvs','results-big-1h-nolim/csvs', model_names3)[0]
    #df=df[df['n']>80]
    df = df[df['m'] > 20]

    mre=[]
    temp = []
    model = 'MaxSAT'
    obj_ = list(df["Objective"])
    obj = []
    x = list(df['m'])
    for item in obj_:
        if not ((is_int(item)) or (is_float(item))):
            obj.append(float(item.split(",")[0][1:]))
        else:
            obj.append(item)
    nia = list(df["nia_id"])
    for j in range(len(nia)):
        temp.append(abs(obj[j] - D[nia[j]]) / D[nia[j]])
    mre.append(np.mean(np.array(temp)))

    df_bar_right = pandas.DataFrame({
        'model': ['MaxSAT'],
        '1h': mre
    })

    f, (a0, a1) = plt.subplots(1, 2, gridspec_kw={'width_ratios': [10, 1]})
    df_bar_left.plot(x='model', y=['10s', '300s','1h'], kind='bar',ax=a0)

    df_bar_right.plot(x='model', y=['1h'], kind='bar', ax=a1,color=['purple'])
    a0.set_ylim(0, 0.18)
    a1.set_ylim(0, 0.18)

    # for container in a0.containers:
    #     plt.setp(container, width=0.3)
    for container in a1.containers:
        plt.setp(container, width=0.4)

    #a0.plot(x, y)
    #a1.plot(y, x)
    #


    f.tight_layout()
    plt.title('Mean relative error')
    plt.savefig(os.path.join(folder, plot_save_name + ".png"))


    return




def plot_time(dfs, model_names, plot_save_name, folder):
    plt.clf()
    print(x_axis)

    for i in range(len(dfs)):
        df=dfs[i]
        model=model_names[i]


        sorted_x=sorted(x)
        sorted_time=[t for _, t in sorted(zip(x, time))]
        #plt.plot(sorted_x, sorted_time, '--o', label=model)
        # pure dots
        plt.plot(sorted_x, sorted_time, 'o', label=model)
        L_less_20=[]
        L_geq_20=[]

        L_20_40 = []
        geq_40 = []
        print(model)

        for i in range(len(sorted_x)):
            if sorted_x[i]<20:
                L_less_20.append(sorted_time[i])
            else:
                L_geq_20.append(sorted_time[i])


            if (sorted_x[i] >= 20) and (sorted_x[i] < 40):
                L_20_40.append(sorted_time[i])
            if (sorted_x[i] >= 40):
                geq_40.append(sorted_time[i])

        print("name: ",model)
        print("tot instances:",len(df))
        print("avg_all:",np.mean(np.array(time)))
        print("avg_<20:", np.mean(np.array(L_less_20)))
        print("avg_>=20:", np.mean(np.array(L_geq_20)))
        print("max m",np.max(np.array(df["m"])))

        print("20-40",np.mean(np.array(L_20_40)))
        print("40+", np.mean(np.array(geq_40)))

    if x_axis == 'M+N':
        x_axis_name = "|M|+|N|"
    if x_axis == 'M':
        x_axis_name = "|M|"
    if x_axis == 'N':
        x_axis_name = "|N|"
    plt.xlabel(x_axis_name)
    # Set the y axis label of the current axis.
    plt.ylabel('Solving Time (s)')
    # Set a title of the current axes.
    if x_axis == 'M+N':
        title = "Solving Time - |M|+|N|"
    if x_axis == 'M':
        title = "Solving Time - |M|"
    if x_axis == 'N':
        title = "Solving Time - |N|"
    plt.title(title)
    plt.legend()
    plt.savefig(os.path.join(folder, plot_save_name + ".png"))

def plot_gap(dfs, model_names, x_axis,plot_save_name, folder):
    plt.clf()

    cp2_nias = dfs[2]["nia_id"]
    cp2_obj = []
    sat_obj = []

    cp2_obj_less_20 = []
    cp2_obj_geq_20 = []

    sat_obj_less_20 = []
    sat_obj_geq_20 = []

    sat_obj_20_33 = []
    sat_obj_geq_33 = []

    for i in range(len(cp2_nias)):
        item=dfs[2].iloc[i]["Objective"]
        if not ((is_int(item)) or (is_float(item))):
            print(item)
            cp2_obj.append(float(item.split(",")[0][1:]))
            if dfs[2].iloc[i]["m"]<20:
                cp2_obj_less_20.append(float(item.split(",")[0][1:]))
            else:
                cp2_obj_geq_20.append(float(item.split(",")[0][1:]))
        else:
            cp2_obj.append(item)
            if dfs[2].iloc[i]["m"]<20:
                cp2_obj_less_20.append(item)
            else:
                cp2_obj_geq_20.append(item)

        for j in range(len(dfs[3])):
            if dfs[3].iloc[j]["nia_id"] == cp2_nias[i]:
                sat_obj.append(dfs[3].iloc[j]["Objective"])
                if dfs[3].iloc[j]["m"] < 20:
                    sat_obj_less_20.append(dfs[3].iloc[j]["Objective"])
                else:
                    sat_obj_geq_20.append(dfs[3].iloc[j]["Objective"])

    for i in range(len(dfs)):
        df=dfs[i]
        model=model_names[i]
        if x_axis=='M+N':
            x = sorted(list(df['m'] + df['n']))
        if x_axis=='M':
            x = sorted(list(df['m']))
        if x_axis=='N':
            x = sorted(list(df['n']))

        if model == 'SAT':
            obj = sorted(list(df['Objective']))
        else:
            L_temp=list(df['Objective'])#float(dfs[2]['Objective'][0][1:8])
            L=[]
            for item in L_temp:
                if not ((is_int(item)) or (is_float(item))):
                    print(item)
                    L.append(float(item.split(",")[0][1:]))
                else:
                    L.append(item)


        sorted_x = sorted(x)
        sorted_obj = [o for _, o in sorted(zip(x, L))]
        # plt.plot(sorted_x, sorted_time, '--o', label=model)
        # pure dots
        plt.plot(sorted_x, sorted_obj, '--o', label=model)
        #plt.plot(x, obj, '--o', label=model)





#plot_gap([CP_1,CP_1b,CP_2,SAT],['CP_1','CP_1b','CP_2','MaxSAT'],'M+N',"gap",fig_path)
#plot_time([CP_1,CP_1b,CP_2,SAT],['CP_1','CP_1b','CP_2','MaxSAT'],'M+N',"time_mn3",fig_path)
#plot_time([CP_1,CP_1b,CP_2,SAT],['CP_1','CP_1b','CP_2','MaxSAT'],'M',"time_m3",fig_path)
#plot_time([CP_1,CP_1b,CP_2,SAT],['CP_1','CP_1b','CP_2','MaxSAT'],'N',"time_n3",fig_path)

#get_num_optimal()
get_num_instances()
#feas_bar('bar_ge20', 'results-plot')
#plot_time2('results-1h-nolim/csvs', ['CP_1', 'CP_1b', 'CP_2', 'CP_2b_no_x', 'MILP','MaxSAT'],'time', 'results-plot')