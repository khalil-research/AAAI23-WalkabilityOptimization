import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np
from CP_models import *
from matplotlib.ticker import MaxNLocator

WALK_SCORE = True

def plot_obj(csv_name):
    # if WALK_SCORE:
    #     csv_name = "max_score_scratch_1.csv"
    # else:
    #     csv_name = "min_dist.csv"
    save_name=csv_name.split('.')[0]
    folder = "results/summary"
    df=pd.read_csv(os.path.join(folder,csv_name))
    if WALK_SCORE:
        df['avg_objective'] = df['Objective']
    # else:
    #     df['avg_objective']=df['Objective']/df['n']

    # plot objective
    #fig = plt.figure(figsize=(10,7))
    for nia in df["nia_id"].unique():

        # if nia==2:
        #     continue
        print(nia)
        new_df=df[df["nia_id"]==nia]
        x=list(new_df["k"])
        y=list(new_df["avg_objective"])
        name=list(new_df["nia_name"])[0]
        print(name)
        plt.plot(x,y,'--o',label=name)
    if WALK_SCORE:
        plt.title('Walk Score - Number of Resources Allocated')
        plt.xlabel('Number of Resources Allocated')
        plt.ylabel('Walk Score')
    else:
        plt.title('Averaged Walking Distance - Number of Resources Allocated')
        plt.xlabel('Number of Resources Allocated')
        plt.ylabel('Averaged Walking Distance (m)')

    #ax = plt.figure().gca()
    #ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.xticks([0,1,2,3])
    plt.legend()
    plt.savefig(os.path.join(folder,save_name+"_obj.png"))

def plot_time(csv_names,plot_save_name):
    folder = "results/summary"
    for csv_name in csv_names:
        save_name = csv_name.split('.')[0]
        df = pd.read_csv(os.path.join(folder, csv_name))
        df=df[df['k']>0]
        # plot solving time
        avg=df.groupby('nia_id').mean()
        #plt.tick_params(labelsize=26)
        #fig = plt.figure(figsize=(10,7))
        x = sorted(list(avg['m'] + avg['n']))
        y = sorted(list(avg['solving_time']))
        plt.plot(x,y,'--o',label=save_name)

    plt.xlabel('Input Size')
    # Set the y axis label of the current axis.
    plt.ylabel('Averaged Solving Time (s)')
    # Set a title of the current axes.
    plt.title('Averaged Solving Time - Input Size')
    plt.legend()
    plt.savefig(os.path.join(folder,plot_save_name+".png"))


def plot_score():
    # plot walk score
    folder = "results"
    d=np.linspace(0, 3000, 100)
    scores=dist_to_score(d,L_a,L_f_a)
    plt.plot(d, scores)
    plt.ylim([0, 100])
    plt.xlabel("l_n")
    plt.ylabel("f_n")
    plt.savefig(os.path.join(os.path.join(folder,"visualization"),"PWL.png"))
    return

def comp_obj(csv_names,plot_save_name):
    folder = "experiments_cp1_initial"
    for csv_name in csv_names:
        save_name = csv_name.split('.')[0]
        df = pd.read_csv(os.path.join(folder, csv_name))
        x = (list(df['m'] + df['n']))
        y = (list(df['Objective']))
        plt.plot(x,y,'--o',label=save_name)

    plt.xlabel('Input Size')
    # Set the y axis label of the current axis.
    plt.ylabel('Best objective value found')
    # Set a title of the current axes.
    plt.title('Best objective value found - Input size')
    plt.legend()
    plt.savefig(os.path.join(folder,plot_save_name+".png"))



if __name__ == "__main__":
    #comp_obj(["CP_adjusted_search_phase.csv","CP_defaulted_search_phase.csv","MIP_optimal.csv"],"obj_comp")
    #csv_name = "max_score_Gurobi_PWL.csv"
    #plot_obj(csv_name)
    #plot_time(["max_score_scratch_1.csv", "max_score_Gurobi_PWL.csv"], "sol_time_compare")
    #plot_time(["max_score_scratch_1.csv","max_score_scratch_2.csv","max_score_Gurobi_PWL.csv"],"sol_time_compare")

    plot_score()



