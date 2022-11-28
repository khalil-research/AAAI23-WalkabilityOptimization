#import gurobipy as gp
#from gurobipy import GRB
from itertools import product
import pandas as pd
import numpy as np
import copy
from map_utils import map_back_allocate, map_back_assign
import matplotlib.pyplot as plt
from docplex.cp.model import CpoModel
from docplex.cp.model import *
from ortools.sat.python import cp_model
from pysat.examples.rc2 import RC2
from pysat.formula import WCNF
from pysat.card import *
import time
from threading import Timer

# pieceweise linear interval and slopes
L_a=[0,400,1800,2400,5000000]
L_f_a=[100,95,10,0,0]
slope=[-0.0125, -0.0607, -0.0167] #[-0.0125, -0.06071428571428574, -0.016666666666666653]
intercept=[100.0000, 119.2857, 40.0000] #[100.00000000000001, 119.28571428571433, 39.999999999999964]

# amenity types: grocery, restaurant, shopping mall, coffee, school
# k=[2,4,2,5,2]
# use this for final
k = [1, 2, 2, 3, 1]
#k = [1, 1, 1, 1, 1]
w=[0.3,0.2,0.2,0.2,0.1]

#time_limit=9*60*60
time_limit=60*60
#time_limit=5




def CP_1(df_from,df_to,SP_matrix,solver_path, EPS=1.e-6):
    if len(df_from)>0:
        df_from = df_from[['geometry', 'node_ids']]
    if len(df_to)>0:
        df_to = df_to[['geometry', 'node_ids']]

    model = CpoModel(name="max_score")

    groups_to=df_to.groupby('node_ids').groups
    #group_keys_to=list(groups_to.keys())#[2,5,...]
    group_values_to=list(groups_to.values())

    groups_from = df_from.groupby('node_ids').groups
    #group_keys_from = list(groups_from.keys())  # [2,5,...]
    group_values_from = list(groups_from.values())


    # data
    num_residents = len(group_values_from)
    num_allocation = len(group_values_to)
    capacity = [len(item) for item in group_values_to]
    # indices
    cartesian_prod = list(product(range(num_residents), range(num_allocation)))
    # retrieve distances
    d = {(i, j): SP_matrix[df_from.iloc[group_values_from[i][0]]["node_ids"], df_to.iloc[group_values_to[j][0]]["node_ids"]] for i, j in cartesian_prod}


    # variables
    y = {}
    for j in range(num_allocation):
        for a in range(len(k)):
            y[(j, a)] = model.integer_var(min=0, max=k[a],name=f'y[{j},{a}]')
    x = {}
    for i in range(num_residents):
        for j in range(num_allocation):
            for a in range(len(k)):
                x[(i, j, a)] = model.binary_var(name=f'x[{i},{j},{a}]')
    f = {}
    for i in range(num_residents):
        f[i] = model.float_var(min=0, max=100, name=f'f[{i}]')
    l = {}
    for i in range(num_residents):
        l[i] = model.float_var(min=0, max=L_a[-1], name=f'z[{i}]')


    # Constraints
    # activation
    for i in range(num_residents):
        for j in range(num_allocation):
            for a in range(len(k)):
                model.add(x[(i,j,a)]<=y[(j,a)])
    # amenity upper bound
    for a in range(len(k)):
        model.add((model.sum(y[(j, a)] for j in range(num_allocation))) <= k[a])
    # capacity upper bound
    for j in range(num_allocation):
        model.add((model.sum(y[(j,a)] for a in range(len(k)))) <= capacity[j])
    # each resident visit one of each amenity
    for i in range(num_residents):
        for a in range(len(k)):
            model.add(model.sum(x[(i,j,a)] for j in range(num_allocation)) == 1)

    # side
    # side 1
    for j in range(num_allocation):
        model.add(model.if_then(model.logical_or((y[(j,0)]>=1),(y[(j,2)]>=1)), y[(j,3)]>=1))
    # side 2
        model.add(model.if_then(model.logical_and((y[(j, 0)] >= 1), (y[(j, 2)] >= 1)), y[(j, 1)] >= 1))

    # calculate dist
    for i in range(num_residents):
        model.add(l[i]==((model.sum(w[a]*model.sum(x[(i,j,a)]*d[(i,j)] for j in range(num_allocation)) for a in range(len(k))))))
    # PWL
    for i in range(num_residents):
    #     model.add(model.if_then(((l[i] >= 0) & (l[i] < 400)), f[i] == model.round(slope[0] * l[i] + intercept[0])))
    #     model.add(model.if_then(((l[i] >= 400) & (l[i] < 1800)), f[i] == model.round(slope[1] * l[i] + intercept[1])))
    #     model.add(model.if_then(((l[i] >= 1800) & (l[i] < 2400)), f[i] == model.round(slope[2] * l[i] + intercept[2])))
    #     model.add(model.if_then(l[i] >= 2400, f[i] == 0))
        model.add(model.slope_piecewise_linear(l[i], [400,1800,2400], [-0.0125, -0.0607, -0.0167,0], 0, 100)==f[i])


    # objective
    model.add(model.maximize(model.sum(f[i] for i in range(num_residents))/num_residents))

    # solving
    #model.set_search_phases([search_phase(y.values()), search_phase(x.values()),search_phase(l.values())])
    #model.set_search_phases([search_phase(y.values()), search_phase(x.values())])
    msol = model.solve(execfile=solver_path,TimeLimit=time_limit) #13h
    #msol = model.solve(execfile=solver_path, TimeLimit=20)
    obj_value = msol.get_objective_values()[0]

    # save solution for debugging and visualization

    assignments = [(i, j, a) for (i, j, a) in x.keys() if msol[x[(i, j, a)]] == 1]
    allocations = [(j, a) for (j, a) in y.keys() if msol[y[(j, a)]] == 1]

    # map back to df
    # allocated_nodes = [df_to.iloc[j]["node_ids"] for j in allocations]
    # allocated_df = df_to.iloc[allocations]
    # allocated_nodes=[j for (j,a) in allocations]

    allocated_nodes = [df_to.iloc[group_values_to[j][0]]["node_ids"] for (j,a) in allocations]
    allocated_df = df_to.iloc[[group_values_to[j][0] for (j,a) in allocations]]

    str = msol.solver_log

    i_s = []
    j_s = []
    d_s = []
    i_id = []
    j_id = []
    a_id = []

    for (i, j, a) in assignments:
        i_s.append(i)
        j_s.append(j)
        d_s.append(d[(i, j)])
        i_id.append(df_from.iloc[group_values_from[i][0]]["node_ids"])
        j_id.append(df_to.iloc[group_values_to[j][0]]["node_ids"])
        a_id.append(a)
    assigned_nodes = {
        "i": i_s,
        "j": j_s,
        "i_id": i_id,
        "j_id": j_id,
        "a_id": a_id,
        "d_s": d_s,
    }


    return obj_value, msol.get_solve_time(), allocated_nodes, allocated_df, assigned_nodes, model, str, num_residents, num_allocation


def CP_1b(df_from,df_to,SP_matrix,solver_path, EPS=1.e-6):
    if len(df_from)>0:
        df_from = df_from[['geometry', 'node_ids']]
    if len(df_to)>0:
        df_to = df_to[['geometry', 'node_ids']]

    model = CpoModel(name="max_score")

    groups_to=df_to.groupby('node_ids').groups
    #group_keys_to=list(groups_to.keys())#[2,5,...]
    group_values_to=list(groups_to.values())

    groups_from = df_from.groupby('node_ids').groups
    #group_keys_from = list(groups_from.keys())  # [2,5,...]
    group_values_from = list(groups_from.values())


    # data
    num_residents = len(group_values_from)
    num_allocation = len(group_values_to)
    capacity = [len(item) for item in group_values_to]
    # indices
    cartesian_prod = list(product(range(num_residents), range(num_allocation)))
    # retrieve distances
    d = {(i, j): SP_matrix[df_from.iloc[group_values_from[i][0]]["node_ids"], df_to.iloc[group_values_to[j][0]]["node_ids"]] for i, j in cartesian_prod}


    # variables
    y = {}
    for j in range(num_allocation):
        for a in range(len(k)):
            y[(j, a)] = model.binary_var(name=f'y[{j},{a}]')
    x = {}
    for i in range(num_residents):
        for j in range(num_allocation):
            for a in range(len(k)):
                x[(i, j, a)] = model.binary_var(name=f'x[{i},{j},{a}]')
    f = {}
    for i in range(num_residents):
        f[i] = model.float_var(name=f'f[{i}]')
        #f[i] = model.float_var(min=0, max=100, name=f'f[{i}]')
    l = {}
    for i in range(num_residents):
        l[i] = model.float_var(name=f'z[{i}]')
        #l[i] = model.float_var(min=0, max=L_a[-1], name=f'z[{i}]')

    # Constraints
    # activation
    for i in range(num_residents):
        for j in range(num_allocation):
            for a in range(len(k)):
                model.add(x[(i,j,a)]<=y[(j,a)])
    # # amenity upper bound
    for a in range(len(k)):
        model.add(model.sum([y[(j,a)] for j in range(num_allocation)]) <= k[a])
    # # # capacity upper bound
    for j in range(num_allocation):
        model.add(model.sum([y[(j,a)] for a in range(len(k))]) <= capacity[j])
    # each resident visit one of each amenity
    for i in range(num_residents):
        for a in range(len(k)):
            model.add(model.sum([x[(i,j,a)] for j in range(num_allocation)]) == 1)

    # side
    # side 1
    for j in range(num_allocation):
        model.add(model.if_then(model.logical_or((y[(j,0)]>=1),(y[(j,2)]>=1)), y[(j,3)]>=1))
    # side 2
        model.add(model.if_then(model.logical_and((y[(j, 0)] >= 1), (y[(j, 2)] >= 1)), y[(j, 1)] >= 1))

    # # calculate dist
    for i in range(num_residents):
        model.add(l[i]==((model.sum(w[a]*model.sum(x[(i,j,a)]*d[(i,j)] for j in range(num_allocation)) for a in range(len(k))))))
    # PWL
    for i in range(num_residents):
        model.add(model.slope_piecewise_linear(l[i], [400,1800,2400], [-0.0125, -0.0607, -0.0167,0], 0, 100)==f[i])


    # objective
    model.add(model.maximize(model.sum(f[i] for i in range(num_residents))/num_residents))

    # solving
    #model.set_search_phases([search_phase(y.values()), search_phase(x.values()),search_phase(l.values())])
    #model.set_search_phases([search_phase(y.values()), search_phase(x.values())])
    #msol = model.solve(execfile=solver_path,TimeLimit=46800) #13h
    msol = model.solve(execfile=solver_path, TimeLimit=time_limit)
    #msol = model.solve(execfile=solver_path, TimeLimit=20)
    obj_value = msol.get_objective_values()[0]

    # save solution for debugging and visualization

    assignments = [(i, j, a) for (i, j, a) in x.keys() if msol[x[(i, j, a)]] == 1]
    allocations = [(j, a) for (j, a) in y.keys() if msol[y[(j, a)]] == 1]

    # map back to df
    # allocated_nodes = [df_to.iloc[j]["node_ids"] for j in allocations]
    # allocated_df = df_to.iloc[allocations]
    # allocated_nodes=[j for (j,a) in allocations]

    allocated_nodes = [df_to.iloc[group_values_to[j][0]]["node_ids"] for (j,a) in allocations]
    allocated_df = df_to.iloc[[group_values_to[j][0] for (j,a) in allocations]]

    i_s = []
    j_s = []
    d_s = []
    i_id = []
    j_id = []
    a_id = []

    for (i, j, a) in assignments:
        i_s.append(i)
        j_s.append(j)
        d_s.append(d[(i, j)])
        i_id.append(df_from.iloc[group_values_from[i][0]]["node_ids"])
        j_id.append(df_to.iloc[group_values_to[j][0]]["node_ids"])
        a_id.append(a)
    assigned_nodes = {
        "i": i_s,
        "j": j_s,
        "i_id": i_id,
        "j_id": j_id,
        "a_id": a_id,
        "d_s": d_s,
    }
    str=msol.solver_log


    return obj_value, msol.get_solve_time(), allocated_nodes, allocated_df, assigned_nodes, model, str, num_residents, num_allocation


def CP_1b_no_x(df_from,df_to,SP_matrix,solver_path, EPS=1.e-6):
    if len(df_from)>0:
        df_from = df_from[['geometry', 'node_ids']]
    if len(df_to)>0:
        df_to = df_to[['geometry', 'node_ids']]

    model = CpoModel(name="max_score")

    groups_to=df_to.groupby('node_ids').groups
    #group_keys_to=list(groups_to.keys())#[2,5,...]
    group_values_to=list(groups_to.values())

    groups_from = df_from.groupby('node_ids').groups
    #group_keys_from = list(groups_from.keys())  # [2,5,...]
    group_values_from = list(groups_from.values())


    # data
    num_residents = len(group_values_from)
    num_allocation = len(group_values_to)
    capacity = [len(item) for item in group_values_to]
    # indices
    cartesian_prod = list(product(range(num_residents), range(num_allocation)))
    # retrieve distances
    d = {(i, j): SP_matrix[df_from.iloc[group_values_from[i][0]]["node_ids"], df_to.iloc[group_values_to[j][0]]["node_ids"]] for i, j in cartesian_prod}


    # variables
    y = {}
    for j in range(num_allocation):
        for a in range(len(k)):
            y[(j, a)] = model.binary_var(name=f'y[{j},{a}]')
    # x = {}
    # for i in range(num_residents):
    #     for j in range(num_allocation):
    #         for a in range(len(k)):
    #             x[(i, j, a)] = model.binary_var(name=f'x[{i},{j},{a}]')
    f = {}
    for i in range(num_residents):
        f[i] = model.float_var(min=0, max=100, name=f'f[{i}]')
    l = {}
    for i in range(num_residents):
        l[i] = model.float_var(min=0, max=L_a[-1], name=f'z[{i}]')

    dist = {}  # z
    for i in range(num_residents):
        for a in range(len(k)):
            for k_ in range(k[a]):
                dist[(i, a, k_)] = model.float_var(min=0, max=L_a[-1], name=f'dist[{i},{a},{k_}]')

    # Constraints
    # activation
    # for i in range(num_residents):
    #     for j in range(num_allocation):
    #         for a in range(len(k)):
    #             model.add(x[(i,j,a)]<=y[(j,a)])
    # distance element constrain
    for i in range(num_residents):
        for a in range(len(k)):
            for k_ in range(k[a]):
                model.add(dist[(i, a, k_)] == (model.element([d[(i, m)] for m in range(num_allocation)], y[(k_, a)])))

    # amenity upper bound
    for a in range(len(k)):
        model.add((model.sum(y[(j, a)] for j in range(num_allocation))) <= k[a])
        model.add((model.sum(y[(j, a)] for j in range(num_allocation))) > 0)
    # capacity upper bound
    for j in range(num_allocation):
        model.add((model.sum(y[(j,a)] for a in range(len(k)))) <= capacity[j])
    # each resident visit one of each amenity
    # for i in range(num_residents):
    #     for a in range(len(k)):
    #         model.add(model.sum(x[(i,j,a)] for j in range(num_allocation)) == 1)

    # side
    # side 1
    for j in range(num_allocation):
        model.add(model.if_then(model.logical_or((y[(j,0)]>=1),(y[(j,2)]>=1)), y[(j,3)]>=1))
    # side 2
        model.add(model.if_then(model.logical_and((y[(j, 0)] >= 1), (y[(j, 2)] >= 1)), y[(j, 1)] >= 1))

    # calculate dist
    for i in range(num_residents):

        #model.add(l[i]==((model.sum(w[a]*model.sum(x[(i,j,a)]*d[(i,j)] for j in range(num_allocation)) for a in range(len(k))))))
        # model.add(l[i] == (
        #  (model.sum(w[a] * model.min(model.max(y[(j, a)] * d[(i, j)],0) for j in range(num_allocation)) for a in range(len(k))))))
        model.add(l[i] == (
            (model.sum(w[a] * model.min(y[(j, a)] * d[(i, j)] for j in range(num_allocation) if (model.greater(y[(j, a)],0)==True) for a in
                       range(len(k)))))))

    # PWL
    for i in range(num_residents):
    #     model.add(model.if_then(((l[i] >= 0) & (l[i] < 400)), f[i] == model.round(slope[0] * l[i] + intercept[0])))
    #     model.add(model.if_then(((l[i] >= 400) & (l[i] < 1800)), f[i] == model.round(slope[1] * l[i] + intercept[1])))
    #     model.add(model.if_then(((l[i] >= 1800) & (l[i] < 2400)), f[i] == model.round(slope[2] * l[i] + intercept[2])))
    #     model.add(model.if_then(l[i] >= 2400, f[i] == 0))
        model.add(model.slope_piecewise_linear(l[i], [400,1800,2400], [-0.0125, -0.0607, -0.0167,0], 0, 100)==f[i])


    # objective
    model.add(model.maximize(model.sum(f[i] for i in range(num_residents))/num_residents))

    # solving
    #model.set_search_phases([search_phase(y.values()), search_phase(x.values()),search_phase(l.values())])
    #model.set_search_phases([search_phase(y.values()), search_phase(x.values())])
    msol = model.solve(execfile=solver_path,TimeLimit=time_limit) #13h
    #msol = model.solve(execfile=solver_path, TimeLimit=20)
    obj_value = msol.get_objective_values()[0]

    # save solution for debugging and visualization

    #assignments = [(i, j, a) for (i, j, a) in x.keys() if msol[x[(i, j, a)]] == 1]
    assignments = [(0,0,0)]
    allocations = [(j, a) for (j, a) in y.keys() if msol[y[(j, a)]] == 1]

    # map back to df
    # allocated_nodes = [df_to.iloc[j]["node_ids"] for j in allocations]
    # allocated_df = df_to.iloc[allocations]
    # allocated_nodes=[j for (j,a) in allocations]

    allocated_nodes = [df_to.iloc[group_values_to[j][0]]["node_ids"] for (j,a) in allocations]
    allocated_df = df_to.iloc[[group_values_to[j][0] for (j,a) in allocations]]

    str = msol.solver_log

    i_s = []
    j_s = []
    d_s = []
    i_id = []
    j_id = []
    a_id = []

    for (i, j, a) in assignments:
        i_s.append(i)
        j_s.append(j)
        d_s.append(d[(i, j)])
        i_id.append(df_from.iloc[group_values_from[i][0]]["node_ids"])
        j_id.append(df_to.iloc[group_values_to[j][0]]["node_ids"])
        a_id.append(a)
    assigned_nodes = {
        "i": i_s,
        "j": j_s,
        "i_id": i_id,
        "j_id": j_id,
        "a_id": a_id,
        "d_s": d_s,
    }


    return obj_value, msol.get_solve_time(), allocated_nodes, allocated_df, assigned_nodes, model, str, num_residents, num_allocation


def CP_2(df_from,df_to,SP_matrix,solver_path, EPS=1.e-6):
    # deal with the null option for y
    if len(df_from)>0:
        df_from = df_from[['geometry', 'node_ids']]
    if len(df_to)>0:
        df_to = df_to[['geometry', 'node_ids']]

    model = CpoModel(name="max_score")

    groups_to=df_to.groupby('node_ids').groups
    #group_keys_to=list(groups_to.keys())#[2,5,...]
    group_values_to=list(groups_to.values())

    groups_from = df_from.groupby('node_ids').groups
    #group_keys_from = list(groups_from.keys())  # [2,5,...]
    group_values_from = list(groups_from.values())


    # data
    num_residents = len(group_values_from)
    num_allocation = len(group_values_to)
    capacity = [len(item) for item in group_values_to]
    # indices
    cartesian_prod = list(product(range(num_residents), range(num_allocation)))
    # retrieve distances
    d = {(i, j): SP_matrix[df_from.iloc[group_values_from[i][0]]["node_ids"], df_to.iloc[group_values_to[j][0]]["node_ids"]] for i, j in cartesian_prod}


    # variables
    y = {}
    for a in range(len(k)):
        for k_ in range(k[a]):
            y[(k_, a)] = model.integer_var(min=0, max=num_allocation, name=f'y[{k_},{a}]') #include dummy node
    x = {}
    for i in range(num_residents):
        for a in range(len(k)):
            x[(i, a)] = model.integer_var(min=0, max=num_allocation-1, name=f'x[{i},{a}]')
    # b = {}
    # for i in range(num_residents):
    #     for a in range(len(k)):
    #         b[(i, a)] = model.binary_var(name=f'b[{i},{a}]')
    dist= {} # z
    for i in range(num_residents):
        for a in range(len(k)):
            dist[(i, a)] = model.float_var(min=0,max=L_a[-1],name=f'dist[{i},{a}]')
    f = {}
    for i in range(num_residents):
        f[i] = model.float_var(min=0, max=100, name=f'f[{i}]')
    l = {}
    for i in range(num_residents):
        l[i] = model.float_var(min=0, max=L_a[-1], name=f'z[{i}]')

    # Constraints
    # distance element constrain
    for i in range(num_residents):
        for a in range(len(k)):
            model.add(dist[(i,a)]==(model.element([d[(i,m)] for m in range(num_allocation)],x[(i, a)])))
    # capacity
    #model.add(model.pack(capacity,list(y.values()),[1]*len(list(y.values()))))
    for j in range(num_allocation):
        model.add(model.count(list(y.values()),j)<=capacity[j])

    # ensure activation
    # for a in range(len(k)):
    #     abstracted=[b[(i, a)] for i in range(num_residents)]
    #     ref=[x[(i, a)] for i in range(num_residents)]
    #     values=[y[(k_,a)] for k_ in range(k[a])]
    #     model.add(bool_abstraction(abstracted,ref,values))
    # model.add(model.all_of(list(b.values())))
    for j in range(num_allocation):
        for a in range(len(k)):
        # >=1 grocery
            cond = [x[(i, a)] == j for i in range(num_residents)]
            cond2 = [y[(k_, a)] == j for k_ in range(k[a])]
            model.add(model.if_then(model.any(cond),model.any(cond2)))

    # calculate dist
    for i in range(num_residents):
        model.add(l[i] == (model.sum(w[a] * dist[(i,a)] for a in range(len(k)))))
    # PWL
    for i in range(num_residents):
        model.add(model.slope_piecewise_linear(l[i], [400, 1800, 2400], [-0.0125, -0.0607, -0.0167, 0], 0, 100) == f[i])

    # symmetry breaking
    for a in range(len(k)):
        if k[a]>1:
            # old
            for k_ in range((k[a])-1):
                model.add(y[(k_,a)]<=y[(k_+1,a)])
                print(k_,a)



    # side
    # side 1

    for j in range(num_allocation):
        # >=1 grocery
        # cond1 = [y[(k_, 0)] == j for k_ in range(k[0])]
        # cond2= [y[(k_,2)]==j for k_ in range(k[2])]
        # cond3= [y[(k_,3)]==j for k_ in range(k[3])]
        # cond4 = [y[(k_, 1)] == j for k_ in range(k[1])]
        L1 = [y[(k_, 0)] for k_ in range(k[0])]
        L2 = [y[(k_, 2)] for k_ in range(k[2])]
        L3 = [y[(k_, 3)] for k_ in range(k[3])]
        L4 = [y[(k_, 1)] for k_ in range(k[1])]
        #model.add(model.if_then(model.any(cond1+cond2),model.any(cond3)))
        model.add(model.if_then(model.count(L1+L2,j)>=1, model.count(L3,j)>=1))
        # side 2
        #model.add(model.if_then(model.logical_and(model.any(cond1), model.any(cond2)), model.any(cond4)))
        model.add(
            model.if_then(model.logical_and(model.count(L1, j) >= 1, model.count(L2, j) >= 1), model.count(L4, j) >= 1))

    # objective
    model.add(model.maximize(model.sum(f[i] for i in range(num_residents))/num_residents))

    # solving
    #model.set_search_phases([search_phase(y.values()), search_phase(x.values()),search_phase(l.values())])
    #model.set_search_phases([search_phase(y.values()), search_phase(x.values())])
    #msol = model.solve(execfile=solver_path,TimeLimit=46800) #13h
    msol = model.solve(execfile=solver_path, TimeLimit=time_limit)
    #msol = model.solve(execfile=solver_path, TimeLimit=20)
    obj_value = msol.get_objective_values()[0]

    # save solution for debugging and visualization

    assignments = [(i, msol[x[(i,a)]], a) for (i, a) in x.keys()]
    allocations = [(msol[y[k_,a]], a) for (k_, a) in y.keys() if msol[y[k_,a]] < num_allocation] # exclude dummy node

    # map back to df
    # allocated_nodes = [df_to.iloc[j]["node_ids"] for j in allocations]
    # allocated_df = df_to.iloc[allocations]
    # allocated_nodes=[j for (j,a) in allocations]

    allocated_nodes = [df_to.iloc[group_values_to[j][0]]["node_ids"] for (j,a) in allocations]
    allocated_df = df_to.iloc[[group_values_to[j][0] for (j,a) in allocations]]

    i_s = []
    j_s = []
    d_s = []
    i_id = []
    j_id = []
    a_id = []

    for (i, j, a) in assignments:
        i_s.append(i)
        j_s.append(j)
        d_s.append(d[(i, j)])
        i_id.append(df_from.iloc[group_values_from[i][0]]["node_ids"])
        j_id.append(df_to.iloc[group_values_to[j][0]]["node_ids"])
        a_id.append(a)
    assigned_nodes = {
        "i": i_s,
        "j": j_s,
        "i_id": i_id,
        "j_id": j_id,
        "a_id": a_id,
        "d_s": d_s,
    }
    str=msol.solver_log


    return obj_value, msol.get_solve_time(), allocated_nodes, allocated_df, assigned_nodes, model, str, num_residents, num_allocation



def CP_2b(df_from,df_to,SP_matrix,solver_path, EPS=1.e-6):
    # deal with the null option for y
    if len(df_from)>0:
        df_from = df_from[['geometry', 'node_ids']]
    if len(df_to)>0:
        df_to = df_to[['geometry', 'node_ids']]

    model = CpoModel(name="max_score")

    groups_to=df_to.groupby('node_ids').groups
    #group_keys_to=list(groups_to.keys())#[2,5,...]
    group_values_to=list(groups_to.values())

    groups_from = df_from.groupby('node_ids').groups
    #group_keys_from = list(groups_from.keys())  # [2,5,...]
    group_values_from = list(groups_from.values())


    # data
    num_residents = len(group_values_from)
    num_allocation = len(group_values_to)
    capacity = [len(item) for item in group_values_to]
    # indices
    cartesian_prod = list(product(range(num_residents), range(num_allocation)))
    # retrieve distances
    d = {(i, j): SP_matrix[df_from.iloc[group_values_from[i][0]]["node_ids"], df_to.iloc[group_values_to[j][0]]["node_ids"]] for i, j in cartesian_prod}


    # variables
    y = {}
    for a in range(len(k)):
        for k_ in range(k[a]):
            y[(k_, a)] = model.integer_var(min=0, max=num_allocation, name=f'y[{k_},{a}]') #include dummy node
    x = {}
    for i in range(num_residents):
        for a in range(len(k)):
            x[(i, a)] = model.integer_var(min=0, max=num_allocation-1, name=f'x[{i},{a}]')
    # b = {}
    # for i in range(num_residents):
    #     for a in range(len(k)):
    #         b[(i, a)] = model.binary_var(name=f'b[{i},{a}]')
    dist= {} # z
    for i in range(num_residents):
        for a in range(len(k)):
            dist[(i, a)] = model.float_var(min=0,max=L_a[-1],name=f'dist[{i},{a}]')
    f = {}
    for i in range(num_residents):
        f[i] = model.float_var(min=0, max=100, name=f'f[{i}]')
    l = {}
    for i in range(num_residents):
        l[i] = model.float_var(min=0, max=L_a[-1], name=f'z[{i}]')

    # Constraints
    # distance element constrain
    for i in range(num_residents):
        for a in range(len(k)):
            model.add(dist[(i,a)]==(model.element([d[(i,m)] for m in range(num_allocation)],x[(i, a)])))
    # capacity
    #model.add(model.pack(capacity,list(y.values()),[1]*len(list(y.values()))))
    for j in range(num_allocation):
        model.add(model.count(list(y.values()),j)<=capacity[j])

    # ensure activation
    # for a in range(len(k)):
    #     abstracted=[b[(i, a)] for i in range(num_residents)]
    #     ref=[x[(i, a)] for i in range(num_residents)]
    #     values=[y[(k_,a)] for k_ in range(k[a])]
    #     model.add(bool_abstraction(abstracted,ref,values))
    # model.add(model.all_of(list(b.values())))
    for j in range(num_allocation):
        for a in range(len(k)):
        # >=1 grocery
            cond = [x[(i, a)] == j for i in range(num_residents)]
            cond2 = [y[(k_, a)] == j for k_ in range(k[a])]
            model.add(model.if_then(model.any(cond),model.any(cond2)))

    # calculate dist
    for i in range(num_residents):
        model.add(l[i] == (model.sum(w[a] * dist[(i,a)] for a in range(len(k)))))
    # PWL
    for i in range(num_residents):
        model.add(model.slope_piecewise_linear(l[i], [400, 1800, 2400], [-0.0125, -0.0607, -0.0167, 0], 0, 100) == f[i])

    # symmetry breaking
    for a in range(len(k)):
        if k[a]>1:
            # old
            for k_ in range((k[a])-1):
                model.add(y[(k_,a)]<=y[(k_+1,a)])
                print(k_,a)
            # lexicographic
            # model.add(model.lexicographic(list(range(k[a]), [ y[(k_,a)] for k_ in range(k[a]) ])))


    # side
    # side 1

    for j in range(num_allocation):
        # >=1 grocery
        # cond1 = [y[(k_, 0)] == j for k_ in range(k[0])]
        # cond2= [y[(k_,2)]==j for k_ in range(k[2])]
        # cond3= [y[(k_,3)]==j for k_ in range(k[3])]
        # cond4 = [y[(k_, 1)] == j for k_ in range(k[1])]
        L1 = [y[(k_, 0)] for k_ in range(k[0])]
        L2 = [y[(k_, 2)] for k_ in range(k[2])]
        L3 = [y[(k_, 3)] for k_ in range(k[3])]
        L4 = [y[(k_, 1)] for k_ in range(k[1])]
        #model.add(model.if_then(model.any(cond1+cond2),model.any(cond3)))
        model.add(model.if_then(model.count(L1+L2,j)>=1, model.count(L3,j)>=1))
        # side 2
        #model.add(model.if_then(model.logical_and(model.any(cond1), model.any(cond2)), model.any(cond4)))
        model.add(
            model.if_then(model.logical_and(model.count(L1, j) >= 1, model.count(L2, j) >= 1), model.count(L4, j) >= 1))

    # symmetric
    for a in range(len(k)):
        model.add(model.all_diff([y[(k_, a)] for k_ in range(k[a])]))

    # objective
    model.add(model.maximize(model.sum(f[i] for i in range(num_residents))/num_residents))

    # solving
    #model.set_search_phases([search_phase(y.values()), search_phase(x.values()),search_phase(l.values())])
    #model.set_search_phases([search_phase(y.values()), search_phase(x.values())])
    #msol = model.solve(execfile=solver_path,TimeLimit=46800) #13h
    msol = model.solve(execfile=solver_path, TimeLimit=time_limit)
    #msol = model.solve(execfile=solver_path, TimeLimit=20)
    obj_value = msol.get_objective_values()[0]

    # save solution for debugging and visualization

    assignments = [(i, msol[x[(i,a)]], a) for (i, a) in x.keys()]
    allocations = [(msol[y[k_,a]], a) for (k_, a) in y.keys() if msol[y[k_,a]] < num_allocation] # exclude dummy node

    # map back to df
    # allocated_nodes = [df_to.iloc[j]["node_ids"] for j in allocations]
    # allocated_df = df_to.iloc[allocations]
    # allocated_nodes=[j for (j,a) in allocations]

    allocated_nodes = [df_to.iloc[group_values_to[j][0]]["node_ids"] for (j,a) in allocations]
    allocated_df = df_to.iloc[[group_values_to[j][0] for (j,a) in allocations]]

    i_s = []
    j_s = []
    d_s = []
    i_id = []
    j_id = []
    a_id = []

    for (i, j, a) in assignments:
        i_s.append(i)
        j_s.append(j)
        d_s.append(d[(i, j)])
        i_id.append(df_from.iloc[group_values_from[i][0]]["node_ids"])
        j_id.append(df_to.iloc[group_values_to[j][0]]["node_ids"])
        a_id.append(a)
    assigned_nodes = {
        "i": i_s,
        "j": j_s,
        "i_id": i_id,
        "j_id": j_id,
        "a_id": a_id,
        "d_s": d_s,
    }
    str=msol.solver_log


    return obj_value, msol.get_solve_time(), allocated_nodes, allocated_df, assigned_nodes, model, str, num_residents, num_allocation



def CP_2b_no_x(df_from,df_to,SP_matrix,solver_path, EPS=1.e-6):
    # deal with the null option for y
    if len(df_from)>0:
        df_from = df_from[['geometry', 'node_ids']]
    if len(df_to)>0:
        df_to = df_to[['geometry', 'node_ids']]

    model = CpoModel(name="max_score")

    groups_to=df_to.groupby('node_ids').groups
    #group_keys_to=list(groups_to.keys())#[2,5,...]
    group_values_to=list(groups_to.values())

    groups_from = df_from.groupby('node_ids').groups
    #group_keys_from = list(groups_from.keys())  # [2,5,...]
    group_values_from = list(groups_from.values())


    # data
    num_residents = len(group_values_from)
    num_allocation = len(group_values_to)
    capacity = [len(item) for item in group_values_to]
    # indices
    cartesian_prod = list(product(range(num_residents), range(num_allocation)))
    # retrieve distances
    d = {(i, j): SP_matrix[df_from.iloc[group_values_from[i][0]]["node_ids"], df_to.iloc[group_values_to[j][0]]["node_ids"]] for i, j in cartesian_prod}
    for i in range(num_residents):
        d[(i,num_allocation)]=L_a[-1]


    # variables
    y = {}
    for a in range(len(k)):
        for k_ in range(k[a]):
            y[(k_, a)] = model.integer_var(min=0, max=num_allocation, name=f'y[{k_},{a}]') #include dummy node
    # x = {}
    # for i in range(num_residents):
    #     for a in range(len(k)):
    #         x[(i, a)] = model.integer_var(min=0, max=num_allocation-1, name=f'x[{i},{a}]')
    dist= {} # z
    for i in range(num_residents):
        for a in range(len(k)):
            for k_ in range(k[a]):
                dist[(i, a, k_)] = model.float_var(min=0,max=L_a[-1],name=f'dist[{i},{a},{k_}]')

    # dist_min = {}  # z
    # for i in range(num_residents):
    #     for a in range(len(k)):
    #         dist_min[(i, a)] = model.float_var(min=0, max=L_a[-1], name=f'dist_min[{i},{a}]')
    f = {}
    for i in range(num_residents):
        f[i] = model.float_var(min=0, max=100, name=f'f[{i}]')
    l = {}
    for i in range(num_residents):
        l[i] = model.float_var(min=0, max=L_a[-1], name=f'z[{i}]')

    # Constraints
    # distance element constrain
    for i in range(num_residents):
        for a in range(len(k)):
            for k_ in range(k[a]):
                model.add(dist[(i,a,k_)]==(model.element([d[(i,m)] for m in range(num_allocation+1)],y[(k_, a)])))

    for j in range(num_allocation):
        model.add(model.count(list(y.values()),j)<=capacity[j])


    # for j in range(num_allocation):
    #     for a in range(len(k)):
    #     # >=1 grocery
    #         cond = [x[(i, a)] == j for i in range(num_residents)]
    #         cond2 = [y[(k_, a)] == j for k_ in range(k[a])]
    #         model.add(model.if_then(model.any(cond),model.any(cond2)))

    # calculate dist
    # for i in range(num_residents):
    #     for a in range(len(k)):
    #         model.add(dist_min[(i,a)] == model.min(dist[(i,a,k_)] for k_ in range(k[a])))
    #         #model.add(dist_min[(i, a)] == model.min([200,100]))
    for i in range(num_residents):
        model.add(l[i] == (model.sum(w[a] * model.min(dist[(i,a,k_)] for k_ in range(k[a])) for a in range(len(k)))))
    # PWL
    for i in range(num_residents):
        model.add(f[i]==model.slope_piecewise_linear(l[i], [400, 1800, 2400], [-0.0125, -0.0607, -0.0167, 0], 0, 100))

    # symmetry breaking
    for a in range(len(k)):
        if k[a]>1:
            # old
            for k_ in range((k[a])-1):
                model.add(y[(k_,a)]<=y[(k_+1,a)])
                print(k_,a)
            # lexicographic
            # model.add(model.lexicographic(list(range(k[a]), [ y[(k_,a)] for k_ in range(k[a]) ])))


    # side
    # side 1

    for j in range(num_allocation):
        # >=1 grocery
        # cond1 = [y[(k_, 0)] == j for k_ in range(k[0])]
        # cond2= [y[(k_,2)]==j for k_ in range(k[2])]
        # cond3= [y[(k_,3)]==j for k_ in range(k[3])]
        # cond4 = [y[(k_, 1)] == j for k_ in range(k[1])]
        L1 = [y[(k_, 0)] for k_ in range(k[0])]
        L2 = [y[(k_, 2)] for k_ in range(k[2])]
        L3 = [y[(k_, 3)] for k_ in range(k[3])]
        L4 = [y[(k_, 1)] for k_ in range(k[1])]
        #model.add(model.if_then(model.any(cond1+cond2),model.any(cond3)))
        model.add(model.if_then(model.count(L1+L2,j)>=1, model.count(L3,j)>=1))
        # side 2
        #model.add(model.if_then(model.logical_and(model.any(cond1), model.any(cond2)), model.any(cond4)))
        model.add(
            model.if_then(model.logical_and(model.count(L1, j) >= 1, model.count(L2, j) >= 1), model.count(L4, j) >= 1))

    # symmetric
    for a in range(len(k)):
        model.add(model.all_diff([y[(k_, a)] for k_ in range(k[a])]))

    # # objective
    model.add(model.maximize(model.sum(f[i] for i in range(num_residents))/num_residents))

    # solving
    #model.set_search_phases([search_phase(y.values()), search_phase(x.values()),search_phase(l.values())])
    #model.set_search_phases([search_phase(y.values()), search_phase(x.values())])
    #msol = model.solve(execfile=solver_path,TimeLimit=46800) #13h
    msol = model.solve(execfile=solver_path, TimeLimit=time_limit)
    #msol = model.solve(execfile=solver_path, TimeLimit=20)
    obj_value = msol.get_objective_values()[0]

    # save solution for debugging and visualization

    #assignments = [(i, msol[x[(i,a)]], a) for (i, a) in x.keys()]
    assignments = [(0, 0, 0) ]
    allocations = [(msol[y[k_,a]], a) for (k_, a) in y.keys() if msol[y[k_,a]] < num_allocation] # exclude dummy node


    # map back to df
    # allocated_nodes = [df_to.iloc[j]["node_ids"] for j in allocations]
    # allocated_df = df_to.iloc[allocations]
    # allocated_nodes=[j for (j,a) in allocations]

    allocated_nodes = [df_to.iloc[group_values_to[j][0]]["node_ids"] for (j,a) in allocations]
    allocated_df = df_to.iloc[[group_values_to[j][0] for (j,a) in allocations]]

    i_s = []
    j_s = []
    d_s = []
    i_id = []
    j_id = []
    a_id = []

    for (i, j, a) in assignments:
        i_s.append(i)
        j_s.append(j)
        d_s.append(d[(i, j)])
        i_id.append(df_from.iloc[group_values_from[i][0]]["node_ids"])
        j_id.append(df_to.iloc[group_values_to[j][0]]["node_ids"])
        a_id.append(a)
    assigned_nodes = {
        "i": i_s,
        "j": j_s,
        "i_id": i_id,
        "j_id": j_id,
        "a_id": a_id,
        "d_s": d_s,
    }
    str=msol.solver_log


    return obj_value, msol.get_solve_time(), allocated_nodes, allocated_df, assigned_nodes, model, str, num_residents, num_allocation

def SAT(df_from,df_to,SP_matrix,solver_path, EPS=1.e-6):

    if len(df_from)>0:
        df_from = df_from[['geometry', 'node_ids']]
    if len(df_to)>0:
        df_to = df_to[['geometry', 'node_ids']]

    # Model
    model = cp_model.CpModel()

    groups_to=df_to.groupby('node_ids').groups
    #group_keys_to=list(groups_to.keys())#[2,5,...]
    group_values_to=list(groups_to.values())

    groups_from = df_from.groupby('node_ids').groups
    #group_keys_from = list(groups_from.keys())  # [2,5,...]
    group_values_from = list(groups_from.values())


    # data
    num_residents = len(group_values_from)
    num_allocation = len(group_values_to)
    capacity = [len(item) for item in group_values_to]
    # indices
    cartesian_prod = list(product(range(num_residents), range(num_allocation)))
    # retrieve distances
    d = {(i, j): SP_matrix[df_from.iloc[group_values_from[i][0]]["node_ids"], df_to.iloc[group_values_to[j][0]]["node_ids"]] for i, j in cartesian_prod}


    # variables
    y = {}
    for j in range(num_allocation):
        for a in range(len(k)):
            y[(j, a)] = model.NewBoolVar(f'y[{j},{a}]')
    x = {}
    for i in range(num_residents):
        for j in range(num_allocation):
            for a in range(len(k)):
                x[(i, j, a)] = model.NewBoolVar(name=f'x[{i},{j},{a}]')
    delta={}
    for i in range(num_residents):
        for p in range(3):
            delta[(i, p)] = model.NewBoolVar(name=f'x[{i},{p}]')

    # Constraints
    for i in range(num_residents):
        dist=sum(((int(10*w[a]) * sum(x[(i, j, a)] * int(d[(i, j)]) for j in range(num_allocation)) for a in range(len(k)))))
        # model.AddImplication(dist <= 10*int(L_a[3]),delta[(i, 2)])
        # model.AddImplication(dist > 10 * int(L_a[3]), delta[(i, 2)].Not())
        # model.AddImplication(dist <= 10 * int(L_a[2]), delta[(i, 1)])
        # model.AddImplication(dist > 10 * int(L_a[2]), delta[(i, 1)].Not())
        # model.AddImplication(dist <= 10 * int(L_a[1]), delta[(i, 0)])
        # model.AddImplication(dist > 10 * int(L_a[1]), delta[(i, 0)].Not())
        model.Add(dist <= 10*int(L_a[3])).OnlyEnforceIf(delta[(i, 2)])
        model.Add(dist > 10*int(L_a[3])).OnlyEnforceIf(delta[(i, 2)].Not())
        model.Add(dist <= 10*int(L_a[2])).OnlyEnforceIf(delta[(i, 1)])
        model.Add(dist > 10*int(L_a[2])).OnlyEnforceIf(delta[(i, 1)].Not())
        model.Add(dist <= 10*int(L_a[1])).OnlyEnforceIf(delta[(i, 0)])
        model.Add(dist > 10*int(L_a[1])).OnlyEnforceIf(delta[(i, 0)].Not())

    # activation
    for i in range(num_residents):
        for j in range(num_allocation):
            for a in range(len(k)):
                #model.Add(y[(j,a)]).OnlyEnforceIf(x[(i,j,a)])
                model.AddImplication(x[(i,j,a)], y[(j,a)])
    # amenity upper bound
    for a in range(len(k)):
        model.Add(sum(y[(j,a)] for j in range(num_allocation)) <= k[a])
    # capacity upper bound
    for j in range(num_allocation):
        model.Add(sum(y[(j, a)] for a in range(len(k))) <= capacity[j])
    # each resident visit one of each amenity
    for i in range(num_residents):
        for a in range(len(k)):
            model.Add(sum(x[(i,j,a)] for j in range(num_allocation)) == 1)

    # Interval relationship
    for i in range(num_residents):
        model.AddImplication(delta[(i, 1)], delta[(i, 2)])
        model.AddImplication(delta[(i, 0)], delta[(i , 1)])

    # side
    # side 1
    for j in range(num_allocation):
        model.AddImplication(y[(j, 0)],y[(j, 3)])
        model.AddImplication(y[(j, 2)], y[(j, 3)])
    # side 2
        #tmp2 = model.AddBoolAnd([y[(j, 0)], y[(j, 2)]])
        #model.AddImplication(tmp2,y[(j, 1)])
        model.AddBoolOr([y[(j, 0)].Not(), y[(j, 2)].Not(), y[(j, 1)]])

    # Objective
    objective_terms = []
    for i in range(num_residents):
        for p in range(3):
            objective_terms.append(delta[(i,p)])
    model.Maximize(sum(objective_terms))

    # Solve

    solver = cp_model.CpSolver()
    solver.parameters.enumerate_all_solutions = True
    status = solver.Solve(model)

    # Print solution.
    assignments=[]
    allocations=[]
    if status == cp_model.OPTIMAL:
        print(f'Total cost = {solver.ObjectiveValue()}\n')
        for i in range(num_residents):
            for j in range(num_allocation):
                for a in range(len(k)):
                    if solver.BooleanValue(x[(i, j, a)]):
                        assignments.append((i,j,a))
        for j in range(num_allocation):
            for a in range(len(k)):
                if solver.BooleanValue(y[(j, a)]):
                    allocations.append((j, a))


    # solving
    #model.set_search_phases([search_phase(y.values()), search_phase(x.values()),search_phase(l.values())])
    #model.set_search_phases([search_phase(y.values()), search_phase(x.values())])
    #msol = model.solve(execfile=solver_path,TimeLimit=46800) #13h
    #msol = model.solve(execfile=solver_path, TimeLimit=time_limit)
    #msol = model.solve(execfile=solver_path, TimeLimit=20)
    obj = solver.ObjectiveValue()

    # save solution for debugging and visualization

    # assignments = [(i, j, a) for (i, j, a) in x.keys() if msol[x[(i, j, a)]] == 1]
    # allocations = [(j, a) for (j, a) in y.keys() if msol[y[(j, a)]] == 1]

    # map back to df
    # allocated_nodes = [df_to.iloc[j]["node_ids"] for j in allocations]
    # allocated_df = df_to.iloc[allocations]
    # allocated_nodes=[j for (j,a) in allocations]

    allocated_nodes = [df_to.iloc[group_values_to[j][0]]["node_ids"] for (j,a) in allocations]
    allocated_df = df_to.iloc[[group_values_to[j][0] for (j,a) in allocations]]

    i_s = []
    j_s = []
    d_s = []
    i_id = []
    j_id = []
    a_id = []

    my_L={}
    my_score=[]
    for (i, j, a) in assignments:
        i_s.append(i)
        j_s.append(j)
        d_s.append(d[(i, j)])
        i_id.append(df_from.iloc[group_values_from[i][0]]["node_ids"])
        j_id.append(df_to.iloc[group_values_to[j][0]]["node_ids"])
        a_id.append(a)
        if i in my_L.keys():
            my_L[i].append(d[(i, j)]*w[a])
        else:
            my_L[i]=[d[(i, j)]*w[a]]
    assigned_nodes = {
        "i": i_s,
        "j": j_s,
        "i_id": i_id,
        "j_id": j_id,
        "a_id": a_id,
        "d_s": d_s,
    }
    for i in range(len(my_L)):
        my_score.append(np.sum(my_L[i]))
    obj_value=np.mean(dist_to_score(np.array(my_score),L_a,L_f_a))
    #str=msol.solver_log

    str = 'SAT'
    time=solver.WallTime()


    return obj_value, time, allocated_nodes, allocated_df, assigned_nodes, model, str, num_residents, num_allocation

def MaxSAT(df_from,df_to,SP_matrix,solver_path, EPS=1.e-6):

    if len(df_from)>0:
        df_from = df_from[['geometry', 'node_ids']]
    if len(df_to)>0:
        df_to = df_to[['geometry', 'node_ids']]

    # Model
    #model = cp_model.CpModel()

    groups_to=df_to.groupby('node_ids').groups
    #group_keys_to=list(groups_to.keys())#[2,5,...]
    group_values_to=list(groups_to.values())

    groups_from = df_from.groupby('node_ids').groups
    #group_keys_from = list(groups_from.keys())  # [2,5,...]
    group_values_from = list(groups_from.values())


    # data
    num_residents = len(group_values_from)
    num_allocation = len(group_values_to)
    capacity = [len(item) for item in group_values_to]
    # indices
    cartesian_prod = list(product(range(num_residents), range(num_allocation)))
    # retrieve distances
    d = {(i, j): SP_matrix[df_from.iloc[group_values_from[i][0]]["node_ids"], df_to.iloc[group_values_to[j][0]]["node_ids"]] for i, j in cartesian_prod}

    wcnf = WCNF()
    # add clauses examples
    # wcnf.append([-1, -2])  # adding hard clauses
    # wcnf.append([-1, -3])
    # wcnf.append([1], weight=1)  # adding soft clauses
    # wcnf.append([2], weight=1)
    # wcnf.append([3], weight=1)

    # solve
    # with RC2(wcnf) as rc2:
    #     rc2.compute()

    # to get solution
    #rc2.model
    #interval=200
    #new_L=[i * interval for i in range(1,(int(max(d.values()))//interval))]
    #L_a = [0, 200, 400, 800, 1000, 1200, ]
    #p_list=list(range(len(new_L)))
    #my_L=[0,400,1800]
    #p_list=[1]

    new_L=L_a

    p_list=[1,2,3]
    #p_list = [1]
    counter = 1
    map_y_ja = {}
    map_iap = {}
    map_y_ja_inv = {}
    map_iap_inv = {}
    for j in range(num_allocation):
        for a in range(len(k)):
            map_y_ja[(j,a)] = counter
            map_y_ja_inv[counter] = (j, a)
            counter+=1
    for i in range(num_residents):
        for a in range(len(k)):
            for p in p_list:
                map_iap[(i,a,p)] = counter
                map_iap_inv[counter] = (i,a,p)
                counter+=1


    for i in range(num_residents):
        for a in range(len(k)):
            for p in p_list:
                pos=[]
                for j in range(num_allocation):
                    #if d[(i, j)] <= L_a[p]:
                    if d[(i, j)] <= new_L[p]:
                        pos.append(j)
                        wcnf.append([map_iap[(i,a,p)], -map_y_ja[(j,a)]])
                if len(pos)>0:
                    c = [map_y_ja[(location,a)] for location in pos] + [-map_iap[(i,a,p)]]
                    wcnf.append(c)
                else:
                    wcnf.append([-map_iap[(i,a,p)]])

    # soft
    for i in range(num_residents):
        for a in range(len(k)):
            for p in p_list:
                wcnf.append([map_iap[(i,a,p)]], weight=(w[a]*10)) # weight by amenity
                #wcnf.append([map_iap[(i, a, p)]], weight=1)  # weight by amenity

    max_id=counter
    # for a in range(len(k)):
    #     cnf = CardEnc.atleast(lits=[map_y_ja[(j,a)] for j in range(num_allocation)], bound=1,encoding=EncType.totalizer, top_id=max_id)
    #     if len(cnf.clauses)>0:
    #         #wcnf.append(cnf.clauses)
    #         broken = [x for sublist in cnf.clauses for x in sublist]
    #         top = max([abs(s) for s in broken])
    #         if top>max_id:
    #             max_id=top
    #         for c in cnf.clauses:
    #             wcnf.append(c)



    for a in range(len(k)):
        cnf = CardEnc.atmost(lits=[map_y_ja[(j,a)] for j in range(num_allocation)], bound=k[a],encoding=EncType.seqcounter, top_id=max_id)
        if len(cnf.clauses)>0:
            for c in cnf.clauses:
                wcnf.append(c)

        broken=[x for sublist in cnf.clauses for x in sublist]
        top = max([abs(s) for s in broken])
        if top>max_id:
            max_id=top

    for j in range(num_allocation):
        cnf = CardEnc.atmost(lits=[map_y_ja[(j, a)] for a in range(len(k))], bound=capacity[j],encoding=EncType.seqcounter,top_id=max_id)
        if len(cnf.clauses) > 0:
            #wcnf.append(cnf.clauses)
            for c in cnf.clauses:
                wcnf.append(c)

        broken = [x for sublist in cnf.clauses for x in sublist]
        top = max([abs(s) for s in broken])
        if top>max_id:
            max_id=top


    # Interval relationship
    # for i in range(num_residents):
    #     model.AddImplication(delta[(i, 1)], delta[(i, 2)])
    #     model.AddImplication(delta[(i, 0)], delta[(i , 1)])
    # for i in range(num_residents):
    #     for a in range(len(k)):
    #         wcnf.append([-map_iap[(i,a,1)],map_iap[(i,a,2)]])
    #         wcnf.append([-map_iap[(i, a, 2)], map_iap[(i, a, 3)]])


    # # # side 1
    for j in range(num_allocation):
        wcnf.append([-map_y_ja[(j, 0)], map_y_ja[(j, 3)]])
        wcnf.append([-map_y_ja[(j, 2)], map_y_ja[(j, 3)]])
        wcnf.append([-map_y_ja[(j, 0)], -map_y_ja[(j, 2)], map_y_ja[(j, 1)]])

# 1    for j in range(num_allocation):
#         model.AddImplication(y[(j, 0)],y[(j, 3)])
#         model.AddImplication(y[(j, 2)], y[(j, 3)])
#     # side 2
#         model.AddBoolOr([y[(j, 0)].Not(), y[(j, 2)].Not(), y[(j, 1)]])



    # Solve
    #m=RC2(wcnf)
    #timer = Timer(10, interrupt, [m])
    #timer.start()
    #print(m.solve_limited(expect_interrupt=True))
    #m.delete()

    with RC2(wcnf,verbose=1) as rc2:
        start = time.time()
        rc2.compute()
        end = time.time()

    #model_sol = m.model
    #to get solution
    model_sol=rc2.model
    print([item for item in model_sol if item > 0])

    # Print solution.
    assignments=[]
    allocations=[]

    for i in range(num_residents):
        for a in range(len(k)):
            for p in p_list:
                if map_iap[(i, a, p)] in model_sol:
                    assignments.append((i,a,p))
    for j in range(num_allocation):
        for a in range(len(k)):
            if map_y_ja[(j,a)] in model_sol:
                allocations.append((j, a))


    #msol = model.solve(execfile=solver_path,TimeLimit=46800) #13h
    #msol = model.solve(execfile=solver_path, TimeLimit=time_limit)
    #msol = model.solve(execfile=solver_path, TimeLimit=20)
    #obj = solver.ObjectiveValue()

    # save solution for debugging and visualization

    # assignments = [(i, j, a) for (i, j, a) in x.keys() if msol[x[(i, j, a)]] == 1]
    # allocations = [(j, a) for (j, a) in y.keys() if msol[y[(j, a)]] == 1]

    # map back to df
    # allocated_nodes = [df_to.iloc[j]["node_ids"] for j in allocations]
    # allocated_df = df_to.iloc[allocations]
    # allocated_nodes=[j for (j,a) in allocations]

    print(allocations)

    allocated_nodes = [df_to.iloc[group_values_to[j][0]]["node_ids"] for (j,a) in allocations]
    allocated_df = df_to.iloc[[group_values_to[j][0] for (j,a) in allocations]]

    allocated_map= {}
    for a in range(len(k)):
        allocated_map[a]=[]
    for (j, a) in allocations:
        allocated_map[a].append(j)


    i_s = []
    j_s = []
    d_s = []
    i_id = []
    j_id = []
    a_id = []

    my_L={}
    my_score=[]

    for i in range(num_residents):
        for a in range(len(k)):
            j_list=allocated_map[a]
            j=j_list[np.argmin(d[(i, j_)] for j_ in j_list)]
            i_s.append(i)
            j_s.append(j)
            d_s.append(d[(i, j)])
            i_id.append(df_from.iloc[group_values_from[i][0]]["node_ids"])
            j_id.append(df_to.iloc[group_values_to[j][0]]["node_ids"])
            a_id.append(a)
            if i in my_L.keys():
                my_L[i].append(d[(i, j)]*w[a])
            else:
                my_L[i]=[d[(i, j)]*w[a]]
    assigned_nodes = {
        "i": i_s,
        "j": j_s,
        "i_id": i_id,
        "j_id": j_id,
        "a_id": a_id,
        "d_s": d_s,
    }
    for i in range(len(my_L)):
        my_score.append(np.sum(my_L[i]))
    obj_value=np.mean(dist_to_score(np.array(my_score),[0,400,1800,2400,5000000],L_f_a))
    #str=msol.solver_log

    str = 'SAT'
    #time=solver.WallTime()

    model = None
    return obj_value, end-start, allocated_nodes, allocated_df, assigned_nodes, model, str, num_residents, num_allocation


def interrupt(s):
    s.interrupt()

def dist_to_score(d,L_a,L_f_a):
    a = copy.deepcopy(L_a[:-1])
    f_a = copy.deepcopy(L_f_a[:-1])
    L_m = []
    L_c = []
    for i in range(len(a) - 1):
        x = np.array(a[i:i + 2])
        y = np.array(f_a[i:i + 2])
        A = np.vstack([x, np.ones(len(x))]).T
        m, c = np.linalg.lstsq(A, y, rcond=None)[0]
        L_m.append(m)
        L_c.append(c)
    scores = np.piecewise(d, [d<a[1], (a[1]<=d) & (d<a[2]), (a[2]<=d) & (d<a[3]), d>=a[3]],
                 [lambda d: L_m[0]*d+L_c[0], lambda d: L_m[1]*d+L_c[1], lambda d: L_m[2]*d+L_c[2], lambda d:0])
    return scores


# not used
def old_cp_checkpoint(df_from,df_to,supermarkets_df, SP_matrix,k,solver_path, EPS=1.e-6):
    if len(df_from)>0:
        df_from = df_from[['geometry', 'node_ids']]
    if len(df_to)>0:
        df_to = df_to[['geometry', 'node_ids']]
    if len(supermarkets_df)>0:
        supermarkets_df = supermarkets_df[['geometry', 'node_ids']]
        df_to_2 = pd.concat([df_to, supermarkets_df])
    else:
        df_to_2 = df_to

    model = CpoModel(name="max_score")

    df_to__=df_to.groupby('node_ids')
    count_df=df_to.groupby('node_ids').count()
    groups=df_to.groupby('node_ids').groups
    group_names=list(df_to.groupby('node_ids').groups.keys())#[2,5,...]
    #groups[group_id][1]

    # data
    num_residents = len(df_from)
    num_allocation = len(df_to)
    num_cur = len(supermarkets_df)
    cartesian_prod_assign = list(product(range(num_residents), range(num_allocation + num_cur)))  # a list of tuples
    cartesian_prod_activate = list(product(range(num_residents), range(num_allocation)))

    # retrieve distances
    d = {(i, j): SP_matrix[df_from.iloc[i]["node_ids"], df_to_2.iloc[j]["node_ids"]] for i, j in cartesian_prod_assign}

    # variables

    y = {}
    for m in range(num_allocation):
        y[m] = model.binary_var(name=f'y[{m}]')
    x = {}
    for n in range(num_residents):
        for m in range(num_allocation + num_cur):
            x[(n, m)] = model.binary_var(name=f'x[{n},{m}]')
    f = {}
    for n in range(num_residents):
        f[n] = model.integer_var(min=0, max=100, name=f'f[{n}]')
    z = {}
    for n in range(num_residents):
        z[n] = model.integer_var(min=0, max=L_a[-1], name=f'z[{n}]')

    # Constraints
    # each resident visit one amenity
    for n in range(num_residents):
        model.add(model.sum(x[(n,m)] for m in range(num_allocation+num_cur)) == 1)
    # amenity upper bound
    model.add(model.sum(y[m] for m in range(num_allocation)) <= k)
    # activate allocation
    for n in range(num_residents):
        for m in range(num_allocation):
            model.add(model.if_then(x[(n,m)]==1, y[m]==1)) #if_then(e1, e2) e1 => e2

    # calculate dist
    for n in range(num_residents):
        model.add(z[n]==model.round(model.sum((x[(n,m)]*d[(n,m)]) for m in range(num_allocation+num_cur))))
    # PWL
    for n in range(num_residents):
        model.add(model.if_then(((z[n] >= 0) & (z[n] < 400)), f[n] == model.round(slope[0] * z[n] + intercept[0])))
        model.add(model.if_then(((z[n] >= 400) & (z[n] < 1800)), f[n] == model.round(slope[1] * z[n] + intercept[1])))
        model.add(model.if_then(((z[n] >= 1800) & (z[n] < 2400)), f[n] == model.round(slope[2] * z[n] + intercept[2])))
        model.add(model.if_then(z[n] >= 2400, f[n] == 0))


    #for n in range(num_residents):
    #    model.add(f[n]==-0.006*6+100)

    # objective
    model.add(model.maximize(model.sum(f[n] for n in range(num_residents))/num_residents))

    model.set_search_phases([search_phase(y.values()), search_phase(x.values()),search_phase(z.values())])

    msol = model.solve(execfile=solver_path)

    assignments = [(i, j) for (i, j) in x.keys() if msol[x[(i,j)]] == 1]
    allocations = [j for j in y.keys() if msol[y[j]] == 1]

    # map back to df
    # allocated_nodes = [df_to.iloc[j]["node_ids"] for j in allocations]
    # allocated_df = df_to.iloc[allocations]
    allocated_nodes, allocated_df = map_back_allocate(allocations, df_to)
    assigned_nodes = map_back_assign(assignments, df_from, df_to_2, d)

    #obj = m.getObjective()
    #obj_value = obj.getValue()
    obj_value=msol.get_objective_values()[0]


    return obj_value, msol.get_solve_time(), allocated_nodes, allocated_df, assigned_nodes, model

# not used
def old_cp2_pwl(df_from,df_to,SP_matrix,solver_path, EPS=1.e-6):
    if len(df_from)>0:
        df_from = df_from[['geometry', 'node_ids']]
    if len(df_to)>0:
        df_to = df_to[['geometry', 'node_ids']]

    model = CpoModel(name="max_score")

    groups_to=df_to.groupby('node_ids').groups
    #group_keys_to=list(groups_to.keys())#[2,5,...]
    group_values_to=list(groups_to.values())

    groups_from = df_from.groupby('node_ids').groups
    #group_keys_from = list(groups_from.keys())  # [2,5,...]
    group_values_from = list(groups_from.values())


    # data
    num_residents = len(group_values_from)
    num_allocation = len(group_values_to)
    capacity = [len(item) for item in group_values_to]
    # indices
    cartesian_prod = list(product(range(num_residents), range(num_allocation)))
    # retrieve distances
    d = {(i, j): SP_matrix[df_from.iloc[group_values_from[i][0]]["node_ids"], df_to.iloc[group_values_to[j][0]]["node_ids"]] for i, j in cartesian_prod}



    # variables
    y = {}
    for j in range(num_allocation):
        for a in range(len(k)):
            y[(j, a)] = model.integer_var(min=0, max=k[a],name=f'y[{j},{a}]')
    x = {}
    for i in range(num_residents):
        for j in range(num_allocation):
            for a in range(len(k)):
                x[(i, j, a)] = model.binary_var(name=f'x[{i},{j},{a}]')
    f = {}
    for i in range(num_residents):
        f[i] = model.integer_var(min=0, max=100, name=f'f[{i}]')
    l = {}
    for i in range(num_residents):
        l[i] = model.integer_var(min=0, max=L_a[-1], name=f'z[{i}]')

    # Constraints
    # activation
    for i in range(num_residents):
        for j in range(num_allocation):
            for a in range(len(k)):
                model.add(x[(i,j,a)]<=y[(j,a)])
    # amenity upper bound
    for a in range(len(k)):
        model.add((model.sum(y[(j, a)] for j in range(num_allocation))) <= k[a])
    # capacity upper bound
    for j in range(num_allocation):
        model.add((model.sum(y[(j,a)] for a in range(len(k)))) <= capacity[j])
    # each resident visit one of each amenity
    for i in range(num_residents):
        for a in range(len(k)):
            model.add(model.sum(x[(i,j,a)] for j in range(num_allocation)) == 1)

    # side
    # side 1
    for j in range(num_allocation):
        model.add(model.if_then(model.logical_or((y[(j,0)]>=1),(y[(j,2)]>=1)), y[(j,3)]>=1))
    # side 2
        model.add(model.if_then(model.logical_and((y[(j, 0)] >= 1), (y[(j, 2)] >= 1)), y[(j, 1)] >= 1))

    # calculate dist
    for i in range(num_residents):
        model.add(l[i]==(model.round(w[a]*model.sum(model.sum(x[(i,j,a)]*d[(i,j)] for j in range(num_allocation)) for a in range(len(k))))))
    # PWL
    for i in range(num_residents):
        model.add(model.if_then(((l[i] >= 0) & (l[i] < 400)), f[i] == model.round(slope[0] * l[i] + intercept[0])))
        model.add(model.if_then(((l[i] >= 400) & (l[i] < 1800)), f[i] == model.round(slope[1] * l[i] + intercept[1])))
        model.add(model.if_then(((l[i] >= 1800) & (l[i] < 2400)), f[i] == model.round(slope[2] * l[i] + intercept[2])))
        model.add(model.if_then(l[i] >= 2400, f[i] == 0))


    # objective
    model.add(model.maximize(model.sum(f[i] for i in range(num_residents))/num_residents))

    # solving
    model.set_search_phases([search_phase(y.values()), search_phase(x.values()),search_phase(l.values())])
    msol = model.solve(execfile=solver_path,TimeLimit=time_limit) #13h
    #msol = model.solve(execfile=solver_path, TimeLimit=20)
    obj_value = msol.get_objective_values()[0]

    # save solution for debugging and visualization

    assignments = [(i, j, a) for (i, j, a) in x.keys() if msol[x[(i, j, a)]] == 1]
    allocations = [(j, a) for (j, a) in y.keys() if msol[y[(j, a)]] == 1]

    # map back to df
    # allocated_nodes = [df_to.iloc[j]["node_ids"] for j in allocations]
    # allocated_df = df_to.iloc[allocations]
    # allocated_nodes=[j for (j,a) in allocations]

    allocated_nodes = [df_to.iloc[group_values_to[j][0]]["node_ids"] for (j,a) in allocations]
    allocated_df = df_to.iloc[[group_values_to[j][0] for (j,a) in allocations]]

    i_s = []
    j_s = []
    d_s = []
    i_id = []
    j_id = []
    a_id = []

    for (i, j, a) in assignments:
        i_s.append(i)
        j_s.append(j)
        d_s.append(d[(i, j)])
        i_id.append(df_from.iloc[group_values_from[j][0]]["node_ids"])
        j_id.append(df_to.iloc[group_values_to[j][0]]["node_ids"])
        a_id.append(a)
    assigned_nodes = {
        "i": i_s,
        "j": j_s,
        "i_id": i_id,
        "j_id": j_id,
        "a_id": a_id,
        "d_s": d_s,
    }


    return obj_value, msol.get_solve_time(), allocated_nodes, allocated_df, assigned_nodes, model
