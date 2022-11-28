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
#import time
L_a=[0,400,1800,2400,5000000]
L_f_a=[100,95,10,0,0]
slope=[-0.0125, -0.0607, -0.0167] #[-0.0125, -0.06071428571428574, -0.016666666666666653]
intercept=[100.0000, 119.2857, 40.0000] #[100.00000000000001, 119.28571428571433, 39.999999999999964]

def max_score_cp1(df_from,df_to,supermarkets_df, SP_matrix,k,EPS=1.e-6):
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

    #model.set_search_phases([search_phase(y.values()), search_phase(x.values()),search_phase(z.values())])

    #msol = model.solve(execfile="/Applications/CPLEX_Studio201/cpoptimizer/bin/x86-64_osx/cpoptimizer")
    msol = model.solve(execfile='/home/huangw98/modulefiles/mycplex/cpoptimizer/bin/x86-64_linux/cpoptimizer',TimeLimit=18000)

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
