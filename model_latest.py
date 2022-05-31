import gurobipy as gp
from gurobipy import GRB
from itertools import product
import os
import pandas as pd
import numpy as np
import copy
from map_utils import map_back_allocate, map_back_assign
import matplotlib.pyplot as plt

L_a=[0,400,1800,2400,5000000]
L_f_a=[100,95,10,0,0]

def opt_single(df_from,df_to,amenity_df, SP_matrix,k,threads,results_sava_path,EPS=0.5):
    '''single amenity case, no depth of choice'''

    if len(df_from)>0:
        df_from = df_from[['geometry', 'node_ids']]
    if len(df_to)>0:
        df_to = df_to[['geometry', 'node_ids']]

    m = gp.Model('max_walk_score')

    # grouping
    groups_to=df_to.groupby('node_ids').groups # keys are node id, values are indices
    group_values_to=list(groups_to.values())
    num_allocation = len(group_values_to)
    capacity = [len(item) for item in group_values_to]

    groups_from = df_from.groupby('node_ids').groups
    group_values_from = list(groups_from.values())
    num_residents = len(group_values_from)

    num_cur = len(amenity_df)

    cartesian_prod_assign = list(product(range(num_residents), range(num_allocation + num_cur)))  # a list of tuples
    cartesian_prod_allocate = list(product(range(num_residents), range(num_allocation)))

    # retrieve distances
    d = {(i, j): SP_matrix[df_from.iloc[group_values_from[i][0]]["node_ids"], df_to.iloc[group_values_to[j][0]]["node_ids"]] for i, j in cartesian_prod_allocate}

    for i in range(num_residents):
        for l in range(num_cur):
            cur_id = num_allocation+l
            d[(i, cur_id)] = SP_matrix[df_from.iloc[group_values_from[i][0]]["node_ids"], amenity_df.iloc[l]["node_ids"]]

    # Variables
    x = m.addVars(cartesian_prod_assign, vtype=GRB.BINARY, name='assign')
    y = m.addVars(num_allocation, vtype=GRB.INTEGER, name='activate')
    a = m.addVars(num_residents, vtype=GRB.CONTINUOUS, name='dist')
    f = m.addVars(num_residents, vtype=GRB.CONTINUOUS, ub=100,name='score')

    # branching priority
    # for m in range(num_allocation):
    #     y[m].setAttr("BranchPriority", 5)
    # for (n,m) in cartesian_prod_assign:
    #     x[(n,m)].setAttr("BranchPriority",4)

    # Constraints
    ## WalkScore
    m.addConstrs((a[n] ==
                 (gp.quicksum(d[(n, m)] * x[(n, m)] for m in range(num_allocation + num_cur)))) for n in
                 range(num_residents))
    for n in range(num_residents):
        m.addGenConstrPWL(a[n], f[n], L_a, L_f_a)
    ## assgined nodes satisfy demand
    m.addConstrs(
        (gp.quicksum(x[(n, m)] for m in range(num_allocation + num_cur)) == 1 for n in range(num_residents)),
        name='Demand')
    ## resource constraint
    m.addConstr(gp.quicksum(y[m] for m in range(num_allocation)) <= k, name='resource')
    ## activation
    m.addConstrs((x[(n, m)] <= y[m] for n, m in cartesian_prod_allocate), name='setup')

    m.addConstrs(y[j] <= capacity[j] for j in range(num_allocation))

    # objective
    m.Params.Threads = threads
    m.setObjective(gp.quicksum(f[n] for n in range(num_residents))/num_residents, GRB.MAXIMIZE)
    m.setParam("LogFile", results_sava_path)

    m.optimize()

    assignments = [(i, j) for (i, j) in x.keys() if (x[i, j].x > EPS)]
    allocations = [j for j in y.keys() if (y[j].x > EPS)]

    # save allocation solutions
    allocate_var_id = allocations
    allocate_row_id = []
    allocate_node_id = []
    for j in allocate_var_id:
        for l in range(round(y[j].x)):
            allocate_row_id.append(group_values_to[j][l])
            allocate_node_id.append(df_to.iloc[group_values_to[j][l]]["node_ids"])
    allocated_D = {
        "allocate_var_id": allocate_var_id,
        "allocate_node_id": allocate_node_id,
        "allocate_row_id": allocate_row_id
    }

    assign_from_var_id = [i for (i, j) in assignments]
    assign_to_var_id = [j for (i, j) in assignments]
    assign_from_node_id = [df_from.iloc[group_values_from[i][0]]["node_ids"] for (i, j) in assignments]
    assign_to_node_id = []
    assign_type = []
    dist=[]
    for (i,j) in assignments:
        if j < num_allocation:
            assign_to_node_id.append(df_to.iloc[group_values_to[j][0]]["node_ids"])
            assign_type.append('allocated')
            dist.append(SP_matrix[df_from.iloc[group_values_from[i][0]]["node_ids"], df_to.iloc[group_values_to[j][0]]["node_ids"]])
        else:
            assign_to_node_id.append(amenity_df.iloc[j-num_allocation]["node_ids"])
            assign_type.append('existing')
            dist.append(SP_matrix[df_from.iloc[group_values_from[i][0]]["node_ids"],amenity_df.iloc[j-num_allocation]["node_ids"]])

    assigned_D = {
        "assign_from_var_id": assign_from_var_id,
        "assign_to_var_id": assign_to_var_id,
        "assign_from_node_id": assign_from_node_id,
        "assign_to_node_id": assign_to_node_id,
        "assign_type": assign_type,
        "dist": dist}

    obj = m.getObjective()
    obj_value = obj.getValue()
    dist_obj = np.mean(dist)

    return obj_value, dist_obj, m.Runtime, m, allocated_D, assigned_D, num_residents, num_allocation, num_cur


def opt_multiple(df_from,df_to,grocery_df, restaurant_df, school_df, SP_matrix,k_array,threads,results_sava_path,EPS=0.5):
    '''single amenity case, no depth of choice'''

    if len(df_from)>0:
        df_from = df_from[['geometry', 'node_ids']]
    if len(df_to)>0:
        df_to = df_to[['geometry', 'node_ids']]

    m = gp.Model('max_walk_score')

    # grouping
    groups_to=df_to.groupby('node_ids').groups # keys are node id, values are indices
    group_values_to=list(groups_to.values())
    num_allocation = len(group_values_to)
    capacity = [len(item) for item in group_values_to]

    groups_from = df_from.groupby('node_ids').groups
    group_values_from = list(groups_from.values())
    num_residents = len(group_values_from)

    num_cur_grocery = len(grocery_df)
    num_cur_restaurant = len(restaurant_df)
    num_cur_school = len(school_df)

    cur_index=num_allocation
    range_grocery = range(cur_index, cur_index+num_cur_grocery)
    cur_index+=num_cur_grocery
    range_restaurant = range(cur_index, cur_index+num_cur_restaurant)
    cur_index+=num_cur_restaurant
    range_school = range(cur_index,cur_index+num_cur_school)

    cartesian_prod_assign_grocery = list(product(range(num_residents), list(range(num_allocation)) + list(range_grocery), [0]))
    cartesian_prod_assign_restaurant = list(product(range(num_residents), list(range(num_allocation)) + list(range_restaurant), [1]))
    cartesian_prod_assign_school = list(product(range(num_residents), list(range(num_allocation)) + list(range_school), [2]))

    # retrieve distances
    d = {(i, j): SP_matrix[df_from.iloc[group_values_from[i][0]]["node_ids"], df_to.iloc[group_values_to[j][0]]["node_ids"]] for i, j in list(product(range(num_residents), range(num_allocation)))}

    for i in range(num_residents):
        start_id = num_allocation
        for amenity_df in [grocery_df, restaurant_df, school_df]:
            for l in range(len(amenity_df)):
                cur_id = start_id + l
                d[(i, cur_id)] = SP_matrix[df_from.iloc[group_values_from[i][0]]["node_ids"], amenity_df.iloc[l]["node_ids"]]
            start_id += len(amenity_df)

    x = m.addVars(cartesian_prod_assign_grocery + cartesian_prod_assign_restaurant + cartesian_prod_assign_school, vtype=GRB.BINARY, name='assign')
    y = m.addVars(list(product(range(num_allocation), range(len(k_array)))), vtype=GRB.INTEGER, name='activate')
    a = m.addVars(num_residents, vtype=GRB.CONTINUOUS, name='dist')
    f = m.addVars(num_residents, vtype=GRB.CONTINUOUS, ub=100, name='score')

    # branching priority
    # for m in range(num_allocation):
    #     y[m].setAttr("BranchPriority", 5)
    # for (n,m) in cartesian_prod_assign:
    #     x[(n,m)].setAttr("BranchPriority",4)

    # Constraints
    ## WalkScore
    m.addConstrs((a[n] ==
                 (gp.quicksum(d[(n, m)] * x[(n, m)] for m in range(num_allocation + num_cur)))) for n in
                 range(num_residents))
    for n in range(num_residents):
        m.addGenConstrPWL(a[n], f[n], L_a, L_f_a)
    ## assgined nodes satisfy demand
    m.addConstrs(
        (gp.quicksum(x[(n, m)] for m in range(num_allocation + num_cur)) == 1 for n in range(num_residents)),
        name='Demand')
    ## resource constraint
    m.addConstr(gp.quicksum(y[m] for m in range(num_allocation)) <= k, name='resource')
    ## activation
    m.addConstrs((x[(n, m)] <= y[m] for n, m in cartesian_prod_allocate), name='setup')

    m.addConstrs(y[j] <= capacity[j] for j in range(num_allocation))

    # objective
    m.Params.Threads = threads
    m.setObjective(gp.quicksum(f[n] for n in range(num_residents))/num_residents, GRB.MAXIMIZE)
    m.setParam("LogFile", results_sava_path)

    m.optimize()

    assignments = [(i, j) for (i, j) in x.keys() if (x[i, j].x > EPS)]
    allocations = [j for j in y.keys() if (y[j].x > EPS)]

    # save allocation solutions
    allocate_var_id = allocations
    allocate_row_id = []
    allocate_node_id = []
    for j in allocate_var_id:
        for l in range(int(y[j].x)):
            allocate_row_id.append(group_values_to[j][l])
            allocate_node_id.append(df_to.iloc[group_values_to[j][l]]["node_ids"])
    allocated_D = {
        "allocate_var_id": allocate_var_id,
        "allocate_node_id": allocate_node_id,
        "allocate_row_id": allocate_row_id
    }

    assign_from_var_id = [i for (i, j) in assignments]
    assign_to_var_id = [j for (i, j) in assignments]
    assign_from_node_id = [df_from.iloc[group_values_from[i][0]]["node_ids"] for (i, j) in assignments]
    assign_to_node_id = []
    assign_type = []
    dist=[]
    for (i,j) in assignments:
        if j < num_allocation:
            assign_to_node_id.append(df_to.iloc[group_values_to[j][0]]["node_ids"])
            assign_type.append('allocated')
            dist.append(SP_matrix[df_from.iloc[group_values_from[i][0]]["node_ids"], df_to.iloc[group_values_to[j][0]]["node_ids"]])
        else:
            assign_to_node_id.append(amenity_df.iloc[j-num_allocation]["node_ids"])
            assign_type.append('existing')
            dist.append(SP_matrix[df_from.iloc[group_values_from[i][0]]["node_ids"],amenity_df.iloc[j-num_allocation]["node_ids"]])

    assigned_D = {
        "assign_from_var_id": assign_from_var_id,
        "assign_to_var_id": assign_to_var_id,
        "assign_from_node_id": assign_from_node_id,
        "assign_to_node_id": assign_to_node_id,
        "assign_type": assign_type,
        "dist": dist}

    obj = m.getObjective()
    obj_value = obj.getValue()

    return obj_value, m.Runtime, m, allocated_D, assigned_D, num_residents, num_allocation, num_cur


def cur_assignments(df_from,amenity_df, SP_matrix,EPS=1.e-6):
    ''' get assignment for the case with no allocation, no depth of choice'''
    if len(amenity_df) == 0:
        print("no existing amenities!")
        return 0, None, None, None, None, None, 0

    m = gp.Model('cur_assignment')
    groups_from = df_from.groupby('node_ids').groups
    group_values_from = list(groups_from.values())
    num_residents = len(group_values_from)
    num_amenity = len(amenity_df)
    cartesian_prod = list(product(range(num_residents), range(num_amenity)))  # a list of tuples
    distances = {(i, j): SP_matrix[df_from.iloc[group_values_from[i][0]]["node_ids"], amenity_df.iloc[j]["node_ids"]] for i, j in cartesian_prod}
    assign = m.addVars(cartesian_prod, vtype=GRB.BINARY, name='Assign')

    # demand
    m.addConstrs((gp.quicksum(assign[(i, j)] for j in range(num_amenity)) == 1 for i in range(num_residents)),
                 name='Demand')
    # objective
    m.setObjective(assign.prod(distances)/num_residents, GRB.MINIMIZE)
    m.optimize()

    obj = m.getObjective()
    obj_value = obj.getValue() # min total dist

    assignments = [(i, j) for (i, j) in assign.keys() if (assign[i, j].x > EPS)]


    assign_from_var_id = [i for (i, j) in assignments]
    assign_to_var_id = [j for (i, j) in assignments]
    assign_from_node_id = [df_from.iloc[group_values_from[i][0]]["node_ids"] for (i, j) in assignments]
    assign_to_node_id = [amenity_df.iloc[j]["node_ids"] for (i, j) in assignments]

    dist=[SP_matrix[df_from.iloc[group_values_from[i][0]]["node_ids"], amenity_df.iloc[j]["node_ids"]] for (i, j) in assignments]
    scores = dist_to_score(np.array(dist), L_a, L_f_a)
    score_obj = np.mean(scores)

    assigned_D = {
        "assign_from_var_id": assign_from_var_id,
        "assign_to_var_id": assign_to_var_id,
        "assign_from_node_id": assign_from_node_id,
        "assign_to_node_id": assign_to_node_id,
        "dist": dist}

    return score_obj, obj_value, m.Runtime, m, assigned_D, num_residents, num_amenity

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