import gurobipy as gp
from gurobipy import GRB
from itertools import product
import os
import pandas as pd
import numpy as np
from docplex.cp.model import *
import copy
from map_utils import map_back_allocate, map_back_assign
import matplotlib.pyplot as plt

amenity_weights_dict = { "grocery": [3],
"restaurants": [.75, .45, .25, .25, .225, .225, .225, .225, .2, .2],
"shopping": [.5, .45, .4, .35, .3],
"coffee": [1.25, .75],
"banks": [1],
"parks": [1], "schools": [1], "books": [1], "entertainment": [1]}

choice_weights_raw = np.array([.75, .45, .25, .25, .225, .225, .225, .225, .2, .2])  # for restaurant
restaurant_sum = np.sum(choice_weights_raw)
choice_weights = choice_weights_raw / restaurant_sum  # for restaurant

L_a=[0,400,1800,2400,5000000]
L_f_a=[100,95,10,0,0]
weights_array = np.array([3,restaurant_sum,1]) / (restaurant_sum+3+1) # grocery, restaurant, school (temp)
weights_array_multi = np.array([3, .75, .45, .25, .25, .225, .225, .225, .225, .2, .2, 1]) / (restaurant_sum+3+1)
w_choice_multi_amenity = choice_weights_raw / (restaurant_sum+3+1)
time_limit=5*60*60 # 5h time limit

def opt_single(df_from,df_to,amenity_df, SP_matrix,k,threads,results_sava_path,bp, focus,EPS=0.5):
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
    y = m.addVars(num_allocation, vtype=GRB.BINARY, name='activate')
    a = m.addVars(num_residents, vtype=GRB.CONTINUOUS, name='dist')
    f = m.addVars(num_residents, vtype=GRB.CONTINUOUS, ub=100,name='score')

    if bp:
        print("branch priority set")
        m.update()
        # branching priority
        # if BranchPriority:
        for j in range(num_allocation):
            y[j].setAttr("BranchPriority", 100)
        m.update()
        # for (n,m) in cartesian_prod_assign:
        #     x[(n,m)].setAttr("BranchPriority",0)

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
    m.Params.TimeLimit = time_limit
    m.Params.MIPFocus = focus
    m.Params.NodefileStart = 0.5

    m.optimize()

    assignments = [(i, j) for (i, j) in x.keys() if (x[i, j].x > EPS)]
    allocations = [j for j in y.keys() if (y[j].x > EPS)]

    # save allocation solutions
    allocate_var_id = allocations
    allocate_row_id = []
    allocate_node_id = []
    for j in allocate_var_id:
        for l in range(int(np.round(y[j].x))):
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

    return obj_value, dist_obj, m.Runtime, m, allocated_D, assigned_D, num_residents, num_allocation, num_cur, m.status

def opt_single_depth(df_from,df_to,amenity_df, SP_matrix,k,threads,results_sava_path,bp, focus,EPS=0.5):
    '''single amenity case, with consideration of depth of choice. For amenity=restaurant specifically'''

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

    tot_choices = min(k + num_cur, len(choice_weights))
    no_choices = list(range(tot_choices, len(choice_weights)))

    cartesian_prod_assign = list(product(range(num_residents), range(num_allocation + num_cur), range(tot_choices)))  # last index is for depth of choice
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

    if bp:
        print("branch priority set")
        m.update()
        # branching priority
        # if BranchPriority:
        for j in range(num_allocation):
            y[j].setAttr("BranchPriority", 100)
        m.update()
        # for (n,m) in cartesian_prod_assign:
        #     x[(n,m)].setAttr("BranchPriority",0)

    # Constraints
    ## WalkScore
    no_choice_sum =sum([choice_weights[c]*L_a[-2] for c in no_choices])
    m.addConstrs((
                    a[n] ==
                    (gp.quicksum(choice_weights[c]*(gp.quicksum(d[(n, m)] * x[(n, m, c)] for m in range(num_allocation + num_cur))) for c in range(tot_choices)) + no_choice_sum)
                  )
                 for n in range(num_residents))

    for n in range(num_residents):
        m.addGenConstrPWL(a[n], f[n], L_a, L_f_a)
    ## assign choices
    m.addConstrs((
        (gp.quicksum(x[(n, m, c)] for m in range(num_allocation + num_cur)) == 1) for c in range(tot_choices) for n in range(num_residents)),
        name='choices')

    ## resource constraint
    m.addConstr(gp.quicksum(y[m] for m in range(num_allocation)) <= k, name='resource')
    ## activation
    m.addConstrs((x[(n, m, c)] <= y[m] for (n, m, c) in list(product(range(num_residents), range(num_allocation), range(tot_choices)))), name='setup')
    ## node capacity
    m.addConstrs(y[j] <= capacity[j] for j in range(num_allocation))
    # choices can not be the same place
    ## newly allocated
    m.addConstrs(((gp.quicksum(x[(n, m, c)] for c in range(tot_choices)) <= y[m]) for m in range(num_allocation) for n in range(num_residents)), name='choices')
    ## currently existing
    m.addConstrs(((gp.quicksum(x[(n, m, c)] for c in range(tot_choices)) <= 1) for m in range(num_allocation,num_allocation+num_cur) for n in range(num_residents)), name='choices')
    # # symmetry
    # ## choice id=2 and 3
    # if tot_choices >= 4:
    #     m.addConstrs(gp.quicksum(d[(n, m)] * x[(n, m, 2)] for m in range(num_allocation + num_cur)) <= gp.quicksum(d[(n, m)] * x[(n, m, 3)] for m in range(num_allocation + num_cur))
    #                  for n in range(num_residents))
    # ## choice id = 8 and 9
    # if tot_choices >= 10:
    #     m.addConstrs(gp.quicksum(d[(n, m)] * x[(n, m, 8)] for m in range(num_allocation + num_cur)) <= gp.quicksum(
    #         d[(n, m)] * x[(n, m, 9)] for m in range(num_allocation + num_cur))
    #                  for n in range(num_residents))
    # ## choice id=4,5,6,7
    # for id in [4,5,6]:
    #     if tot_choices >= (id+2):
    #         m.addConstrs(gp.quicksum(d[(n, m)] * x[(n, m, id)] for m in range(num_allocation + num_cur)) <= gp.quicksum(
    #             d[(n, m)] * x[(n, m, id+1)] for m in range(num_allocation + num_cur))
    #                      for n in range(num_residents))


    # objective
    m.Params.Threads = threads
    m.setObjective(gp.quicksum(f[n] for n in range(num_residents))/num_residents, GRB.MAXIMIZE)
    m.setParam("LogFile", results_sava_path)
    m.Params.TimeLimit = time_limit
    m.Params.MIPFocus = focus
    m.Params.NodefileStart = 0.5

    m.optimize()

    allocations = [j for j in y.keys() if (y[j].x > EPS)]

    # save allocation solutions
    allocate_var_id = []
    allocate_var_id_ = allocations
    allocate_row_id = []
    allocate_node_id = []
    for j in allocate_var_id_:
        for l in range(int(np.round(y[j].x))):
            allocate_var_id.append(j)
            allocate_row_id.append(group_values_to[j][l])
            allocate_node_id.append(df_to.iloc[group_values_to[j][l]]["node_ids"])
    allocated_D = {
        "allocate_var_id": allocate_var_id,
        "allocate_node_id": allocate_node_id,
        "allocate_row_id": allocate_row_id
    }

    assigned_D={}

    for choice in range(tot_choices):

        all = [(i, j, c) for (i, j, c) in x.keys() if (x[(i, j, c)].x > EPS)]
        assignments = [(i,j,c) for (i,j,c) in all if (c==choice)]

        assign_from_var_id = [i for (i, j, c) in assignments]
        assign_to_var_id = [j for (i, j, c) in assignments]
        assign_from_node_id = [df_from.iloc[group_values_from[i][0]]["node_ids"] for (i, j, c) in assignments]
        assign_to_node_id = []
        assign_type = []
        dist=[]
        for (i, j, c) in assignments:
            if j < num_allocation:
                assign_to_node_id.append(df_to.iloc[group_values_to[j][0]]["node_ids"])
                assign_type.append('allocated')
                dist.append(SP_matrix[df_from.iloc[group_values_from[i][0]]["node_ids"], df_to.iloc[group_values_to[j][0]]["node_ids"]])
            else:
                assign_to_node_id.append(amenity_df.iloc[j-num_allocation]["node_ids"])
                assign_type.append('existing')
                dist.append(SP_matrix[df_from.iloc[group_values_from[i][0]]["node_ids"],amenity_df.iloc[j-num_allocation]["node_ids"]])

        assigned_D[str(choice)+"_assign_from_var_id"]=assign_from_var_id
        assigned_D[str(choice)+"_assign_to_var_id"]=assign_to_var_id
        assigned_D[str(choice)+"_assign_from_node_id"]=assign_from_node_id
        assigned_D[str(choice)+"_assign_to_node_id"]=assign_to_node_id
        assigned_D[str(choice)+"_assign_type"]=assign_type
        assigned_D[str(choice)+"_dist"]=dist

    obj = m.getObjective()
    obj_value = obj.getValue()
    dist_obj = [np.mean(assigned_D[str(c)+"_dist"]) if (str(c)+"_dist") in assigned_D.keys() else 0 for c in range(len(choice_weights))]

    return obj_value, dist_obj, m.Runtime, m, allocated_D, assigned_D, num_residents, num_allocation, num_cur, m.status


def opt_multiple(df_from,df_to,grocery_df, restaurant_df, school_df, SP_matrix, k_array, threads,results_sava_path,bp, focus,EPS=0.5):
    '''multiple amenity case, no depth of choice'''

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
    range_grocery_dest_list = list(range(num_allocation)) + list(range(cur_index, cur_index + num_cur_grocery))
    cur_index+=num_cur_grocery
    range_restaurant_dest_list = list(range(num_allocation)) + list(range(cur_index, cur_index + num_cur_restaurant))
    cur_index+=num_cur_restaurant
    range_school_dest_list = list(range(num_allocation)) + list(range(cur_index, cur_index + num_cur_school))

    cartesian_prod_assign_grocery = list(product(range(num_residents), range_grocery_dest_list, [0]))
    cartesian_prod_assign_restaurant = list(product(range(num_residents),range_restaurant_dest_list, [1]))
    cartesian_prod_assign_school = list(product(range(num_residents), range_school_dest_list, [2]))

    cartesian_prod_allocate = list(product(range(num_residents), list(range(num_allocation)), [0,1,2]))

    # retrieve distances
    d = {(i, j): SP_matrix[df_from.iloc[group_values_from[i][0]]["node_ids"], df_to.iloc[group_values_to[j][0]]["node_ids"]] for i, j in list(product(range(num_residents), range(num_allocation)))}

    for i in range(num_residents):
        start_id = num_allocation
        for amenity_df in [grocery_df, restaurant_df, school_df]:
            for inst_row in range(len(amenity_df)):
                cur_id = start_id + inst_row
                d[(i, cur_id)] = SP_matrix[df_from.iloc[group_values_from[i][0]]["node_ids"], amenity_df.iloc[inst_row]["node_ids"]]
            start_id += len(amenity_df)

    x = m.addVars(cartesian_prod_assign_grocery + cartesian_prod_assign_restaurant + cartesian_prod_assign_school, vtype=GRB.BINARY, name='assign')
    y = m.addVars(list(product(range(num_allocation), range(len(k_array)))), vtype=GRB.BINARY, name='activate')
    l = m.addVars(num_residents, vtype=GRB.CONTINUOUS, name='dist')
    f = m.addVars(num_residents, vtype=GRB.CONTINUOUS, ub=100, name='score')

    # branching priority
    if bp:
        for t in list(product(range(num_allocation), range(len(k_array)))):
            y[t].setAttr("BranchPriority", 100)
        # for (n,m) in cartesian_prod_assign:
        #     x[(n,m)].setAttr("BranchPriority",4)

    # Constraints
    ## weighted distance
    m.addConstrs(l[i] == (
                 (weights_array[0] * gp.quicksum(x[(i, j, 0)] * d[(i, j)] for j in range_grocery_dest_list))
                + (weights_array[1] * gp.quicksum(x[(i, j, 1)] * d[(i, j)] for j in range_restaurant_dest_list))
                + (weights_array[2] * gp.quicksum(x[(i, j, 2)] * d[(i, j)] for j in range_school_dest_list))
                )
                 for i in range(num_residents))
    # PWL score
    for i in range(num_residents):
        m.addGenConstrPWL(l[i], f[i], L_a, L_f_a)
    ## assgined to one instance of amenity
    m.addConstrs((gp.quicksum(x[(i, j, 0)] for j in range_grocery_dest_list) == 1 for i in range(num_residents)), name='grocery demand')
    m.addConstrs((gp.quicksum(x[(i, j, 1)] for j in range_restaurant_dest_list) == 1 for i in range(num_residents)), name='restaurant demand')
    m.addConstrs((gp.quicksum(x[(i, j, 2)] for j in range_school_dest_list) == 1 for i in range(num_residents)), name='school demand')
    ## resource constraint
    m.addConstrs(((gp.quicksum(y[(j,a)] for a in range(len(weights_array))) <= capacity[j]) for j in range(num_allocation)), name='capacity')
    # activation
    m.addConstrs((x[(i,j,a)] <= y[(j,a)] for (i, j ,a) in cartesian_prod_allocate), name='activation')
    # resource constraint
    m.addConstrs(((gp.quicksum(y[(j, a)] for j in range(num_allocation)) <= k_array[a]) for a in range(len(k_array))), name='resource')

    # objective
    m.Params.Threads = threads
    m.setObjective(gp.quicksum(f[n] for n in range(num_residents))/num_residents, GRB.MAXIMIZE)
    m.setParam("LogFile", results_sava_path)
    m.Params.TimeLimit = time_limit
    m.Params.MIPFocus = focus
    m.Params.NodefileStart = 0.5

    m.optimize()

    allocations = [(j, a) for (j, a) in y.keys() if (y[(j, a)].x) > EPS]

    # save allocation solutions
    allocate_var_id_grocery = [(j, a) for (j, a) in allocations if a==0]
    allocate_row_id_grocery = []
    allocate_node_id_grocery = []
    for (j, a) in allocate_var_id_grocery:
        for l in range(int(y[(j, a)].x)):
            allocate_row_id_grocery.append(group_values_to[j][l])
            allocate_node_id_grocery.append(df_to.iloc[group_values_to[j][l]]["node_ids"])

    allocate_var_id_restaurant = [(j, a) for (j, a) in allocations if a==1]
    allocate_row_id_restaurant = []
    allocate_node_id_restaurant = []
    for (j, a) in allocate_var_id_restaurant:
        for l in range(int(y[(j, a)].x)):
            allocate_row_id_restaurant.append(group_values_to[j][l])
            allocate_node_id_restaurant.append(df_to.iloc[group_values_to[j][l]]["node_ids"])

    allocate_var_id_school = [(j, a) for (j, a) in allocations if a==2]
    allocate_row_id_school = []
    allocate_node_id_school = []
    for (j, a) in allocate_var_id_school:
        for l in range(int(y[(j, a)].x)):
            allocate_row_id_school.append(group_values_to[j][l])
            allocate_node_id_school.append(df_to.iloc[group_values_to[j][l]]["node_ids"])

    allocated_D = {
        "allocate_var_id_grocery": allocate_var_id_grocery,
        "allocate_node_id_grocery": allocate_node_id_grocery,
        "allocate_row_id_grocery": allocate_row_id_grocery,
        "allocate_var_id_restaurant": allocate_var_id_restaurant,
        "allocate_node_id_restaurant": allocate_node_id_restaurant,
        "allocate_row_id_restaurant": allocate_row_id_restaurant,
        "allocate_var_id_school": allocate_var_id_school,
        "allocate_row_id_school": allocate_row_id_school,
        "allocate_node_id_school": allocate_node_id_school
    }

    assignments = [(i, j, a) for (i, j, a) in x.keys() if (x[(i, j, a)].x > EPS)]

    assign_from_var_id_grocery = [i for (i, j, a) in assignments if a==0]
    assign_to_var_id_grocery = [j for (i, j, a) in assignments if a==0]
    assign_from_node_id_grocery = [df_from.iloc[group_values_from[i][0]]["node_ids"] for (i, j, a) in assignments if a==0]
    assign_to_node_id_grocery = []
    assign_type_grocery = []
    dist_grocery =[]

    assign_from_var_id_restaurant = [i for (i, j, a) in assignments if a == 0]
    assign_to_var_id_restaurant = [j for (i, j, a) in assignments if a == 0]
    assign_from_node_id_restaurant = [df_from.iloc[group_values_from[i][0]]["node_ids"] for (i, j, a) in assignments if a == 1]
    assign_to_node_id_restaurant = []
    assign_type_restaurant = []
    dist_restaurant = []

    assign_from_var_id_school = [i for (i, j, a) in assignments if a == 0]
    assign_to_var_id_school = [j for (i, j, a) in assignments if a == 0]
    assign_from_node_id_school = [df_from.iloc[group_values_from[i][0]]["node_ids"] for (i, j, a) in assignments if a == 2]
    assign_to_node_id_school = []
    assign_type_school = []
    dist_school = []

    for (i, j, a) in assignments:
        if a==0:
            if j < num_allocation:
                assign_to_node_id_grocery.append(df_to.iloc[group_values_to[j][0]]["node_ids"])
                assign_type_grocery.append('allocated')
                dist_grocery.append(SP_matrix[df_from.iloc[group_values_from[i][0]]["node_ids"], df_to.iloc[group_values_to[j][0]]["node_ids"]])
            else:
                assign_to_node_id_grocery.append(grocery_df.iloc[j-num_allocation]["node_ids"])
                assign_type_grocery.append('existing')
                dist_grocery.append(SP_matrix[df_from.iloc[group_values_from[i][0]]["node_ids"],grocery_df.iloc[j-num_allocation]["node_ids"]])
        elif a==1:
            if j < num_allocation:
                assign_to_node_id_restaurant.append(df_to.iloc[group_values_to[j][0]]["node_ids"])
                assign_type_restaurant.append('allocated')
                dist_restaurant.append(SP_matrix[df_from.iloc[group_values_from[i][0]]["node_ids"], df_to.iloc[group_values_to[j][0]]["node_ids"]])
            else:
                assign_to_node_id_restaurant.append(restaurant_df.iloc[j-num_allocation-num_cur_grocery]["node_ids"])
                assign_type_restaurant.append('existing')
                dist_restaurant.append(SP_matrix[df_from.iloc[group_values_from[i][0]]["node_ids"],restaurant_df.iloc[j-num_allocation-num_cur_grocery]["node_ids"]])
        elif a==2:
            if j < num_allocation:
                assign_to_node_id_school.append(df_to.iloc[group_values_to[j][0]]["node_ids"])
                assign_type_school.append('allocated')
                dist_school.append(SP_matrix[df_from.iloc[group_values_from[i][0]]["node_ids"], df_to.iloc[group_values_to[j][0]]["node_ids"]])
            else:
                assign_to_node_id_school.append(school_df.iloc[j-num_allocation-num_cur_restaurant-num_cur_grocery]["node_ids"])
                assign_type_school.append('existing')
                dist_school.append(SP_matrix[df_from.iloc[group_values_from[i][0]]["node_ids"],school_df.iloc[j-num_allocation-num_cur_restaurant-num_cur_grocery]["node_ids"]])

    assigned_D = {
        "assign_from_var_id_grocery": assign_from_var_id_grocery,
        "assign_to_var_id_grocery": assign_to_var_id_grocery,
        "assign_from_node_id_grocery": assign_from_node_id_grocery,
        "assign_to_node_id_grocery": assign_to_node_id_grocery,
        "assign_type_grocery": assign_type_grocery,
        "dist_grocery": dist_grocery,
        "assign_from_var_id_restaurant": assign_from_var_id_restaurant,
        "assign_to_var_id_restaurant": assign_to_var_id_restaurant,
        "assign_from_node_id_restaurant": assign_from_node_id_restaurant,
        "assign_to_node_id_restaurant": assign_to_node_id_restaurant,
        "assign_type_restaurant": assign_type_restaurant,
        "dist_restaurant": dist_restaurant,
        "assign_from_var_id_school": assign_from_var_id_school,
        "assign_to_var_id_school": assign_to_var_id_school,
        "assign_from_node_id_school": assign_from_node_id_school,
        "assign_to_node_id_school": assign_to_node_id_school,
        "assign_type_school": assign_type_school,
        "dist_school": dist_school
    }

    obj = m.getObjective()
    obj_value = obj.getValue()

    return obj_value, [np.mean(dist_grocery), np.mean(dist_restaurant), np.mean(dist_school)],   m.Runtime, m, allocated_D, assigned_D, num_residents, num_allocation, [num_cur_grocery, num_cur_restaurant, num_cur_school], m.status


def opt_multiple_depth(df_from,df_to,grocery_df, restaurant_df, school_df, SP_matrix, k_array, threads, results_sava_path, bp, focus,EPS=0.5):
    '''multiple amenity case, with depth of choice'''

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
    range_grocery_existing = list(range(cur_index, cur_index + num_cur_grocery))
    range_grocery_dest_list = list(range(num_allocation)) + range_grocery_existing
    cur_index+=num_cur_grocery
    range_restaurant_existing = list(range(cur_index, cur_index + num_cur_restaurant))
    range_restaurant_dest_list = list(range(num_allocation)) + range_restaurant_existing
    cur_index+=num_cur_restaurant
    range_school_existing = list(range(cur_index, cur_index + num_cur_school))
    range_school_dest_list = list(range(num_allocation)) + range_school_existing

    tot_choices = min(k_array[1] + num_cur_restaurant, len(w_choice_multi_amenity))
    no_choices = list(range(tot_choices, len(w_choice_multi_amenity)))

    cartesian_prod_assign_grocery = list(product(range(num_residents), range_grocery_dest_list, [0]))
    cartesian_prod_assign_restaurant = list(product(range(num_residents),range_restaurant_dest_list, [1], range(tot_choices)))
    cartesian_prod_assign_school = list(product(range(num_residents), range_school_dest_list, [2]))

    #cartesian_prod_allocate = list(product(range(num_residents), list(range(num_allocation)), [0,1,2]))

    # retrieve distances
    d = {(i, j): SP_matrix[df_from.iloc[group_values_from[i][0]]["node_ids"], df_to.iloc[group_values_to[j][0]]["node_ids"]] for i, j in list(product(range(num_residents), range(num_allocation)))}

    for i in range(num_residents):
        start_id = num_allocation
        for amenity_df in [grocery_df, restaurant_df, school_df]:
            for inst_row in range(len(amenity_df)):
                cur_id = start_id + inst_row
                d[(i, cur_id)] = SP_matrix[df_from.iloc[group_values_from[i][0]]["node_ids"], amenity_df.iloc[inst_row]["node_ids"]]
            start_id += len(amenity_df)

    x = m.addVars(cartesian_prod_assign_grocery + cartesian_prod_assign_school, vtype=GRB.BINARY, name='assign')
    x_choice = m.addVars(cartesian_prod_assign_restaurant, vtype=GRB.BINARY, name='assign')
    y = m.addVars(list(product(range(num_allocation), range(len(k_array)))), vtype=GRB.INTEGER, name='activate')
    l = m.addVars(num_residents, vtype=GRB.CONTINUOUS, name='dist')
    f = m.addVars(num_residents, vtype=GRB.CONTINUOUS, ub=100, name='score')

    # branching priority
    if bp:
        for t in list(product(range(num_allocation), range(len(k_array)))):
            y[t].setAttr("BranchPriority", 100)
        # for (n,m) in cartesian_prod_assign:
        #     x[(n,m)].setAttr("BranchPriority",4)


    # Constraints
    ## weighted distance
    no_choice_sum = sum([w_choice_multi_amenity[c] * L_a[-2] for c in no_choices])
    m.addConstrs(l[i] == (
                 (weights_array[0] * gp.quicksum(x[(i, j, 0)] * d[(i, j)] for j in range_grocery_dest_list))
                + (gp.quicksum(w_choice_multi_amenity[c] * (gp.quicksum(x_choice[(i, j, 1, c)] * d[(i, j)] for j in range_restaurant_dest_list)) for c in range(tot_choices)) + no_choice_sum)
                + (weights_array[2] * gp.quicksum(x[(i, j, 2)] * d[(i, j)] for j in range_school_dest_list))
                )
                 for i in range(num_residents))
    # PWL score
    for i in range(num_residents):
        m.addGenConstrPWL(l[i], f[i], L_a, L_f_a)
    ## assgined to one instance of amenity
    m.addConstrs((gp.quicksum(x[(i, j, 0)] for j in range_grocery_dest_list) == 1 for i in range(num_residents)), name='grocery demand')
    #m.addConstrs((gp.quicksum(x[(i, j, 1)] for j in range_restaurant_dest_list) == 1 for i in range(num_residents)), name='restaurant demand')
    m.addConstrs((gp.quicksum(x[(i, j, 2)] for j in range_school_dest_list) == 1 for i in range(num_residents)), name='school demand')
    ## assign choices
    m.addConstrs(((gp.quicksum(x_choice[(i, j, 1, c)] for j in range_restaurant_dest_list) == 1) for c in range(tot_choices) for i in range(num_residents)), name='choices')
    ## resource constraint
    m.addConstrs(((gp.quicksum(y[(j,a)] for a in range(len(weights_array))) <= capacity[j]) for j in range(num_allocation)), name='capacity')
    # activation
    m.addConstrs((x[(i,j,a)] <= y[(j,a)] for (i, j ,a) in list(product(range(num_residents), list(range(num_allocation)), [0,2]))), name='activation1')
    m.addConstrs((x_choice[(i, j, a, c)] <= y[(j, a)] for (i, j, a, c) in list(product(range(num_residents), list(range(num_allocation)),[1], range(tot_choices)))), name='activation2')
    # resource constraint
    m.addConstrs(((gp.quicksum(y[(j, a)] for j in range(num_allocation)) <= k_array[a]) for a in range(len(k_array))), name='resource')

    # choices can not be the same place
    ## newly allocated
    m.addConstrs(((gp.quicksum(x_choice[(i, j, 1, c)] for c in range(tot_choices)) <= y[(j, 1)]) for j in range(num_allocation) for i in range(num_residents)), name='choices')
    ## currently existing
    m.addConstrs(((gp.quicksum(x_choice[(i, j, 1, c)] for c in range(tot_choices)) <= 1) for j in range_restaurant_existing for i in range(num_residents)), name='choices')

    # objective
    m.Params.Threads = threads
    m.setObjective(gp.quicksum(f[n] for n in range(num_residents))/num_residents, GRB.MAXIMIZE)
    m.setParam("LogFile", results_sava_path)
    m.Params.TimeLimit = time_limit
    m.Params.MIPFocus = focus
    m.Params.NodefileStart = 0.5

    m.optimize()

    allocations = [(j, a) for (j, a) in y.keys() if (y[(j, a)].x) > EPS]

    # save allocation solutions
    # grocery
    allocate_var_id_grocery_ = [(j, a) for (j, a) in allocations if a==0]
    allocate_var_id_grocery = []
    allocate_row_id_grocery = []
    allocate_node_id_grocery = []
    for (j, a) in allocate_var_id_grocery_:
        for l in range(int(y[(j, a)].x)):
            allocate_var_id_grocery.append(j)
            allocate_row_id_grocery.append(group_values_to[j][l])
            allocate_node_id_grocery.append(df_to.iloc[group_values_to[j][l]]["node_ids"])

    # restaurant
    allocate_var_id_restaurant_ = [(j, a) for (j, a) in allocations if a==1]
    allocate_var_id_restaurant = []
    allocate_row_id_restaurant = []
    allocate_node_id_restaurant = []
    for (j, a) in allocate_var_id_restaurant_:
        for l in range(int(y[(j, a)].x)):
            allocate_var_id_restaurant.append(j)
            allocate_row_id_restaurant.append(group_values_to[j][l])
            allocate_node_id_restaurant.append(df_to.iloc[group_values_to[j][l]]["node_ids"])

    # school
    allocate_var_id_school_ = [(j, a) for (j, a) in allocations if a==2]
    allocate_var_id_school = []
    allocate_row_id_school = []
    allocate_node_id_school = []
    for (j, a) in allocate_var_id_school_:
        for l in range(int(y[(j, a)].x)):
            allocate_var_id_school.append(j)
            allocate_row_id_school.append(group_values_to[j][l])
            allocate_node_id_school.append(df_to.iloc[group_values_to[j][l]]["node_ids"])

    allocated_D = {
        "allocate_var_id_grocery": allocate_var_id_grocery,
        "allocate_node_id_grocery": allocate_node_id_grocery,
        "allocate_row_id_grocery": allocate_row_id_grocery,
        "allocate_var_id_restaurant": allocate_var_id_restaurant,
        "allocate_node_id_restaurant": allocate_node_id_restaurant,
        "allocate_row_id_restaurant": allocate_row_id_restaurant,
        "allocate_var_id_school": allocate_var_id_school,
        "allocate_row_id_school": allocate_row_id_school,
        "allocate_node_id_school": allocate_node_id_school
    }

    # assignments

    assignments = [(i, j, a) for (i, j, a) in x.keys() if (x[(i, j, a)].x > EPS)]
    choice_assignments = [(i,j,a,c) for (i,j,a,c) in x_choice.keys() if (x_choice[(i,j,a,c)].x > EPS)]

    # grocery

    assign_from_var_id_grocery = [i for (i, j, a) in assignments if a==0]
    assign_to_var_id_grocery = [j for (i, j, a) in assignments if a==0]
    assign_from_node_id_grocery = [df_from.iloc[group_values_from[i][0]]["node_ids"] for (i, j, a) in assignments if a==0]
    assign_to_node_id_grocery = []
    assign_type_grocery = []
    dist_grocery =[]

    # school

    assign_from_var_id_school = [i for (i, j, a) in assignments if a == 0]
    assign_to_var_id_school = [j for (i, j, a) in assignments if a == 0]
    assign_from_node_id_school = [df_from.iloc[group_values_from[i][0]]["node_ids"] for (i, j, a) in assignments if a == 2]
    assign_to_node_id_school = []
    assign_type_school = []
    dist_school = []

    for (i, j, a) in assignments:
        if a==0:
            if j < num_allocation:
                assign_to_node_id_grocery.append(df_to.iloc[group_values_to[j][0]]["node_ids"])
                assign_type_grocery.append('allocated')
                dist_grocery.append(SP_matrix[df_from.iloc[group_values_from[i][0]]["node_ids"], df_to.iloc[group_values_to[j][0]]["node_ids"]])
            else:
                assign_to_node_id_grocery.append(grocery_df.iloc[j-num_allocation]["node_ids"])
                assign_type_grocery.append('existing')
                dist_grocery.append(SP_matrix[df_from.iloc[group_values_from[i][0]]["node_ids"],grocery_df.iloc[j-num_allocation]["node_ids"]])
        elif a==2:
            if j < num_allocation:
                assign_to_node_id_school.append(df_to.iloc[group_values_to[j][0]]["node_ids"])
                assign_type_school.append('allocated')
                dist_school.append(SP_matrix[df_from.iloc[group_values_from[i][0]]["node_ids"], df_to.iloc[group_values_to[j][0]]["node_ids"]])
            else:
                assign_to_node_id_school.append(school_df.iloc[j-num_allocation-num_cur_restaurant-num_cur_grocery]["node_ids"])
                assign_type_school.append('existing')
                dist_school.append(SP_matrix[df_from.iloc[group_values_from[i][0]]["node_ids"],school_df.iloc[j-num_allocation-num_cur_restaurant-num_cur_grocery]["node_ids"]])

    assigned_D = {
        "assign_from_var_id_grocery": assign_from_var_id_grocery,
        "assign_to_var_id_grocery": assign_to_var_id_grocery,
        "assign_from_node_id_grocery": assign_from_node_id_grocery,
        "assign_to_node_id_grocery": assign_to_node_id_grocery,
        "assign_type_grocery": assign_type_grocery,
        "dist_grocery": dist_grocery,

        "assign_from_var_id_school": assign_from_var_id_school,
        "assign_to_var_id_school": assign_to_var_id_school,
        "assign_from_node_id_school": assign_from_node_id_school,
        "assign_to_node_id_school": assign_to_node_id_school,
        "assign_type_school": assign_type_school,
        "dist_school": dist_school
    }

    # restaurant

    for choice in range(tot_choices):
        assignments = [(i, j, a, c) for (i, j, a, c) in choice_assignments if (c == choice)]

        assign_from_var_id = [i for (i, j, a, c) in assignments]
        assign_to_var_id = [j for (i, j, a, c) in assignments]
        assign_from_node_id = [df_from.iloc[group_values_from[i][0]]["node_ids"] for (i, j, a, c) in assignments]
        assign_to_node_id = []
        assign_type = []
        dist=[]
        for (i, j, a, c) in assignments:
            if j < num_allocation:
                assign_to_node_id.append(df_to.iloc[group_values_to[j][0]]["node_ids"])
                assign_type.append('allocated')
                dist.append(SP_matrix[df_from.iloc[group_values_from[i][0]]["node_ids"], df_to.iloc[group_values_to[j][0]]["node_ids"]])
            else:
                assign_to_node_id.append(restaurant_df.iloc[j-num_allocation-num_cur_grocery]["node_ids"])
                assign_type.append('existing')
                dist.append(SP_matrix[df_from.iloc[group_values_from[i][0]]["node_ids"],restaurant_df.iloc[j-num_allocation-num_cur_grocery]["node_ids"]])

        assigned_D[str(choice)+"_assign_from_var_id_restaurant"]=assign_from_var_id
        assigned_D[str(choice)+"_assign_to_var_id_restaurant"]=assign_to_var_id
        assigned_D[str(choice)+"_assign_from_node_id_restaurant"]=assign_from_node_id
        assigned_D[str(choice)+"_assign_to_node_id_restaurant"]=assign_to_node_id
        assigned_D[str(choice)+"_assign_type_restaurant"]=assign_type
        assigned_D[str(choice)+"_dist_restaurant"]=dist


    obj = m.getObjective()
    obj_value = obj.getValue()

    restaurant_dist_obj = [np.mean(assigned_D[str(c) + "_dist_restaurant"]) if (str(c) + "_dist_restaurant") in assigned_D.keys() else 0 for c in range(len(choice_weights))]

    return obj_value, [np.mean(dist_grocery), restaurant_dist_obj, np.mean(dist_school)], m.Runtime, m, allocated_D, assigned_D, num_residents, num_allocation, [num_cur_grocery, num_cur_restaurant, num_cur_school], m.status


def cur_assignment_single_depth(df_from,amenity_df, SP_matrix,bp, focus,EPS=1.e-6):
    ''' get assignment for the case with no allocation, considering depth of choice'''

    m = gp.Model('cur_assignment')
    groups_from = df_from.groupby('node_ids').groups
    group_values_from = list(groups_from.values())
    num_residents = len(group_values_from)
    num_amenity = len(amenity_df)

    cartesian_prod = list(product(range(num_residents), range(num_amenity)))  # a list of tuples
    cartesian_prod_assign = list(product(range(num_residents), range(num_amenity), range(len(choice_weights))))
    distances = {(i, j): SP_matrix[df_from.iloc[group_values_from[i][0]]["node_ids"], amenity_df.iloc[j]["node_ids"]] for i, j in cartesian_prod}
    x = m.addVars(cartesian_prod_assign, vtype=GRB.BINARY, name='Assign')

    ## assign choices
    tot_choices = min(num_amenity, len(choice_weights))
    no_choices = list(range(tot_choices, len(choice_weights)))
    m.addConstrs(
        (gp.quicksum(x[(n, m, c)] for m in range(num_amenity)) == 1 for c in range(tot_choices) for n in range(num_residents)),
        name='choices')
    m.addConstrs((x[(n, m, c)] == 0 for c in no_choices for m in range(num_amenity) for n in range(num_residents)),
        name='no choices')

    # choices can not be the same place
    ## currently existing
    m.addConstrs(((gp.quicksum(x[(n, m, c)] for c in range(tot_choices)) <= 1) for m in range(num_amenity) for n in range(num_residents)), name='choices')
    # objective
    no_choice_sum =sum([choice_weights[c]*L_a[-2] for c in no_choices])
    m.setObjective((gp.quicksum(
        (gp.quicksum(
                     choice_weights[c]*(gp.quicksum(distances[(n, m)] * x[(n, m, c)] for m in range(num_amenity)))
                     for c in range(tot_choices))+no_choice_sum)
                    for n in range(num_residents))/num_residents), GRB.MINIMIZE)
    m.optimize()

    obj = m.getObjective()
    obj_value = obj.getValue() # min total dist

    assignments = [(i, j, c) for (i, j, c) in x.keys() if (x[i, j, c].x > EPS)]

    assigned_D={}

    choices_dist = []

    for choice in range(tot_choices):

        assign_from_var_id = [i for (i, j, c) in assignments if c==choice]
        assign_to_var_id = [j for (i, j, c) in assignments if c==choice]
        assign_from_node_id = [df_from.iloc[group_values_from[i][0]]["node_ids"] for (i, j, c) in assignments  if c==choice]
        assign_to_node_id = [amenity_df.iloc[j]["node_ids"] for (i, j, c) in assignments  if c==choice]
        dist = [SP_matrix[df_from.iloc[group_values_from[i][0]]["node_ids"], amenity_df.iloc[j]["node_ids"]] for (i, j, c) in assignments if c==choice]
        choices_dist.append(dist)

        assigned_D[str(choice)+"_assign_from_var_id"]=assign_from_var_id
        assigned_D[str(choice)+"_assign_to_var_id"]=assign_to_var_id
        assigned_D[str(choice)+"_assign_from_node_id"]=assign_from_node_id
        assigned_D[str(choice)+"_assign_to_node_id"]=assign_to_node_id
        assigned_D[str(choice)+"_dist"]=dist

    for choice in range(tot_choices, len(choice_weights)):
        choices_dist.append([L_a[-2]]*num_residents)

    obj = m.getObjective()
    obj_value = obj.getValue()
    dist_obj = [np.mean(assigned_D[str(c)+"_dist"]) if (str(c) + "_dist") in assigned_D.keys() else 0 for c in
                range(len(choice_weights))]

    choices_dist = np.array(choices_dist)
    weighted_choices = np.dot(np.array(choice_weights),  choices_dist)
    scores = dist_to_score(np.array(weighted_choices), L_a, L_f_a)
    score_obj = np.mean(scores)

    return score_obj, dist_obj, m.Runtime, m, assigned_D, num_residents, num_amenity, m.status


def cur_assignment_single(df_from,amenity_df, SP_matrix,bp, focus,EPS=1.e-6):
    ''' get assignment for the case with no allocation, no depth of choice'''

    m = gp.Model('cur_assignment')
    groups_from = df_from.groupby('node_ids').groups
    group_values_from = list(groups_from.values())
    num_residents = len(group_values_from)
    num_amenity = len(amenity_df)
    cartesian_prod = list(product(range(num_residents), range(num_amenity), ))  # a list of tuples
    distances = {(i, j): SP_matrix[df_from.iloc[group_values_from[i][0]]["node_ids"], amenity_df.iloc[j]["node_ids"]] for i, j in cartesian_prod}

    if len(amenity_df) == 0:
        print("no existing amenities!")
        return 0, None, None, None, None, num_residents, 0, None   # score_obj, obj_value, m.Runtime, m, assigned_D, num_residents, num_amenity, m.status


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

    return score_obj, obj_value, m.Runtime, m, assigned_D, num_residents, num_amenity, m.status

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


def opt_single_CP(df_from,df_to,amenity_df, SP_matrix,k,threads,results_sava_path,solver_path, EPS=0.5):
    '''single amenity case, no depth of choice'''

    if len(df_from)>0:
        df_from = df_from[['geometry', 'node_ids']]
    if len(df_to)>0:
        df_to = df_to[['geometry', 'node_ids']]

    model = CpoModel(name="max_score")

    # grouping
    groups_to=df_to.groupby('node_ids').groups # keys are node id, values are indices
    group_values_to=list(groups_to.values())
    num_allocation = len(group_values_to)
    capacity = [len(item) for item in group_values_to]

    groups_from = df_from.groupby('node_ids').groups
    group_values_from = list(groups_from.values())
    num_residents = len(group_values_from)

    num_cur = len(amenity_df)

    cartesian_prod_allocate = list(product(range(num_residents), range(num_allocation)))

    # retrieve distances
    d = {(i, j): SP_matrix[df_from.iloc[group_values_from[i][0]]["node_ids"], df_to.iloc[group_values_to[j][0]]["node_ids"]] for i, j in cartesian_prod_allocate}

    for i in range(num_residents):
        # dummy node: inf distance
        d[(i, num_allocation)] = L_a[-1]
        # distance to existing ones
        for l in range(num_cur):
            cur_id = num_allocation + 1 + l
            d[(i, cur_id)] = SP_matrix[df_from.iloc[group_values_from[i][0]]["node_ids"], amenity_df.iloc[l]["node_ids"]]

    # variables
    y = {}
    for k_ in range(k):
        y[k_] = model.integer_var(min=0, max=num_allocation,name=f'y[{k_}]') #include dummy node
    f = {}
    for i in range(num_residents):
        f[i] = model.float_var(min=0, max=100, name=f'f[{i}]')
    dist = {}
    for i in range(num_residents):
        for k_ in range(k):
            dist[(i, k_)] = model.float_var(min=0, max=L_a[-1], name=f'dist[{i},{k_}]')
    l = {}
    for i in range(num_residents):
        l[i] = model.float_var(min=0, max=L_a[-1], name=f'z[{i}]')

    # Constraints
    ## WalkScore
    # allocated
    for i in range(num_residents):
        for k_ in range(k):
            model.add(dist[(i,k_)] == (model.element([d[(i, m)] for m in range(num_allocation+1)], y[k_])))
    # existing
    for i in range(num_residents):
        model.add(l[i] == model.min([dist[(i,k_)] for k_ in range(k)] + [d[(i,j)] for j in range(num_allocation + 1, num_allocation + 1 + num_cur)]) )

    # # PWL
    for i in range(num_residents):
        #model.add(f[i] == model.slope_piecewise_linear(l[i], [400, 1800, 2400], [-0.0125, -0.0607, -0.0167, 0], 0, 100))
        model.add(f[i] == model.coordinate_piecewise_linear(l[i], -0.0125, [0, 400, 1800, 2400, 5000000], [100, 95, 10, 1, 1], 0))

    for j in range(num_allocation):
        model.add(model.count(list(y.values()),j)<=capacity[j])

    # symmetry breaking
    if k>1:
        for k_ in range(k-1):
            model.add(y[(k_)]<=y[(k_+1)])
    model.add(model.all_diff([y[k_] for k_ in range(k)]))

    # # objective
    model.add(model.maximize(model.sum(f[i] for i in range(num_residents))/num_residents))

    msol = model.solve(execfile=solver_path, TimeLimit=time_limit, Workers=threads)
    obj_value = msol.get_objective_values()[0][0]

    str = msol.solver_log
    with open(results_sava_path, 'w') as f:
        f.write(str)

    allocations = [msol[y[k_]] for k_ in y.keys() if msol[y[k_]] < num_allocation]  # exclude dummy node


    # save allocation solutions
    allocate_var_id = allocations
    allocate_row_id = []
    allocate_node_id = []
    for j in allocate_var_id:
        allocate_row_id.append(group_values_to[j][0])
        allocate_node_id.append(df_to.iloc[group_values_to[j][0]]["node_ids"])
    allocated_D = {
        "allocate_var_id": allocate_var_id,
        "allocate_node_id": allocate_node_id,
        "allocate_row_id": allocate_row_id
    }
    assign_from_var_id = list(range(num_residents))
    assign_from_node_id = [df_from.iloc[group_values_from[i][0]]["node_ids"] for i in assign_from_var_id]
    assign_to_var_id = []
    assign_to_node_id = []
    assign_type = []
    dist = []
    for i in range(num_residents):

        # new_allocated
        j_array = [msol[y[k_]] for k_ in range(k)]
        dist_array_new = [d[i, msol[y[k_]]] for k_ in range(k)]
        j_new = j_array[np.argmin(dist_array_new)]
        # existing
        j_array = [j-1 for j in range(num_allocation, num_allocation + 1 + num_cur)]
        dist_array_e = [d[(i,j)] for j in range(num_allocation, num_allocation + 1 + num_cur)]
        j_exist = j_array[np.argmin(dist_array_e)]
        if min(dist_array_new)<min(dist_array_e):
            assign_to_var_id.append(j_new)
            assign_to_node_id.append(df_to.iloc[group_values_to[j_new][0]]["node_ids"])
            assign_type.append('allocated')
            dist.append(SP_matrix[df_from.iloc[group_values_from[i][0]]["node_ids"], df_to.iloc[group_values_to[j_new][0]]["node_ids"]])
        else:
            assign_to_var_id.append(j_exist)
            assign_to_node_id.append(amenity_df.iloc[j_exist-num_allocation]["node_ids"])
            assign_type.append('existing')
            dist.append(SP_matrix[df_from.iloc[group_values_from[i][0]]["node_ids"],amenity_df.iloc[j_exist-num_allocation]["node_ids"]])


    assigned_D = {
        "assign_from_var_id": assign_from_var_id,
        "assign_to_var_id": assign_to_var_id,
        "assign_from_node_id": assign_from_node_id,
        "assign_to_node_id": assign_to_node_id,
        "assign_type": assign_type,
        "dist": dist}

    dist_obj = np.mean(dist)

    return obj_value, dist_obj, msol.get_solve_time(), msol, allocated_D, assigned_D, num_residents, num_allocation, num_cur, msol.solve_status

def opt_multiple_CP(df_from,df_to,grocery_df, restaurant_df, school_df, SP_matrix, k_array, threads,results_sava_path,solver_path,EPS=0.5):
    '''multiple amenity case, no depth of choice'''

    if len(df_from)>0:
        df_from = df_from[['geometry', 'node_ids']]
    if len(df_to)>0:
        df_to = df_to[['geometry', 'node_ids']]

    model = CpoModel(name="max_score")

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

    cur_index = num_allocation
    range_grocery_dest_list = list(range(cur_index, cur_index + num_cur_grocery))
    range_grocery_dest_list = [item + 1 for item in range_grocery_dest_list]
    cur_index+=num_cur_grocery
    range_restaurant_dest_list = list(range(cur_index, cur_index + num_cur_restaurant))
    range_restaurant_dest_list = [item + 1 for item in range_restaurant_dest_list]
    cur_index+=num_cur_restaurant
    range_school_dest_list = list(range(cur_index, cur_index + num_cur_school))
    range_school_dest_list = [item + 1 for item in range_school_dest_list]
    existing_list = [range_grocery_dest_list, range_restaurant_dest_list, range_school_dest_list]

    # retrieve distances
    d = {(i, j): SP_matrix[df_from.iloc[group_values_from[i][0]]["node_ids"], df_to.iloc[group_values_to[j][0]]["node_ids"]] for i, j in list(product(range(num_residents), range(num_allocation)))}

    for i in range(num_residents):
        # dummy node: inf distance
        d[(i, num_allocation)] = L_a[-1]
        start_id = num_allocation + 1
        for amenity_df in [grocery_df, restaurant_df, school_df]:
            for inst_row in range(len(amenity_df)):
                cur_id = start_id + inst_row
                d[(i, cur_id)] = SP_matrix[df_from.iloc[group_values_from[i][0]]["node_ids"], amenity_df.iloc[inst_row]["node_ids"]]
            start_id += len(amenity_df)

    # variables
    y = {}
    for a in range(len(k_array)):
        for k_ in range(k_array[a]):
            y[(k_, a)] = model.integer_var(min=0, max=num_allocation, name=f'y[{k_},{a}]')  # include dummy node
    dist = {}  # z
    for i in range(num_residents):
        for a in range(len(k_array)):
            for k_ in range(k_array[a]):
                dist[(i, a, k_)] = model.float_var(min=0, max=L_a[-1], name=f'dist[{i},{a},{k_}]')
    f = {}
    for i in range(num_residents):
        f[i] = model.float_var(min=0, max=100, name=f'f[{i}]')
    l = {}
    for i in range(num_residents):
        l[i] = model.float_var(min=0, max=L_a[-1], name=f'z[{i}]')

    # Constraints
    ## weighted distance
    for i in range(num_residents):
        model.add(l[i] == (model.sum((weights_array[a] * model.min([dist[(i,a,k_)] for k_ in range(k_array[a])] + [d[(i,j)] for j in existing_list[a]])) for a in range(len(k_array)))))
    # PWL
    for i in range(num_residents):
        #model.add(f[i]==model.slope_piecewise_linear(l[i], [400, 1800, 2400], [-0.0125, -0.0607, -0.0167, 0], 0, 100))
        model.add(f[i] == model.coordinate_piecewise_linear(l[i], -0.0125, [0, 400, 1800, 2400, 5000000],[100, 95, 10, 1, 1], 0))

    # distance element constrain
    for i in range(num_residents):
        for a in range(len(k_array)):
            for k_ in range(k_array[a]):
                model.add(
                    dist[(i, a, k_)] == (model.element([d[(i, m)] for m in range(num_allocation + 1)], y[(k_, a)])))

    for j in range(num_allocation):
        model.add(model.count(list(y.values()), j) <= capacity[j])

    # symmetry breaking
    for a in range(len(k_array)):
        if k_array[a]>1:
            for k_ in range((k_array[a])-1):
                model.add(y[(k_,a)]<=y[(k_+1,a)])
        #model.add(model.all_diff([y[(k_, a)] for k_ in range(k_array[a])]))

    # # objective
    model.add(model.maximize(model.sum(f[i] for i in range(num_residents))/num_residents))

    msol = model.solve(execfile=solver_path, TimeLimit=time_limit, Workers=threads)
    obj_value = msol.get_objective_values()[0][0]

    str = msol.solver_log
    with open(results_sava_path, 'w') as f:
        f.write(str)

    allocations = [(msol[y[k_,a]], a) for (k_, a) in y.keys() if msol[y[k_,a]] < num_allocation]

    # save allocation solutions
    allocate_var_id_grocery = [(j, a) for (j, a) in allocations if a==0]
    allocate_row_id_grocery = []
    allocate_node_id_grocery = []
    for (j, a) in allocate_var_id_grocery:
        allocate_row_id_grocery.append(group_values_to[j][0])
        allocate_node_id_grocery.append(df_to.iloc[group_values_to[j][0]]["node_ids"])

    allocate_var_id_restaurant = [(j, a) for (j, a) in allocations if a==1]
    allocate_row_id_restaurant = []
    allocate_node_id_restaurant = []
    for (j, a) in allocate_var_id_restaurant:
        allocate_row_id_restaurant.append(group_values_to[j][0])
        allocate_node_id_restaurant.append(df_to.iloc[group_values_to[j][0]]["node_ids"])

    allocate_var_id_school = [(j, a) for (j, a) in allocations if a==2]
    allocate_row_id_school = []
    allocate_node_id_school = []
    for (j, a) in allocate_var_id_school:
        allocate_row_id_school.append(group_values_to[j][0])
        allocate_node_id_school.append(df_to.iloc[group_values_to[j][0]]["node_ids"])

    allocated_D = {
        "allocate_var_id_grocery": allocate_var_id_grocery,
        "allocate_node_id_grocery": allocate_node_id_grocery,
        "allocate_row_id_grocery": allocate_row_id_grocery,
        "allocate_var_id_restaurant": allocate_var_id_restaurant,
        "allocate_node_id_restaurant": allocate_node_id_restaurant,
        "allocate_row_id_restaurant": allocate_row_id_restaurant,
        "allocate_var_id_school": allocate_var_id_school,
        "allocate_row_id_school": allocate_row_id_school,
        "allocate_node_id_school": allocate_node_id_school
    }

    assigned_D={}
    k_strs=["grocery","restaurant","school"]
    all_dfs =[grocery_df, restaurant_df, school_df]


    for a in range(len(k_array)):
        assign_from_var_id = list(range(num_residents))
        assigned_D["assign_from_var_id_"+k_strs[a]]=assign_from_var_id
        assign_from_node_id = [df_from.iloc[group_values_from[i][0]]["node_ids"] for i in assign_from_var_id]
        assign_to_var_id = []
        assign_to_node_id = []
        assign_type = []
        dist = []

        for i in range(num_residents):

            # new_allocated
            j_array = [msol[y[(k_,a)]] for k_ in range(k_array[a])]
            dist_array_new = [d[i, msol[y[(k_,a)]]] for k_ in range(k_array[a])]
            j_new = j_array[np.argmin(dist_array_new)]
            # existing
            if len(existing_list[a])>0:
                j_array = [j-1 for j in existing_list[a]]
                dist_array_e = [d[(i,j)] for j in existing_list[a]]
                j_exist = j_array[np.argmin(dist_array_e)]
                if min(dist_array_new)<min(dist_array_e):
                    assign_to_var_id.append(j_new)
                    assign_to_node_id.append(df_to.iloc[group_values_to[j_new][0]]["node_ids"])
                    assign_type.append('allocated')
                    dist.append(SP_matrix[df_from.iloc[group_values_from[i][0]]["node_ids"], df_to.iloc[group_values_to[j_new][0]]["node_ids"]])
                else:
                    assign_to_var_id.append(j_exist)
                    if a==0:
                        ind = j_exist-num_allocation
                    elif a==1:
                        ind = j_exist-num_allocation-num_cur_grocery
                    else:
                        ind = j_exist - num_allocation - num_cur_grocery-num_cur_restaurant
                    assign_to_node_id.append(all_dfs[a].iloc[ind]["node_ids"])
                    assign_type.append('existing')
                    dist.append(SP_matrix[df_from.iloc[group_values_from[i][0]]["node_ids"],all_dfs[a].iloc[ind]["node_ids"]])
            else:
                assign_to_var_id.append(j_new)
                assign_to_node_id.append(df_to.iloc[group_values_to[j_new][0]]["node_ids"])
                assign_type.append('allocated')
                dist.append(SP_matrix[df_from.iloc[group_values_from[i][0]]["node_ids"], df_to.iloc[group_values_to[j_new][0]]["node_ids"]])
        assigned_D["assign_from_var_id_" + k_strs[a]] = assign_from_var_id
        assigned_D["assign_to_var_id_"+ k_strs[a]] = assign_to_var_id
        assigned_D["assign_from_node_id_"+ k_strs[a]] = assign_from_node_id
        assigned_D["assign_to_node_id_"+ k_strs[a]] = assign_to_node_id
        assigned_D["assign_type_"+ k_strs[a]] = assign_type
        assigned_D["dist_"+ k_strs[a]] = dist

    return obj_value, [np.mean(assigned_D["dist_grocery"]), np.mean(assigned_D["dist_restaurant"]), np.mean(assigned_D["dist_school"])],   msol.get_solve_time(),  msol, allocated_D, assigned_D, num_residents, num_allocation, [num_cur_grocery, num_cur_restaurant, num_cur_school],  msol.solve_status


def opt_single_depth_CP(df_from,df_to,amenity_df, SP_matrix,k,threads,results_sava_path, solver_path,EPS=0.5):
    '''single amenity case, with consideration of depth of choice. For amenity=restaurant specifically'''

    if len(df_from)>0:
        df_from = df_from[['geometry', 'node_ids']]
    if len(df_to)>0:
        df_to = df_to[['geometry', 'node_ids']]

    model = CpoModel(name="max_score")

    # grouping
    groups_to=df_to.groupby('node_ids').groups # keys are node id, values are indices
    group_values_to=list(groups_to.values())
    num_allocation = len(group_values_to)
    capacity = [len(item) for item in group_values_to]

    groups_from = df_from.groupby('node_ids').groups
    group_values_from = list(groups_from.values())
    num_residents = len(group_values_from)

    num_cur = len(amenity_df)

    tot_choices = min(k + num_cur, len(choice_weights))
    no_choices = list(range(tot_choices, len(choice_weights)))

    cartesian_prod_assign = list(product(range(num_residents), range(num_allocation + num_cur), range(tot_choices)))  # last index is for depth of choice
    cartesian_prod_allocate = list(product(range(num_residents), range(num_allocation)))

    # retrieve distances
    d = {(i, j): SP_matrix[df_from.iloc[group_values_from[i][0]]["node_ids"], df_to.iloc[group_values_to[j][0]]["node_ids"]] for i, j in cartesian_prod_allocate}

    for i in range(num_residents):
        # dummy node: inf distance
        d[(i, num_allocation)] = L_a[-1]
        # distance to existing ones
        for l in range(num_cur):
            cur_id = num_allocation + 1 + l
            d[(i, cur_id)] = SP_matrix[
                df_from.iloc[group_values_from[i][0]]["node_ids"], amenity_df.iloc[l]["node_ids"]]

    # variables
    y = {}
    for k_ in range(k):
        y[k_] = model.integer_var(min=0, max=num_allocation,name=f'y[{k_}]') #include dummy node
    x = {}
    for i in range(num_residents):
        for c in range(tot_choices):
            x[(i, c)] = model.integer_var(min=0, max=(num_allocation + num_cur),name=f'x[{i},{c}]')
    f = {}
    for i in range(num_residents):
        f[i] = model.float_var(min=0, max=100, name=f'f[{i}]')
    dist = {}
    for i in range(num_residents):
        for c in range(tot_choices):
            dist[(i, c)] = model.float_var(min=0, max=L_a[-1], name=f'dist[{i},{c}]')
    l = {}
    for i in range(num_residents):
        l[i] = model.float_var(min=0, max=L_a[-1], name=f'z[{i}]')

    # Constraints
    # distance element constrain
    for i in range(num_residents):
        for c in range(tot_choices):
            model.add(dist[(i, c)] == (model.element((d[(i, m)] for m in range(num_allocation+1+num_cur)), x[(i, c)])))
    # calculate dist
    no_choice_sum = sum([choice_weights[c] * L_a[-2] for c in no_choices])
    for i in range(num_residents):
        model.add(l[i] == (model.sum(choice_weights[c] * dist[(i, c)] for c in range(tot_choices)) + no_choice_sum))
    # PWL
    for i in range(num_residents):
        #model.add(model.slope_piecewise_linear(l[i], [400, 1800, 2400], [-0.0125, -0.0607, -0.0167, 0], 0, 100) == f[i])
        model.add(f[i] == model.coordinate_piecewise_linear(l[i], -0.0125, [0, 400, 1800, 2400, 5000000], [100, 95, 10, 0.5, 0.5],0))

    ## activation
    for j in range(num_allocation):
        cond = [(x[(i, c)] == j) for i in range(num_residents) for c in range(tot_choices)]
        cond2 = [(y[k_] == j) for k_ in range(k)]
        model.add(model.if_then(model.any(cond),model.any(cond2)))
    ## node capacity
    for j in range(num_allocation):
        model.add(model.count(list(y.values()), j) <= capacity[j])
    # choices can not be the same place
    # newly allocated
    for i in range(num_residents):
        for j in range(num_allocation):
            model.add(model.count([x[(i, c)] for c in range(tot_choices)], j) <= model.count(list(y.values()), j))
        for m in range(num_allocation + 1, num_allocation + 1 + num_cur):
            model.add(model.count([x[(i, c)] for c in range(tot_choices)], m) <= 1)

    # symmetry breaking
    if k>1:
        for k_ in range(k-1):
            model.add(y[(k_)]<=y[(k_+1)])
    if tot_choices>1:
        for c in range(tot_choices - 1):
            for i in range(num_residents):
                model.add(dist[(i, c)]<=dist[(i, c+1)])

    # # objective
    model.add(model.maximize(model.sum(f[i] for i in range(num_residents))/num_residents))

    msol = model.solve(execfile=solver_path, TimeLimit=time_limit, Workers=threads)
    obj_value = msol.get_objective_values()[0][0]

    log_str = msol.solver_log
    with open(results_sava_path, 'w') as f:
        f.write(log_str)

    allocations = [msol[y[k_]] for k_ in y.keys() if msol[y[k_]] < num_allocation]  # exclude dummy node

    # save allocation solutions
    allocate_var_id = allocations
    allocate_row_id = []
    allocate_node_id = []
    for j in allocate_var_id:
        allocate_row_id.append(group_values_to[j][0])
        allocate_node_id.append(df_to.iloc[group_values_to[j][0]]["node_ids"])
    allocated_D = {
        "allocate_var_id": allocate_var_id,
        "allocate_node_id": allocate_node_id,
        "allocate_row_id": allocate_row_id
    }

    # save assignment solutions

    assigned_D={}

    for choice in range(tot_choices):

        assign_from_var_id = list(range(num_residents))
        assign_from_node_id = [df_from.iloc[group_values_from[i][0]]["node_ids"] for i in assign_from_var_id]
        assign_to_var_id = []
        assign_to_node_id = []
        assign_type = []
        dist = []

        for i in range(num_residents):

            # new_allocated
            j = msol[x[(i,choice)]]
            if j < num_allocation:
                assign_to_var_id.append(j)
                assign_to_node_id.append(df_to.iloc[group_values_to[j][0]]["node_ids"])
                assign_type.append('allocated')
                dist.append(SP_matrix[df_from.iloc[group_values_from[i][0]]["node_ids"],
                                      df_to.iloc[group_values_to[j][0]]["node_ids"]])
            else:
                assign_to_var_id.append(j-1)
                assign_to_node_id.append(amenity_df.iloc[j-1 - num_allocation]["node_ids"])
                assign_type.append('existing')
                dist.append(SP_matrix[df_from.iloc[group_values_from[i][0]]["node_ids"],
                                      amenity_df.iloc[j-1 - num_allocation]["node_ids"]])

        assigned_D[str(choice) + "_assign_from_var_id"] = assign_from_var_id
        assigned_D[str(choice) + "_assign_to_var_id"] = assign_to_var_id
        assigned_D[str(choice) + "_assign_from_node_id"] = assign_from_node_id
        assigned_D[str(choice) + "_assign_to_node_id"] = assign_to_node_id
        assigned_D[str(choice) + "_assign_type"] = assign_type
        assigned_D[str(choice) + "_dist"] = dist

    dist_obj = [np.mean(assigned_D[str(c)+"_dist"]) if (str(c)+"_dist") in assigned_D.keys() else 0 for c in range(len(choice_weights))]

    return obj_value, dist_obj, msol.get_solve_time(), msol, allocated_D, assigned_D, num_residents, num_allocation, num_cur, msol.solve_status


def opt_multiple_depth_CP(df_from,df_to,grocery_df, restaurant_df, school_df, SP_matrix, k_array, threads, results_sava_path, solver_path,EPS=0.5):
    '''multiple amenity case, with depth of choice'''

    if len(df_from)>0:
        df_from = df_from[['geometry', 'node_ids']]
    if len(df_to)>0:
        df_to = df_to[['geometry', 'node_ids']]

    model = CpoModel(name="max_score")

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

    cur_index = num_allocation
    range_grocery_dest_list = list(range(cur_index, cur_index + num_cur_grocery))
    range_grocery_dest_list = [item + 1 for item in range_grocery_dest_list]
    cur_index+=num_cur_grocery
    range_restaurant_dest_list = list(range(cur_index, cur_index + num_cur_restaurant))
    range_restaurant_dest_list = [item + 1 for item in range_restaurant_dest_list]
    cur_index+=num_cur_restaurant
    range_school_dest_list = list(range(cur_index, cur_index + num_cur_school))
    range_school_dest_list = [item + 1 for item in range_school_dest_list]
    existing_list = [range_grocery_dest_list, range_restaurant_dest_list, range_school_dest_list]

    tot_choices = min(k_array[1] + num_cur_restaurant, len(w_choice_multi_amenity))
    no_choices = list(range(tot_choices, len(w_choice_multi_amenity)))

    # retrieve distances
    d = {(i, j): SP_matrix[df_from.iloc[group_values_from[i][0]]["node_ids"], df_to.iloc[group_values_to[j][0]]["node_ids"]] for i, j in list(product(range(num_residents), range(num_allocation)))}

    for i in range(num_residents):
        # dummy node: inf distance
        d[(i, num_allocation)] = L_a[-1]
        start_id = num_allocation + 1
        for amenity_df in [grocery_df, restaurant_df, school_df]:
            for inst_row in range(len(amenity_df)):
                cur_id = start_id + inst_row
                d[(i, cur_id)] = SP_matrix[df_from.iloc[group_values_from[i][0]]["node_ids"], amenity_df.iloc[inst_row]["node_ids"]]
            start_id += len(amenity_df)

    # variables
    y = {}
    for a in range(len(k_array)):
        for k_ in range(k_array[a]):
            y[(k_, a)] = model.integer_var(min=0, max=num_allocation, name=f'y[{k_},{a}]')  # include dummy node
    # for restaurant
    x = {}
    for i in range(num_residents):
        for c in range(tot_choices):
            x[(i, c)] = model.integer_var(name=f'x[{i},{c}]')
            x[(i, c)].set_domain(list(range(0,num_allocation)) + range_restaurant_dest_list)
    # for grocery and school
    dist = {}  # z
    for i in range(num_residents):
        for a in [0,2]:
            for k_ in range(k_array[a]):
                dist[(i, a, k_)] = model.float_var(min=0, max=L_a[-1], name=f'dist[{i},{a},{k_}]')
    f = {}
    for i in range(num_residents):
        f[i] = model.float_var(min=0, max=100, name=f'f[{i}]')
    l = {}
    for i in range(num_residents):
        l[i] = model.float_var(min=0, max=L_a[-1], name=f'z[{i}]')
    # distance to each restaurant choice
    dist_r = {}
    for i in range(num_residents):
        for c in range(tot_choices):
            dist_r[(i, c)] = model.float_var(min=0, max=L_a[-1], name=f'dist[{i},{c}]')

    # Constraints
    ## weighted distance
    no_choice_sum = sum([w_choice_multi_amenity[c] * L_a[-2] for c in no_choices])
    for i in range(num_residents):
        model.add(l[i] ==(weights_array[0] * model.min([dist[(i,0,k_)] for k_ in range(k_array[0])] + [d[(i,j)] for j in existing_list[0]]))
                    + (model.sum(w_choice_multi_amenity[c] * dist_r[(i, c)] for c in range(tot_choices)) + no_choice_sum)
                    + (weights_array[2] * model.min([dist[(i, 2, k_)] for k_ in range(k_array[2])] + [d[(i, j)] for j in existing_list[2]]))
              )

    for i in range(num_residents):
        for a in [0,2]:
            for k_ in range(k_array[a]):
                model.add(
                    dist[(i, a, k_)] == (model.element([d[(i, m)] for m in range(num_allocation + 1)], y[(k_, a)])))

    for i in range(num_residents):
        for c in range(tot_choices):
            model.add(dist_r[(i, c)] == (model.element(([d[(i, m)] for m in range(num_allocation + 1)] + [d[(i, j)] for j in range_restaurant_dest_list]), x[(i, c)])))

    # PWL
    for i in range(num_residents):
        #model.add(model.slope_piecewise_linear(l[i], [400, 1800, 2400], [-0.0125, -0.0607, -0.0167, 0], 0, 100) == f[i])
        model.add(f[i] == model.coordinate_piecewise_linear(l[i], -0.0125, [0, 400, 1800, 2400, 5000000], [100, 95, 10, 1, 1], 0))

    ## activation
    for j in range(num_allocation):
        cond = [(x[(i, c)] == j) for i in range(num_residents) for c in range(tot_choices)]
        cond2 = [(y[(k_,1)] == j) for k_ in range(k_array[1])]
        model.add(model.if_then(model.any(cond),model.any(cond2)))

    ## node capacity
    for j in range(num_allocation):
        model.add(model.count(list(y.values()), j) <= capacity[j])

    # choices can not be the same place
    # newly allocated
    for i in range(num_residents):
        for j in range(num_allocation):
            model.add(model.count([x[(i, c)] for c in range(tot_choices)], j) <= model.count([y[(k_, 1)] for k_ in range(k_array[1])], j))
        for m in range_restaurant_dest_list:
            model.add(model.count([x[(i, c)] for c in range(tot_choices)], m) <= 1)

    # symmetry breaking
    for a in range(len(k_array)):
        if k_array[a] > 1:
            for k_ in range((k_array[a])-1):
                model.add(y[(k_,a)]<=y[(k_+1,a)])
    if tot_choices > 1:
        for c in range(tot_choices - 1):
            for i in range(num_residents):
                model.add(dist_r[(i, c)] <= dist_r[(i, c+1)])

    # # objective
    model.add(model.maximize(model.sum(f[i] for i in range(num_residents)) / num_residents))

    msol = model.solve(execfile=solver_path, TimeLimit=time_limit, Workers=threads)
    obj_value = msol.get_objective_values()[0][0]

    log_str = msol.solver_log
    with open(results_sava_path, 'w') as f:
        f.write(log_str)

    ###################
    allocations = [(msol[y[k_,a]], a) for (k_, a) in y.keys() if msol[y[k_,a]] < num_allocation]

    # save allocation solutions
    allocate_var_id_grocery = [(j, a) for (j, a) in allocations if a == 0]
    allocate_row_id_grocery = []
    allocate_node_id_grocery = []
    for (j, a) in allocate_var_id_grocery:
        allocate_row_id_grocery.append(group_values_to[j][0])
        allocate_node_id_grocery.append(df_to.iloc[group_values_to[j][0]]["node_ids"])

    allocate_var_id_restaurant = [(j, a) for (j, a) in allocations if a == 1]
    allocate_row_id_restaurant = []
    allocate_node_id_restaurant = []
    for (j, a) in allocate_var_id_restaurant:
        allocate_row_id_restaurant.append(group_values_to[j][0])
        allocate_node_id_restaurant.append(df_to.iloc[group_values_to[j][0]]["node_ids"])

    allocate_var_id_school = [(j, a) for (j, a) in allocations if a == 2]
    allocate_row_id_school = []
    allocate_node_id_school = []
    for (j, a) in allocate_var_id_school:
        allocate_row_id_school.append(group_values_to[j][0])
        allocate_node_id_school.append(df_to.iloc[group_values_to[j][0]]["node_ids"])

    allocated_D = {
        "allocate_var_id_grocery": allocate_var_id_grocery,
        "allocate_node_id_grocery": allocate_node_id_grocery,
        "allocate_row_id_grocery": allocate_row_id_grocery,
        "allocate_var_id_restaurant": allocate_var_id_restaurant,
        "allocate_node_id_restaurant": allocate_node_id_restaurant,
        "allocate_row_id_restaurant": allocate_row_id_restaurant,
        "allocate_var_id_school": allocate_var_id_school,
        "allocate_row_id_school": allocate_row_id_school,
        "allocate_node_id_school": allocate_node_id_school
    }

    # assignments

    assigned_D = {}
    k_strs = ["grocery", "restaurant", "school"]
    all_dfs = [grocery_df, restaurant_df, school_df]

    for a in [0,2]:
        assign_from_var_id = list(range(num_residents))
        assigned_D["assign_from_var_id_" + k_strs[a]] = assign_from_var_id
        assign_from_node_id = [df_from.iloc[group_values_from[i][0]]["node_ids"] for i in assign_from_var_id]
        assign_to_var_id = []
        assign_to_node_id = []
        assign_type = []
        dist = []

        for i in range(num_residents):

            # new_allocated
            j_array = [msol[y[(k_, a)]] for k_ in range(k_array[a])]
            dist_array_new = [d[i, msol[y[(k_, a)]]] for k_ in range(k_array[a])]
            j_new = j_array[np.argmin(dist_array_new)]
            # existing
            if len(existing_list[a]) > 0:
                j_array = [j - 1 for j in existing_list[a]]
                dist_array_e = [d[(i, j)] for j in existing_list[a]]
                j_exist = j_array[np.argmin(dist_array_e)]
                if min(dist_array_new) < min(dist_array_e):
                    assign_to_var_id.append(j_new)
                    assign_to_node_id.append(df_to.iloc[group_values_to[j_new][0]]["node_ids"])
                    assign_type.append('allocated')
                    dist.append(SP_matrix[df_from.iloc[group_values_from[i][0]]["node_ids"],
                                          df_to.iloc[group_values_to[j_new][0]]["node_ids"]])
                else:
                    assign_to_var_id.append(j_exist)
                    if a == 0:
                        ind = j_exist - num_allocation
                    elif a == 1:
                        ind = j_exist - num_allocation - num_cur_grocery
                    else:
                        ind = j_exist - num_allocation - num_cur_grocery - num_cur_restaurant
                    assign_to_node_id.append(all_dfs[a].iloc[ind]["node_ids"])
                    assign_type.append('existing')
                    dist.append(
                        SP_matrix[df_from.iloc[group_values_from[i][0]]["node_ids"], all_dfs[a].iloc[ind]["node_ids"]])
            else:
                assign_to_var_id.append(j_new)
                assign_to_node_id.append(df_to.iloc[group_values_to[j_new][0]]["node_ids"])
                assign_type.append('allocated')
                dist.append(SP_matrix[df_from.iloc[group_values_from[i][0]]["node_ids"],
                                      df_to.iloc[group_values_to[j_new][0]]["node_ids"]])
        assigned_D["assign_from_var_id_" + k_strs[a]] = assign_from_var_id
        assigned_D["assign_to_var_id_" + k_strs[a]] = assign_to_var_id
        assigned_D["assign_from_node_id_" + k_strs[a]] = assign_from_node_id
        assigned_D["assign_to_node_id_" + k_strs[a]] = assign_to_node_id
        assigned_D["assign_type_" + k_strs[a]] = assign_type
        assigned_D["dist_" + k_strs[a]] = dist

    # restaurant

    for choice in range(tot_choices):

        assign_from_var_id = list(range(num_residents))
        assign_from_node_id = [df_from.iloc[group_values_from[i][0]]["node_ids"] for i in assign_from_var_id]
        assign_to_var_id = []
        assign_to_node_id = []
        assign_type = []
        dist = []

        for i in range(num_residents):

            # new_allocated
            j = msol[x[(i, choice)]]
            if j < num_allocation:
                assign_to_var_id.append(j)
                assign_to_node_id.append(df_to.iloc[group_values_to[j][0]]["node_ids"])
                assign_type.append('allocated')
                dist.append(SP_matrix[df_from.iloc[group_values_from[i][0]]["node_ids"],
                                      df_to.iloc[group_values_to[j][0]]["node_ids"]])
            else:
                assign_to_var_id.append(j - 1)
                assign_to_node_id.append(amenity_df.iloc[j - 1 - num_allocation]["node_ids"])
                assign_type.append('existing')
                dist.append(SP_matrix[df_from.iloc[group_values_from[i][0]]["node_ids"],
                                      amenity_df.iloc[j - 1 - num_allocation]["node_ids"]])

        assigned_D[str(choice) + "_assign_from_var_id_restaurant"] = assign_from_var_id
        assigned_D[str(choice) + "_assign_to_var_id_restaurant"] = assign_to_var_id
        assigned_D[str(choice) + "_assign_from_node_id_restaurant"] = assign_from_node_id
        assigned_D[str(choice) + "_assign_to_node_id_restaurant"] = assign_to_node_id
        assigned_D[str(choice) + "_assign_type_restaurant"] = assign_type
        assigned_D[str(choice) + "_dist_restaurant"] = dist

    restaurant_dist_obj = [np.mean(assigned_D[str(c) + "_dist_restaurant"]) if (str(c) + "_dist_restaurant") in assigned_D.keys() else 0 for c in range(len(choice_weights))]

    return obj_value, [np.mean(assigned_D["dist_grocery"]), restaurant_dist_obj, np.mean(assigned_D["dist_school"])], msol.get_solve_time(), msol, allocated_D, assigned_D, num_residents, num_allocation, [num_cur_grocery, num_cur_restaurant, num_cur_school], msol.solve_status



