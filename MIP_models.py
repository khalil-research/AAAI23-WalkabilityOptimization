import gurobipy as gp
from gurobipy import GRB
from itertools import product
import pandas as pd
import numpy as np
import copy
from map_utils import map_back_allocate, map_back_assign
import matplotlib.pyplot as plt
#import time
L_a=[0,400,1800,2400,5000000]
L_f_a=[100,95,10,0,0]

k = [1, 2, 2, 3, 1]
#k = [1, 1, 1, 1, 1]
w=[0.3,0.2,0.2,0.2,0.1]

#time_limit=9*60*60
time_limit=60*60
#time_limit=10

def MILP_comp(df_from,df_to,SP_matrix,solver_path, EPS=1.e-6):
    if len(df_from)>0:
        df_from = df_from[['geometry', 'node_ids']]
    if len(df_to)>0:
        df_to = df_to[['geometry', 'node_ids']]

    model = gp.Model('MILP comparison')

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
    y_list = []
    for j in range(num_allocation):
        for a in range(len(k)):
            y_list.append((j, a))
            #y[(j, a)] = model.integer_var(min=0, max=k[a],name=f'y[{j},{a}]')
    x_list = []
    for i in range(num_residents):
        for j in range(num_allocation):
            for a in range(len(k)):
                x_list.append((i, j, a))
                #x[(i, j, a)] = model.binary_var(name=f'x[{i},{j},{a}]')
    f_list = []
    for i in range(num_residents):
        #f[i] = model.float_var(min=0, max=100, name=f'f[{i}]')
        f_list.append(i)
    l_list = []
    for i in range(num_residents):
        #l[i] = model.float_var(min=0, max=L_a[-1], name=f'z[{i}]')
        l_list.append(i)
    h_list = []
    h0_list = []
    h2_list = []
    for j in range(num_allocation):
        # l[i] = model.float_var(min=0, max=L_a[-1], name=f'z[{i}]')
        h_list.append(j)
        h0_list.append(j)
        h2_list.append(j)


    y = model.addVars(y_list, vtype=GRB.BINARY, name='Allocate')
    x = model.addVars(x_list, vtype=GRB.BINARY, name='Assign')
    f = model.addVars(f_list, vtype=GRB.CONTINUOUS, name='score')
    l = model.addVars(l_list, vtype=GRB.CONTINUOUS, name='dist')
    h = model.addVars(h_list, vtype=GRB.BINARY, name='helper')


    # Constraints
    # activation
    for i in range(num_residents):
        for j in range(num_allocation):
            for a in range(len(k)):
                model.addConstr(x[(i,j,a)]<=y[(j,a)])
                #model.add(x[(i,j,a)]<=y[(j,a)])
    # amenity upper bound
    for a in range(len(k)):
        model.addConstr((gp.quicksum(y[(j, a)] for j in range(num_allocation))) <= k[a])
        #model.add((model.sum(y[(j, a)] for j in range(num_allocation))) <= k[a])
    # capacity upper bound
    for j in range(num_allocation):
        model.addConstr((gp.quicksum(y[(j,a)] for a in range(len(k)))) <= capacity[j])
        #model.add((model.sum(y[(j,a)] for a in range(len(k)))) <= capacity[j])
    # each resident visit one of each amenity
    for i in range(num_residents):
        for a in range(len(k)):
            model.addConstr(gp.quicksum(x[(i,j,a)] for j in range(num_allocation)) == 1)

    # side
    # side 1
    for j in range(num_allocation):
        #model.add(model.if_then(model.logical_or((y[(j,0)]>=1),(y[(j,2)]>=1)), y[(j,3)]>=1))
        # model.addConstr((y[(j, 0)]==1) >> (y[(j, 3)]==1))
        # model.addConstr((y[(j, 2)] == 1) >> (y[(j, 3)] == 1))

        model.addConstr(y[(j, 0)] <= y[(j, 3)])
        model.addConstr(y[(j, 2)] <= y[(j, 3)])

        # side 2
        #model.add(model.if_then(model.logical_and((y[(j, 0)] >= 1), (y[(j, 2)] >= 1)), y[(j, 1)] >= 1))
        model.addConstr((y[(j, 0)]+y[(j, 2)]-1)<=h[j])
        model.addConstr(h[j] <= y[(j, 1)])

    # calculate dist
    for i in range(num_residents):
        model.addConstr(l[i]==((gp.quicksum(w[a]*gp.quicksum(x[(i,j,a)]*d[(i,j)] for j in range(num_allocation)) for a in range(len(k))))))
    # PWL
    for i in range(num_residents):
    #     model.add(model.if_then(((l[i] >= 0) & (l[i] < 400)), f[i] == model.round(slope[0] * l[i] + intercept[0])))
    #     model.add(model.if_then(((l[i] >= 400) & (l[i] < 1800)), f[i] == model.round(slope[1] * l[i] + intercept[1])))
    #     model.add(model.if_then(((l[i] >= 1800) & (l[i] < 2400)), f[i] == model.round(slope[2] * l[i] + intercept[2])))
    #     model.add(model.if_then(l[i] >= 2400, f[i] == 0))
        #model.add(model.slope_piecewise_linear(l[i], [400,1800,2400], [-0.0125, -0.0607, -0.0167,0], 0, 100)==f[i])
        model.addGenConstrPWL(l[i], f[i], L_a, L_f_a)


    # objective
    model.setObjective((gp.quicksum(f[i] for i in range(num_residents))/num_residents), GRB.MAXIMIZE)

    # solving
    #model.set_search_phases([search_phase(y.values()), search_phase(x.values()),search_phase(l.values())])
    #model.set_search_phases([search_phase(y.values()), search_phase(x.values())])
    #msol = model.solve(execfile=solver_path,TimeLimit=time_limit) #13h
    #msol = model.solve(execfile=solver_path, TimeLimit=20)
    model.Params.TimeLimit = time_limit
    model.optimize()
    obj = model.getObjective()
    obj_value = obj.getValue()
    #obj_value = msol.get_objective_values()[0]

    # save solution for debugging and visualization

    assignments = [(i, j, a) for (i, j, a) in x.keys() if (x[(i, j, a)].x > EPS)]
    allocations = [(j, a) for (j, a) in y.keys() if (y[(j, a)].x) > EPS]



    # map back to df
    # allocated_nodes = [df_to.iloc[j]["node_ids"] for j in allocations]
    # allocated_df = df_to.iloc[allocations]
    # allocated_nodes=[j for (j,a) in allocations]

    allocated_nodes = [df_to.iloc[group_values_to[j][0]]["node_ids"] for (j,a) in allocations]
    allocated_df = df_to.iloc[[group_values_to[j][0] for (j,a) in allocations]]

    str = "grb"

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


    return obj_value, model.Runtime, allocated_nodes, allocated_df, assigned_nodes, model, str, num_residents, num_allocation


def k_center(df_from,df_to,SP_matrix,k,EPS=1.e-6):
    m = gp.Model('k-center')

    # data
    num_residents=len(df_from)
    num_allocation=len(df_to)
    cartesian_prod = list(product(range(num_residents), range(num_allocation)))  # a list of tuples
    # retrieve distances
    distances = {(i, j): SP_matrix[df_from.iloc[i]["node_ids"], df_to.iloc[j]["node_ids"]] for i, j in cartesian_prod}

    allocate = m.addVars(num_allocation, vtype=GRB.BINARY, name='Allocate')
    assign = m.addVars(cartesian_prod, vtype=GRB.BINARY, name='Assign')

    # activation
    m.addConstrs((assign[(i, j)] <= allocate[j] for i, j in cartesian_prod), name='setup')
    # demand
    m.addConstrs((gp.quicksum(assign[(i, j)] for j in range(num_allocation)) == 1 for i in range(num_residents)),
                 name='Demand')
    # resource contraint
    m.addConstr((gp.quicksum(allocate[j] for j in range(num_allocation))) <= k, name='resource')

    # objective
    m.setObjective(assign.prod(distances), GRB.MINIMIZE)

    #start_time = time.time()
    m.optimize()
    #solving_time = time.time() - start_time

    assignments=[(i, j) for (i, j) in assign.keys() if (assign[i, j].x > EPS)]
    resources = [j for j in allocate.keys() if (allocate[j].x > EPS)]

    # map back to df
    allocated_nodes = [df_to.iloc[j]["node_ids"] for j in resources]
    allocated_df = df_to.iloc[resources]

    obj = m.getObjective()
    obj_value = obj.getValue()

    return obj_value, m.Runtime, allocated_nodes, allocated_df

# add in existing amenities
def k_center2(df_from,df_to,supermarkets_df, SP_matrix,k,EPS=1.e-6):
    if len(df_from)>0:
        df_from = df_from[['geometry', 'node_ids']]
    if len(df_to)>0:
        df_to = df_to[['geometry', 'node_ids']]
    if len(supermarkets_df)>0:
        supermarkets_df = supermarkets_df[['geometry', 'node_ids']]
        df_to_2 = pd.concat([df_to, supermarkets_df])
    else:
        df_to_2 = df_to


    m = gp.Model('k-center2')

    # data
    num_residents=len(df_from)
    num_allocation=len(df_to)
    num_cur=len(supermarkets_df)
    cartesian_prod = list(product(range(num_residents), range(num_allocation+num_cur)))  # a list of tuples
    cartesian_prod_activate = list(product(range(num_residents), range(num_allocation)))
    #z_list=np.zeros(len(df_to_2))
    #z_list[len(df_to):]=1
    # retrieve distances
    distances = {(i, j): SP_matrix[df_from.iloc[i]["node_ids"], df_to_2.iloc[j]["node_ids"]] for i, j in cartesian_prod}

    allocate = m.addVars(num_allocation, vtype=GRB.BINARY, name='Allocate')
    assign = m.addVars(cartesian_prod, vtype=GRB.BINARY, name='Assign')

    # activation
    m.addConstrs((assign[(i, j)] <= allocate[j] for i, j in cartesian_prod_activate), name='setup')
    # demand
    m.addConstrs((gp.quicksum(assign[(i, j)] for j in range(num_allocation+num_cur)) == 1 for i in range(num_residents)),
                 name='Demand')
    # resource constraint
    m.addConstr((gp.quicksum(allocate[j] for j in range(num_allocation))) <= k, name='resource')

    # objective
    m.setObjective(assign.prod(distances), GRB.MINIMIZE)

    #start_time = time.time()
    m.optimize()
    #solving_time = time.time() - start_time

    assignments=[(i, j) for (i, j) in assign.keys() if (assign[i, j].x > EPS)]
    resources = [j for j in allocate.keys() if (allocate[j].x > EPS)]

    # map back to df
    allocated_nodes = [df_to.iloc[j]["node_ids"] for j in resources]
    allocated_df = df_to.iloc[resources]

    obj = m.getObjective()
    obj_value = obj.getValue()

    return obj_value, m.Runtime, allocated_nodes, allocated_df


def max_walk_score_scratch_1(df_from,df_to,supermarkets_df, SP_matrix,k,EPS=1.e-6):
    if len(df_from)>0:
        df_from = df_from[['geometry', 'node_ids']]
    if len(df_to)>0:
        df_to = df_to[['geometry', 'node_ids']]
    if len(supermarkets_df)>0:
        supermarkets_df = supermarkets_df[['geometry', 'node_ids']]
        df_to_2 = pd.concat([df_to, supermarkets_df])
    else:
        df_to_2 = df_to

    m = gp.Model('max_walk_score_scratch')

    # data
    num_residents = len(df_from)
    num_allocation = len(df_to)
    num_cur = len(supermarkets_df)
    cartesian_prod_assign = list(product(range(num_residents), range(num_allocation + num_cur)))  # a list of tuples
    cartesian_prod_activate = list(product(range(num_residents), range(num_allocation)))
    # piecewise linear function
    num_a = len(L_a)
    cartesian_prod_lambda = list(product(range(num_residents),range(num_a)))
    cartesian_prod_z = list(product(range(num_residents),range(num_a-1)))
    # retrieve distances
    d = {(i, j): SP_matrix[df_from.iloc[i]["node_ids"], df_to_2.iloc[j]["node_ids"]] for i, j in cartesian_prod_assign}

    # Variables
    x = m.addVars(cartesian_prod_assign, vtype=GRB.BINARY, name='x')
    y = m.addVars(num_allocation, vtype=GRB.BINARY, name='y')
    z = m.addVars(cartesian_prod_z, vtype=GRB.BINARY, name='z')
    coeff = m.addVars(cartesian_prod_lambda, vtype=GRB.CONTINUOUS, name='lambda')

    # Constraints
    ## WalkScore
    m.addConstrs((gp.quicksum(coeff[(n,i)]*L_a[i] for i in range(num_a)))==
                 (gp.quicksum(d[(n,m)]*x[(n,m)] for m in range(num_allocation + num_cur))) for n in range(num_residents))
    m.addConstrs(gp.quicksum(coeff[(n,i)] for i in range(num_a))==1 for n in range(num_residents))
    m.addConstrs(coeff[(n,i)]>=0 for i in range(num_a) for n in range(num_residents))
    m.addConstrs((gp.quicksum(z[(n,i)] for i in range(num_a-1))==1) for n in range(num_residents))
    m.addConstrs(coeff[(n,0)] <= z[(n,0)] for n in range(num_residents))
    m.addConstrs(coeff[(n,1)] <= (z[(n,0)]+z[(n,1)]) for n in range(num_residents))
    m.addConstrs(coeff[(n,2)] <= (z[(n,1)]+z[(n,2)]) for n in range(num_residents))
    m.addConstrs(coeff[(n,3)] <= (z[(n,2)] + z[(n,3)]) for n in range(num_residents))
    m.addConstrs(coeff[(n,4)] <= z[(n, 3)] for n in range(num_residents))

    ## assgined nodes satisfy demand
    m.addConstrs(
        (gp.quicksum(x[(n, m)] for m in range(num_allocation + num_cur)) == 1 for n in range(num_residents)),
        name='Demand')
    ## resource constraint
    m.addConstr((gp.quicksum(y[m] for m in range(num_allocation))) <= k, name='resource')
    ## activation
    m.addConstrs((x[(n, m)] <= y[m] for n, m in cartesian_prod_activate), name='setup')

    # objective
    m.setObjective(gp.quicksum(gp.quicksum(coeff[(n,i)]*L_f_a[i] for i in range(num_a))
                               for n in range(num_residents))/num_residents,
                   GRB.MAXIMIZE)
    m.optimize()

    assignments = [(i, j) for (i, j) in x.keys() if (x[i, j].x > EPS)]
    allocations = [j for j in y.keys() if (y[j].x > EPS)]

    # map back to df
    #allocated_nodes = [df_to.iloc[j]["node_ids"] for j in allocations]
    #allocated_df = df_to.iloc[allocations]

    allocated_nodes,allocated_df = map_back_allocate(allocations, df_to)
    assigned_nodes = map_back_assign(assignments, df_from, df_to_2, d)

    obj = m.getObjective()
    obj_value = obj.getValue()

    return obj_value, m.Runtime, allocated_nodes, allocated_df, assigned_nodes, m

def max_walk_score(df_from,df_to,supermarkets_df, SP_matrix,k,EPS=1.e-6):
    if len(df_from)>0:
        df_from = df_from[['geometry', 'node_ids']]
    if len(df_to)>0:
        df_to = df_to[['geometry', 'node_ids']]
    if len(supermarkets_df)>0:
        supermarkets_df = supermarkets_df[['geometry', 'node_ids']]
        df_to_2 = pd.concat([df_to, supermarkets_df])
    else:
        df_to_2 = df_to

    m = gp.Model('max_walk_score')

    # data
    num_residents = len(df_from)
    num_allocation = len(df_to)
    num_cur = len(supermarkets_df)
    cartesian_prod_assign = list(product(range(num_residents), range(num_allocation + num_cur)))  # a list of tuples
    cartesian_prod_activate = list(product(range(num_residents), range(num_allocation)))

    # retrieve distances
    d = {(i, j): SP_matrix[df_from.iloc[i]["node_ids"], df_to_2.iloc[j]["node_ids"]] for i, j in cartesian_prod_assign}

    # Variables
    x = m.addVars(cartesian_prod_assign, vtype=GRB.BINARY, name='assign')
    y = m.addVars(num_allocation, vtype=GRB.BINARY, name='activate')
    a = m.addVars(num_residents, vtype=GRB.CONTINUOUS, name='dist')
    f = m.addVars(num_residents, vtype=GRB.CONTINUOUS, ub=100,name='score')

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
    m.addConstr((gp.quicksum(y[m] for m in range(num_allocation))) <= k, name='resource')
    ## activation
    m.addConstrs((x[(n, m)] <= y[m] for n, m in cartesian_prod_activate), name='setup')

    # objective
    m.setObjective(gp.quicksum(f[n] for n in range(num_residents))/num_residents, GRB.MAXIMIZE)
    m.optimize()

    assignments = [(i, j) for (i, j) in x.keys() if (x[i, j].x > EPS)]
    allocations = [j for j in y.keys() if (y[j].x > EPS)]

    # map back to df
    # allocated_nodes = [df_to.iloc[j]["node_ids"] for j in allocations]
    # allocated_df = df_to.iloc[allocations]
    allocated_nodes, allocated_df = map_back_allocate(allocations, df_to)
    assigned_nodes = map_back_assign(assignments, df_from, df_to_2, d)

    obj = m.getObjective()
    obj_value = obj.getValue()

    return obj_value, m.Runtime, allocated_nodes, allocated_df, assigned_nodes, m

def cur_assignments(df_from,supermarkets_df, SP_matrix,EPS=1.e-6):
    if len(supermarkets_df) == 0:
        print("no existing amenities!")
        return

    m = gp.Model('cur_assignment')
    num_from = len(df_from)
    num_amenity = len(supermarkets_df)
    cartesian_prod = list(product(range(num_from), range(num_amenity)))  # a list of tuples
    distances = {(i, j): SP_matrix[df_from.iloc[i]["node_ids"], supermarkets_df.iloc[j]["node_ids"]] for i, j in cartesian_prod}
    assign = m.addVars(cartesian_prod, vtype=GRB.BINARY, name='Assign')
    # demand
    m.addConstrs((gp.quicksum(assign[(i, j)] for j in range(num_amenity)) == 1 for i in range(num_from)),
                 name='Demand')
    # objective
    m.setObjective(assign.prod(distances), GRB.MINIMIZE)
    m.optimize()
    assignments = [(i, j) for (i, j) in assign.keys() if (assign[i, j].x > EPS)]

    obj = m.getObjective()
    obj_value = obj.getValue() # min total dist

    L_d=[distances[(i,j)] for (i,j) in assignments]
    assigned_nodes = map_back_assign(assignments, df_from, supermarkets_df, distances)
    # print("check assignments")
    # print("len L_d:",len(L_d))
    # print("num residents:", len(df_from))

    return assignments, np.array(L_d), obj_value, m.Runtime, assigned_nodes


def dist_to_score__(d):
    L_m = []
    L_c = []
    for i in range(len(L_a) - 1):
        x = np.array(L_a[i:i + 2])
        y = np.array(L_f_a[i:i + 2])
        A = np.vstack([x, np.ones(len(x))]).T
        m, c = np.linalg.lstsq(A, y, rcond=None)[0]
        L_m.append(m)
        L_c.append(c)
    scores = np.piecewise(d, [d<L_a[1], (L_a[1]<=d) & (d<L_a[2]), (L_a[2]<=d) & (d<L_a[3]), d>=L_a[3]],
                 [lambda d: L_m[0]*d+L_c[0], lambda d: L_m[1]*d+L_c[1], lambda d: L_m[2]*d+L_c[2], lambda d:0])

    return scores

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


if __name__ == "__main__":
    x=1




