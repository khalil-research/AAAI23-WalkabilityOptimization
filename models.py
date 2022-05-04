import gurobipy as gp
from gurobipy import GRB
from itertools import product
import pandas as pd
import numpy as np
import copy
import matplotlib.pyplot as plt
#import time
L_a=[0,400,1800,2400,5000000]
L_f_a=[100,95,10,0,0]


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


def max_walk_score_scratch(df_from,df_to,supermarkets_df, SP_matrix,k,EPS=1.e-6):
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
    allocated_nodes = [df_to.iloc[j]["node_ids"] for j in allocations]
    allocated_df = df_to.iloc[allocations]

    obj = m.getObjective()
    obj_value = obj.getValue()

    return obj_value, m.Runtime, allocated_nodes, allocated_df

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
    allocated_nodes = [df_to.iloc[j]["node_ids"] for j in allocations]
    allocated_df = df_to.iloc[allocations]

    obj = m.getObjective()
    obj_value = obj.getValue()

    return obj_value, m.Runtime, allocated_nodes, allocated_df

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
    # print("check assignments")
    # print("len L_d:",len(L_d))
    # print("num residents:", len(df_from))

    return assignments, np.array(L_d), obj_value, m.Runtime


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




