import numpy as np
from model_latest import dist_to_score, L_a, L_f_a, opt_multiple_depth, weights_array_multi, choice_weights
import time
from itertools import product
import copy


def greedy_multiple_depth(df_from,df_to,grocery_df, restaurant_df, school_df, SP_matrix, k_array):
    '''multiple amenity case, with depth of choice'''

    if len(df_from)>0:
        df_from = df_from[['geometry', 'node_ids']]
    if len(df_to)>0:
        df_to = df_to[['geometry', 'node_ids']]

    # grouping
    groups_to=df_to.groupby('node_ids').groups # keys are node id, values are indices
    group_values_to=list(groups_to.values())
    num_allocation = len(group_values_to)

    # initial capacity
    capacity_init = [len(item) for item in group_values_to]

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


    # retrieve distances
    d = {(i, j): SP_matrix[df_from.iloc[group_values_from[i][0]]["node_ids"], df_to.iloc[group_values_to[j][0]]["node_ids"]] for i, j in list(product(range(num_residents), range(num_allocation)))}

    for i in range(num_residents):
        start_id = num_allocation
        for amenity_df in [grocery_df, restaurant_df, school_df]:
            for inst_row in range(len(amenity_df)):
                cur_id = start_id + inst_row
                d[(i, cur_id)] = SP_matrix[df_from.iloc[group_values_from[i][0]]["node_ids"], amenity_df.iloc[inst_row]["node_ids"]]
            start_id += len(amenity_df)

    capacity = copy.deepcopy(capacity_init)

    st = time.time()

    # current score
    # resident and cur amneity matrix
    mat_grocery = np.array([[d[(i, j)] for j in range_grocery_existing] for i in range(num_residents)])
    mat_res = np.array([[d[(i, j)] for j in range_restaurant_existing] for i in range(num_residents)])
    ind = np.argsort(mat_res, axis=1)
    d_res = np.take_along_axis(mat_res, ind, axis=1)
    if d_res.shape[1] < 10:
        # pad
        d_res = np.pad(d_res, ((0, 0), (0, 10-d_res.shape[1])), constant_values=L_a[-2])
    if d_res.shape[1] > 10:
        d_res=d_res[:,:10]
    mat_school = np.array([[d[(i, j)] for j in range_school_existing] for i in range(num_residents)])
    d_grocery = np.amin(mat_grocery, axis=1)
    d_school = np.amin(mat_school, axis=1)

    multiple_dist = np.concatenate((d_grocery.reshape((1, num_residents)), d_res.T, d_school.reshape((1, num_residents))), axis=0)
    weighted_dist = np.dot(np.array(weights_array_multi), multiple_dist)
    scores = dist_to_score(np.array(weighted_dist), L_a, L_f_a)
    score_obj = np.mean(scores)

    L_amenity = ['grocery','restaurant','school']

    # while loop
    allocated_grocery, allocated_res, allocated_school = [], [], []
    prev_score = score_obj

    while (((len(allocated_grocery) < k_array[0]) or (len(allocated_res) < k_array[1]) or (len(allocated_school) < k_array[2])) and (max(capacity)>0)):
        new_obj = np.zeros((len(k_array),num_allocation))

        for m in range(num_allocation):
            if capacity[m]>0:
                allocated = [allocated_grocery, allocated_res, allocated_school]
                for a in range(len(L_amenity)):
                    if (len(allocated[a]) < k_array[a]):
                        L_grocery = range_grocery_existing + allocated_grocery
                        L_res = range_restaurant_existing + allocated_res
                        L_school = range_school_existing + allocated_school
                        if a ==0:
                            L_grocery = L_grocery + [m]
                        if a==1:
                            L_res = L_res + [m]
                        if a==2:
                            L_school = L_school + [m]

                        mat_grocery = np.array([[d[(i, j)] for j in L_grocery] for i in range(num_residents)])
                        mat_res = np.array([[d[(i, j)] for j in L_res] for i in range(num_residents)])
                        ind = np.argsort(mat_res, axis=1)
                        d_res = np.take_along_axis(mat_res, ind, axis=1)
                        if d_res.shape[1] < 10:
                            # pad
                            d_res = np.pad(d_res, ((0, 0), (0, 10 - d_res.shape[1])), constant_values=L_a[-2])
                        if d_res.shape[1] > 10:
                            d_res = d_res[:, :10]
                        mat_school = np.array([[d[(i, j)] for j in L_school] for i in range(num_residents)])
                        d_grocery = np.amin(mat_grocery, axis=1)
                        d_school = np.amin(mat_school, axis=1)

                        multiple_dist = np.concatenate((d_grocery.reshape((1, num_residents)), d_res.T, d_school.reshape((1, num_residents))), axis=0)
                        weighted_dist = np.dot(np.array(weights_array_multi), multiple_dist)
                        scores = dist_to_score(np.array(weighted_dist), L_a, L_f_a)
                        score_obj = np.mean(scores)
                        new_obj[a,m] = score_obj - prev_score

        delta_mat = new_obj - prev_score

        type_id, loc_id = np.unravel_index(np.argmax(delta_mat, axis=None), delta_mat.shape)
        if type_id==0:
            allocated_grocery.append(loc_id)
        if type_id==1:
            allocated_res.append(loc_id)
        if type_id==2:
            allocated_school.append(loc_id)

        capacity[loc_id] = capacity[loc_id] - 1

        prev_score = np.max(score_obj)
        print(str(L_amenity[type_id]),"allocated to",str(loc_id))
        print("current obj: ",prev_score)

    et = time.time()

    # save allocation solutions
    # grocery
    allocate_var_id_grocery = []
    allocate_row_id_grocery = []
    allocate_node_id_grocery = []
    for j in allocated_grocery:
        allocate_var_id_grocery.append(j)
        allocate_row_id_grocery.append(group_values_to[j][0])
        allocate_node_id_grocery.append(df_to.iloc[group_values_to[j][0]]["node_ids"])

    # restaurant
    allocate_var_id_restaurant = []
    allocate_row_id_restaurant = []
    allocate_node_id_restaurant = []
    for j in allocated_res:
        allocate_var_id_restaurant.append(j)
        allocate_row_id_restaurant.append(group_values_to[j][0])
        allocate_node_id_restaurant.append(df_to.iloc[group_values_to[j][0]]["node_ids"])

    # school
    allocate_var_id_school = []
    allocate_row_id_school = []
    allocate_node_id_school = []
    for j in allocated_school:
        allocate_var_id_school.append(j)
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
    # retrieve final distances
    mat_grocery = np.array([[d[(i, j)] for j in (range_grocery_existing + allocated_grocery)] for i in range(num_residents)])
    mat_res = np.array([[d[(i, j)] for j in (range_restaurant_existing + allocated_res)] for i in range(num_residents)])
    ind = np.argsort(mat_res, axis=1)
    d_res = np.take_along_axis(mat_res, ind, axis=1)
    if d_res.shape[1] < 10:
        # pad
        d_res = np.pad(d_res, ((0, 0), (0, 10 - d_res.shape[1])), constant_values=L_a[-2])
    if d_res.shape[1] > 10:
        d_res=d_res[:,:10]

    mat_school = np.array([[d[(i, j)] for j in (range_school_existing + allocated_school)] for i in range(num_residents)])
    d_grocery = np.amin(mat_grocery, axis=1)
    d_school = np.amin(mat_school, axis=1)

    return score_obj, [np.mean(d_grocery), list(np.mean(d_res,axis=0)), np.mean(d_school)], (et - st), None, allocated_D, None, num_residents, num_allocation, [num_cur_grocery, num_cur_restaurant, num_cur_school], None