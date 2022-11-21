#!/usr/bin/env python
# coding: utf-8

# In[1]:


from decimal import *
import numpy as np
from math import prod
import itertools
from sympy.utilities.iterables import multiset_permutations


# In[2]:


def get_rounded_val(val, decimal_pos = 4):
    return Decimal(val).quantize(Decimal(10) ** -decimal_pos, rounding=ROUND_HALF_EVEN)

def get_delta_thres(thres):
    delta_thres = np.diff(thres)
    delta_thres = np.insert(delta_thres, 0, thres[0])
    return delta_thres

## Prints the assignment of all activated tioT0 and tiiTJ variables which corresponds to the resulting join order
def get_join_tree_leaves(sample, num_tables):
    
    num_tiio_variables = num_tables * (num_tables-1)
    
    ## Extract variable assignments from the response
    
    inner_operand_for_join = {}
    # Fetch inner operands
    for i in range(num_tiio_variables, num_tiio_variables*2):
        #key = 'x[' + str(i) + ']'
        key = i
        if sample[key] == 1:
            table_index = (i // (num_tables-1)) - num_tables
            join_index = i % (num_tables-1)
            if join_index in inner_operand_for_join:
                # Erroneous solution, since another table has already been selected for this join
                return None
            inner_operand_for_join[join_index] = table_index
      
    outer_table_index = None
    for i in range(num_tables):
        if i not in inner_operand_for_join.values():
            outer_table_index = i
            break # Outer operand found -> abort search
    
    join_order = []
    join_order.append(outer_table_index)
    for i in range(num_tables-1):
        if i in inner_operand_for_join:
            join_order.append(inner_operand_for_join[i])
            
    if len(join_order) != len(set(join_order)):
        # Invalid solution: At least one relation has been selected for multiple leaves of the join tree
        return None
    
    if len(join_order) != num_tables:
        # Invalid solution: For at least one leaf of the join tree, no relation has been selected
        return None
    
    return join_order

def get_combined_pred_sel(join, pred, pred_sel, join_order):
    relations = np.array(join_order[0:join+1])
    pred_indices = []
    for comb in itertools.combinations(relations, 2):
        comb = tuple(sorted(comb))
        if comb in pred:
            ind = pred.index(comb)
            pred_indices.append(ind)
    combined_sel = prod(pred_sel[i] for i in pred_indices)
    return combined_sel

def calculate_intermediate_cardinality_for_join(join, card, pred, pred_sel, join_order):
    raw_cardinality = prod(card[join_order[i]] for i in range(join+1))
    combined_sel = get_combined_pred_sel(join, pred, pred_sel, join_order)
    return raw_cardinality * combined_sel

def get_actual_costs_for_sample(join_order, card, pred, pred_sel, thres):
    delta_thres = get_delta_thres(thres)
    total_costs = np.int32(0)
    for j in range(1, len(card)-1): # Ignore first join
        int_card = calculate_intermediate_cardinality_for_join(j, card, pred, pred_sel, join_order)
        dec_int_card = get_rounded_val(int_card)
        for i in range(len(thres)):
            if dec_int_card > thres[i]:
                total_costs = total_costs + delta_thres[i]
    return total_costs.item()

def get_actual_costs_for_join_order(join_order, card, pred, pred_sel):
    total_costs = 0 
    int_costs = get_intermediate_costs_for_join_order(join_order, card, pred, pred_sel)
    for cost in int_costs:
        total_costs = total_costs + cost
    return total_costs

def get_intermediate_costs_for_join_order(join_order, card, pred, pred_sel):
    int_costs = []
    for j in range(1, len(card)-1): # Ignore first join
        int_card = calculate_intermediate_cardinality_for_join(j, card, pred, pred_sel, join_order)
        int_costs.append(int_card)
    return int_costs

def brute_force_JO(card, pred, pred_sel, thres = None):
    from math import inf
    perm = np.arange(0, len(card))
    min_costs = inf
    min_order = None
    second_costs = inf
    second_min_order = None
    for join_order in multiset_permutations(perm):
        costs = 0
        if thres is None:
            costs = get_actual_costs_for_join_order(join_order, card, pred, pred_sel)
        else:
            costs = get_threshold_costs_for_join_order(join_order, card, pred, pred_sel, thres)
        #print(join_order)
        #print(costs)
        if costs < min_costs:
            min_costs = costs
            min_order = join_order
        if costs > min_costs and costs < second_costs:
            second_costs = costs
            second_min_order = join_order
    return min_costs, min_order, second_costs, second_min_order

def postprocess_IBMQ_response(response, card, pred, pred_sel, thres):
        
    best_join_order = None
    best_join_order_costs = 0
    valid_ratio = 0
    optimal_ratio = 0
    
    min_order, alternative_min_order = get_optimal_join_order(card, pred, pred_sel)
            
    for i in range(len(response.samples)):
        sample = response.samples[i].x
        energy = response.samples[i].fval
        prob = response.samples[i].probability

        join_order = get_join_tree_leaves(sample, len(card))
        if join_order is not None:
            costs = get_actual_costs_for_sample(join_order, card, pred, pred_sel, thres)
            valid_ratio = valid_ratio + prob
            
            if best_join_order is None or best_join_order_costs > costs:
                best_join_order = join_order
                best_join_order_costs = costs
            
            if join_order == min_order or join_order == alternative_min_order:
                optimal_ratio += prob
    
    return best_join_order, best_join_order_costs, valid_ratio, optimal_ratio

def get_optimal_join_order(card, pred, pred_sel):
    min_costs, min_order, second_costs, second_min_order = brute_force_JO(card, pred, pred_sel, thres = None)
    alternative_min_order = list(min_order)
    alternative_min_order[1], alternative_min_order[0] = alternative_min_order[0], alternative_min_order[1]
    return min_order, alternative_min_order, min_costs

def postprocess_DWave_response(response, card, pred, pred_sel, thres):
       
    val_solution_counter = 0
    opt_solution_counter = 0
    best_join_order = None
    best_join_order_costs = 0
    first_opt_sample = None
    
    min_order, alternative_min_order, min_costs = get_optimal_join_order(card, pred, pred_sel)
    
    for i in range(len(response)):
        (sample, energy, occ) = response[i]
        join_order = get_join_tree_leaves(sample, len(card))
        if join_order is not None:
            val_solution_counter += 1
            
            # If the join order is valid (not None), fetch the corresponding actual costs
            #costs = get_actual_costs_for_sample(join_order, card, pred, pred_sel, thres)
            costs = get_actual_costs_for_join_order(join_order, card, pred, pred_sel)
            
            if best_join_order is None or best_join_order_costs > costs:
                best_join_order = join_order
                best_join_order_costs = costs
            
            #if join_order == min_order or join_order == alternative_min_order:
            if get_rounded_val(min_costs, 0) == get_rounded_val(costs, 0):
                opt_solution_counter += 1
                if first_opt_sample is None:
                    first_opt_sample = i
        
    valid_ratio = val_solution_counter / len(response)
    optimal_ratio = opt_solution_counter / len(response)
        
    return best_join_order, best_join_order_costs, valid_ratio, optimal_ratio, first_opt_sample

