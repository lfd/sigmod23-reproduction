#!/usr/bin/env python
# coding: utf-8

# In[1]:


import itertools
import json
import os
import pathlib 
import pickle
from decimal import *
import numpy as np
import itertools
from sympy.utilities.iterables import multiset_permutations
import numpy as np
from math import inf

def save_data(data, path, filename):
    datapath = os.path.abspath(path)
    pathlib.Path(datapath).mkdir(parents=True, exist_ok=True) 
    
    datafile = os.path.abspath(path + '/' + filename)
    mode = 'a' if os.path.exists(datafile) else 'w'
    with open(datafile, mode) as file:
        json.dump(data, file)

# Code adapted from https://github.com/itrummer/query-optimizer-lib
def get_cardinality():
    r = np.random.uniform(low=0.0, high=1.0, size=None)
    
    if r < 0.15:
        cardinality = np.random.randint(10, 101)
    elif r < 0.45:
        cardinality = np.random.randint(100, 1001)
    elif r < 0.8:
        cardinality = np.random.randint(1000, 10001)
    else:
        cardinality = np.random.randint(10000, 100001)
    return cardinality
    
def get_domain_size():
    r = np.random.uniform(low=0.0, high=1.0, size=None)
    
    if r < 0.05:
        domain_size = np.random.randint(2, 11)
    elif r < 0.55:
        domain_size = np.random.randint(10, 101)
    elif r < 0.85:
        domain_size = np.random.randint(100, 501)
    else:
        domain_size = np.random.randint(500, 1001)
    return domain_size
    
def get_selectivity(domain_size1, domain_size2, cardinality1, cardinality2, join_type):
    if join_type == 'Random':
        selectivity = 0
        while selectivity == 0:
            selectivity = np.random.uniform(low=0.0, high=1.0, size=None)
    elif join_type == 'MIN':
        selectivity = 1.0 / max(cardinality1, cardinality2)
    elif join_type == 'MAX':
        selectivity = 1.0 / min(cardinality1, cardinality2)
    elif join_type == 'MN':
        selectivity = 1.0 / max(domain_size_1, domain_size_2)
    elif join_type == 'MINMAX':
        lb = 1.0/max(cardinality1, cardinality2);
        ub = 1.0/min(cardinality1, cardinality2);
        delta = ub - lb
        selectivity = lb + np.random.uniform(low=0.0, high=1.0, size=None)*delta
    else:
        return False
    return selectivity
   
def produce_steinbrunn_query(num_relations, graph_type, join_type, max_cardinality=100000):
    scaling = max_cardinality / 100000.0
    
    cardinalities = []
    domain_sizes = []
    
    for i in range(num_relations):
        cardinality = get_cardinality()*scaling
        cardinalities.append(cardinality)
        
        domain_size = get_domain_size()
        domain_sizes.append(domain_size)
        
    selectivities = np.ones((num_relations, num_relations)).tolist()
    if graph_type == 'STAR' or graph_type == 'CHAIN':
        num_predicates = num_relations -1 
    elif graph_type == 'CYCLE':
        num_predicates = num_relations
    
    if graph_type == 'STAR':
        for p in range(num_predicates):
            dim_table_index = p + 1
            domain_size1 = domain_sizes[0]
            domain_size2 = domain_sizes[dim_table_index]
            cardinality1 = cardinalities[0]
            cardinality2 = cardinalities[dim_table_index]
            selectivity = get_selectivity(domain_size1, domain_size2, cardinality1, cardinality2, join_type)
            selectivities[0][dim_table_index] = selectivity
            selectivities[dim_table_index][0] = selectivity
    elif graph_type == 'CHAIN':
        for p in range(num_predicates):
            table1 = p
            table2 = table1 + 1
            domain_size1 = domain_sizes[table1]
            domain_size2 = domain_sizes[table2]
            cardinality1 = cardinalities[table1]
            cardinality2 = cardinalities[table2]
            selectivity = get_selectivity(domain_size1, domain_size2, cardinality1, cardinality2, join_type)
            selectivities[table1][table2] = selectivity
            selectivities[table2][table1] = selectivity
    elif graph_type == 'CYCLE':
        for p in range(num_predicates-1):
            table1 = p
            table2 = table1 + 1
            domain_size1 = domain_sizes[table1]
            domain_size2 = domain_sizes[table2]
            cardinality1 = cardinalities[table1]
            cardinality2 = cardinalities[table2]
            selectivity = get_selectivity(domain_size1, domain_size2, cardinality1, cardinality2, join_type)
            selectivities[table1][table2] = selectivity
            selectivities[table2][table1] = selectivity
        table1 = 0
        table2 = num_relations-1
        domain_size1 = domain_sizes[table1]
        domain_size2 = domain_sizes[table2]
        cardinality1 = cardinalities[table1]
        cardinality2 = cardinalities[table2]
        selectivity = get_selectivity(domain_size1, domain_size2, cardinality1, cardinality2, join_type)
        selectivities[table1][table2] = selectivity
        selectivities[table2][table1] = selectivity
    elif graph_type == 'CLIQUE':
        for (table1, table2) in itertools.combinations(np.arange(num_relations), 2):
            domain_size1 = domain_sizes[table1]
            domain_size2 = domain_sizes[table2]
            cardinality1 = cardinalities[table1]
            cardinality2 = cardinalities[table2]
            selectivity = get_selectivity(domain_size1, domain_size2, cardinality1, cardinality2, join_type)
            selectivities[table1][table2] = selectivity
            selectivities[table2][table1] = selectivity
    return cardinalities, selectivities
    
def get_selectivity_for_new_relation(join_order, j, pred, pred_sel):
    sel = 1
    new_relation = join_order[j]
    for i in range(j):
        relation = join_order[i]
        if (relation, new_relation) in pred:
            sel = sel * pred_sel[pred.index((relation, new_relation))]
        elif (new_relation, relation) in pred:
            sel = sel * pred_sel[pred.index((new_relation, relation))]
    return sel

def get_intermediate_costs_for_join_order(join_order, card, pred, pred_sel, card_dict, verbose=False):
    int_costs = []
    join_order = join_order.copy()
    if join_order[0] > join_order[1]:
        join_order[0], join_order[1] = join_order[1], join_order[0]
    prev_join_result = card[join_order[0]]
    for j in range(1, len(card)-1):
        jo_hash = str(join_order[0:j+1])
        if jo_hash in card_dict:
            int_card = card_dict[jo_hash]
        else:
            sel = get_selectivity_for_new_relation(join_order, j, pred, pred_sel)
            int_card = prev_join_result * card[join_order[j]] * sel
            card_dict[jo_hash] = int_card
        prev_join_result = int_card
        int_costs.append(int_card)
    if verbose:
        print(int_costs)
    return int_costs 

def get_exact_thres(card, pred_raw, pred_sel):

    pred = []
    for i in range(len(pred_raw)):
        pred_list = pred_raw[i]
        pred.append((pred_list[0], pred_list[1]))
    
    intermediate_list = []
    for p in multiset_permutations(np.arange(len(card))):
        if p[1] < p[0]:
            h = p[1]
            p[1] = p[0]
            p[0] = h
        int_costs = get_intermediate_costs_for_join_order(p, card, pred, pred_sel, {}, verbose=False)
        intermediate_list.append(int_costs)
    
    thres_list = []
    for j in range(len(card)-2):
        min_costs = inf
        second_costs = inf
        for int_result in intermediate_list:
            int_costs = int_result[j]
            if int_costs <= min_costs:
                min_costs = int_costs
            elif int_costs < second_costs:
                second_costs = int_costs
        delta = (second_costs-min_costs)/2
        thres = int(min_costs+delta)
        if thres < 1:
            thres = 1
        thres_list.append(thres)
    return thres_list

def produce_steinbrunn_queries_for_DWave_processing(relations, graph_types, problems, thres, problem_path_prefix):
    for num_relations in relations:
        for graph_type in graph_types:
            for p in problems:
                cardinalities, selectivities = produce_steinbrunn_query(num_relations, graph_type, "Random", max_cardinality=100000)
                problem_path = problem_path_prefix + '/' + graph_type + '_query/' + str(num_relations) + 'relations/problem' + str(p)

                pred = []
                pred_sel = []
                for (table1, table2) in itertools.combinations(np.arange(len(cardinalities)), 2):
                    if selectivities[table1][table2] < 1.0:
                        pred.append([int(table1), int(table2)])
                        pred_sel.append(selectivities[table1][table2])
                save_data(cardinalities, problem_path, "card.txt")
                save_data(pred, problem_path, "pred.txt")
                save_data(pred_sel, problem_path, "pred_sel.txt")
                
                if thres is None or len(thres) == 0:
                    if len(cardinalities) <= 2:
                        calculated_exact_thres = [50]
                    else:
                        calculated_exact_thres = get_exact_thres(cardinalities, pred, pred_sel)
                    save_data(calculated_exact_thres, problem_path, "thres.txt")
                else:
                    save_data(thres, problem_path, "thres.txt")
                
def produce_steinbrunn_queries_for_codesign(relations, graph_types, problems, problem_path_prefix):
    for num_relations in relations:
        for graph_type in graph_types:
            for p in problems:
                cardinalities, selectivities = produce_steinbrunn_query(num_relations, graph_type, "Random", max_cardinality=100000)
                problem_path = problem_path_prefix + '/' + str(num_relations) + 'relations/' + graph_type + '_graph/' + str(p)
                save_data(cardinalities, problem_path, 'cardinalities.json')
                save_data(selectivities, problem_path, 'selectivities.json')
 

    