#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import itertools
import math
from decimal import *

from dimod import *
from neal.sampler import SimulatedAnnealingSampler
from dimod.reference.samplers import ExactSolver

import dwave_networkx as dnx
import networkx as nx
from dwave.system import DWaveSampler, EmbeddingComposite, FixedEmbeddingComposite
import dwave.embedding
from dwave.cloud.client import Client

import json
import os
import pathlib
import pickle
import csv

import Scripts.ProblemGenerator as ProblemGenerator
import Scripts.QUBOGenerator as QUBOGenerator
import Scripts.Postprocessing as Postprocessing
from multiprocessing import Pool


# In[2]:


def get_rounded_val(val):
    return Decimal(val).quantize(Decimal(10) ** -4, rounding=ROUND_HALF_EVEN)


# In[3]:


problem_path = "problems/"
client = Client.from_config(software=True)
adv_solver = DWaveSampler(aggregate=False, num_qubits__gt=3000)


# In[4]:


# Fuer T=6, chain_strength=60 leads to the percentage of samples with breaks > 0.33 = 1%
def solve_problem(bqm, sampler, num_reads=10000, annealing_time=None, chain_strength=370):
        
    response = None
    if annealing_time is not None:
        response = sampler.sample(bqm, num_reads=num_reads, chain_strength=chain_strength, annealing_time=annealing_time, answer_mode='raw')
    else:
        response = sampler.sample(bqm, num_reads=num_reads, chain_strength=chain_strength, answer_mode='raw')
    
    return response


# In[5]:


def embed_sampler(sampler, embedding):
    QPUGraph = nx.Graph(adv_solver.edgelist)
    structured_sampler = dimod.StructureComposite(sampler, QPUGraph.nodes, QPUGraph.edges)
    embedded_sampler = FixedEmbeddingComposite(structured_sampler, embedding=embedding)
    return embedded_sampler


# In[6]:


# JSON does not support python tuples and therefore replaces tuples with lists.
# This function reverts this change
def format_loaded_embedding(embedding):
    formatted_embedding = {}
    for (k, v) in embedding.items():
        form_key = int(k)
        form_val = tuple(v)
        formatted_embedding[form_key] = form_val
    return formatted_embedding

def format_loaded_pred(pred):
    form_pred = []
    for p in pred:
        form_pred.append(tuple(p))
    return form_pred

def load_from_path(problem_path):
    data_file = os.path.abspath(problem_path)
    if os.path.exists(data_file):
        with open(data_file) as file:
            data = json.load(file)
            return data
        
def load_from_path_pickle(problem_path):
    data_file = os.path.abspath(problem_path)
    if os.path.exists(data_file):
        with open(data_file, 'rb') as file:
            data = pickle.load(file)
            return data
    
def load_problem_from_disc(problem_path):

    card = load_from_path(problem_path + "/card.txt")      
    pred = format_loaded_pred(load_from_path(problem_path + "/pred.txt"))    
    pred_sel = load_from_path(problem_path + "/pred_sel.txt")
    thres = load_from_path(problem_path + "/thres.txt")    
    
    return card, pred, pred_sel, thres

def save_to_path(data, path, filename):
    
    pathdir = os.path.abspath(path)
    pathlib.Path(pathdir).mkdir(parents=True, exist_ok=True) 
    
    with open(path + '/' + filename, 'w') as file:
        json.dump(data, file)
        
def save_to_path_pickle(data, path, filename):
    
    pathdir = os.path.abspath(path)
    pathlib.Path(pathdir).mkdir(parents=True, exist_ok=True) 
    
    ds_name = os.path.abspath(path + '/' + filename)
    with open(ds_name, 'wb') as file:
        pickle.dump(data, file)


# In[7]:


def response_to_dict(raw_response):
    response = []
    for i in range(len(raw_response.record)):
        (sample, energy, occ, chain) = raw_response.record[i]
        response.append([sample.tolist(), energy.item(), occ.item()])
    return response

# Pre-embeds the problems for the actual QPU experiments.
# Thereby determines and persists the best embedding for the problems over the specified number of iterations.
def preembed_problems_for_QPU_experiments(iterations=20):
    
    classical_sampler = SimulatedAnnealingSampler()
    QPUGraph = nx.Graph(adv_solver.edgelist)
    structured_sampler = dimod.StructureComposite(classical_sampler, QPUGraph.nodes, QPUGraph.edges)
    pegasus_embedding_sampler = EmbeddingComposite(structured_sampler)
    
    num_decimal_pos = 0
    
    problem_path = 'ExperimentalAnalysis/DWave/QPUPerformance/Problems/IntLogQueries/'
    
    graph_types = ["CLIQUE", "CHAIN", "STAR", "CYCLE"]
  
    for i in range(2, 6):
        for graph_type in graph_types:
            card, pred, pred_sel, thres = load_problem_from_disc(problem_path + graph_type + '_query/' + str(i) + 'relations')
            bqm, weight_a = QUBOGenerator.generate_QUBO_for_DWave(card, thres, num_decimal_pos, pred, pred_sel)
            best_embedding = None
            best_embedding_size = 0
            for j in range(iterations):
                try:
                    pegasus_response = pegasus_embedding_sampler.sample(bqm, return_embedding=True)
                    pegasus_embedding = pegasus_response.info.get('embedding_context').get('embedding')
        
                    physical_pegasus_qubits = set()
                    for value in pegasus_embedding.values():
                        for phys_qubit in value:
                            physical_pegasus_qubits.add(phys_qubit)
                    if best_embedding is None or len(physical_pegasus_qubits) < best_embedding_size:
                        best_embedding = pegasus_embedding
                        best_embedding_size = len(physical_pegasus_qubits)
                        
                except ValueError:
                    # No embedding determined
                    continue
            if best_embedding is not None:
                save_to_path(best_embedding, problem_path + graph_type + '_query/' + str(i) + 'relations', 'best_embedding.txt')

#def conduct_QPU_experiments(num_reads=900, samplesize=20):
def conduct_QPU_experiments(annealing_times, graph_types, num_reads=1000, samplesize=20):
    
    chain_strengths = {"CHAIN": {3: 5, 4: 300, 5: 450, 6: 600, 7: 200, 8: 270, 9: 370},
                    "STAR": {3: 35, 4: 3000, 5: 300, 6: 450, 7: 200, 8: 270, 9: 370},
                  "CYCLE": {3: 15, 4: 300, 5: 325, 6: 400, 7: 200, 8: 270, 9: 370},
                "CLIQUE": {2: 35, 3: 35, 4: 300, 5: 300, 6: 450, 7: 200, 8: 270, 9: 370}}
    
    problem_path = 'ExperimentalAnalysis/DWave/QPUPerformance/Problems/IntLogQueries/'
    result_path = 'ExperimentalAnalysis/DWave/QPUPerformance/Results/Data/IntLogQueries/'
    
    num_decimal_pos = 0
    
    sampler = adv_solver

    for at in annealing_times:
        for graph_type in graph_types:
            for i in range(2, 6):
                card, pred, pred_sel, thres = load_problem_from_disc(problem_path + graph_type + '_query/' + str(i) + 'relations')
                embedding = format_loaded_embedding(load_from_path(problem_path + graph_type + '_query/' + str(i) + 'relations/best_embedding.txt'))                
                for j in range(samplesize):
                    bqm, weight_a = QUBOGenerator.generate_QUBO_for_DWave(card, thres, num_decimal_pos, pred, pred_sel)
                    embedded_sampler = embed_sampler(sampler, embedding)
                    response = solve_problem(bqm, embedded_sampler, num_reads=num_reads, annealing_time=at, chain_strength=chain_strengths[graph_type][i])
                    save_to_path(response_to_dict(response), result_path + str(at) + '_AT/' + graph_type + '_query/' + str(i) + 'relations/' + str(j), "best_embedding_response.txt")


# In[8]:


def save_result(graph_type, num_relations, num_thres, num_decimal_pos, sample_index, embedding):

    path_string = 'ExperimentalAnalysis/DWave/Embeddings/Results/Data/' + str(num_relations) + '_relations/' + str(num_thres) + '_thres/' + str(num_decimal_pos) + '_decpos/' + str(graph_type) + '_graph/' + str(sample_index)

    datapath = os.path.abspath(path_string)
    pathlib.Path(datapath).mkdir(parents=True, exist_ok=True) 
    
    datafile = os.path.abspath(path_string + '/embedding.txt')
    mode = 'a' if os.path.exists(datafile) else 'w'
    with open(datafile, mode) as file:
        json.dump(embedding, file)
        
def save_bqm(graph_type, num_relations, num_thres, num_decimal_pos, sample_index, bqm):
    
    path_string = 'ExperimentalAnalysis/DWave/Embeddings/Problems/' + str(num_relations) + '_relations/' + str(num_thres) + '_thres/' + str(num_decimal_pos) + '_decpos/' + str(graph_type) + '_graph/' + str(sample_index)

    datapath = os.path.abspath(path_string)
    pathlib.Path(datapath).mkdir(parents=True, exist_ok=True) 
    
    datafile = os.path.abspath(path_string + '/bqm.txt')
    mode = 'ab' if os.path.exists(datafile) else 'wb'
    with open(datafile, mode) as file:
        pickle.dump(bqm, file)
        
def load_bqm(graph_type, num_relations, num_thres, num_decimal_pos, sample_index):

    path_string = 'ExperimentalAnalysis/DWave/Embeddings/Problems/' + str(num_relations) + '_relations/' + str(num_thres) + '_thres/' + str(num_decimal_pos) + '_decpos/' + str(graph_type) + '_graph/' + str(sample_index)

    datapath = os.path.abspath(path_string)
    pathlib.Path(datapath).mkdir(parents=True, exist_ok=True) 
    
    datafile = os.path.abspath(path_string + '/bqm.txt')
    bqm = None
    with open(datafile, 'rb') as file:
        bqm = pickle.load(file)
    return bqm

def embedding_exists(graph_type, num_relations, num_thres, num_decimal_pos):
    path_string = 'ExperimentalAnalysis/DWave/Embeddings/Results/Data/' + str(num_relations) + '_relations/' + str(num_thres) + '_thres/' + str(num_decimal_pos) + '_decpos/' + str(graph_type) + '_graph'
    return os.path.isdir(path_string)
    
def load_result(graph_type, num_relations, num_thres, num_decimal_pos, sample_index):

    path_string = 'ExperimentalAnalysis/DWave/Embeddings/Results/Data/' + str(num_relations) + '_relations/' + str(num_thres) + '_thres/' + str(num_decimal_pos) + '_decpos/' + str(graph_type) + '_graph/' + str(sample_index)
    
    datapath = os.path.abspath(path_string)
    pathlib.Path(datapath).mkdir(parents=True, exist_ok=True) 
    
    datafile = os.path.abspath(path_string + '/embedding.txt')
    embedding = []
    with open(datafile, 'r') as file:
        embedding = json.load(file)
    return embedding
    
def find_embedding(graph_type, num_relations, sample_index, thres, num_decimal_pos):
    sampler = get_pegasus_sampler()
    bqm = load_bqm(graph_type, num_relations, len(thres), num_decimal_pos, sample_index)
    
    try:
        response = sampler.sample(bqm, return_embedding=True)
        embedding = response.info.get('embedding_context').get('embedding')
        
        physical_qubits = set()

        for value in embedding.values():
            for phys_qubit in value:
                physical_qubits.add(phys_qubit)
        
        save_result(graph_type, num_relations, len(thres), num_decimal_pos, sample_index, list(physical_qubits))
    except ValueError:
        save_result(graph_type, num_relations, len(thres), num_decimal_pos, sample_index, list())
    
def get_pegasus_sampler():
    classical_sampler = SimulatedAnnealingSampler()
    QPUGraph = nx.Graph(adv_solver.edgelist)
    structured_sampler = dimod.StructureComposite(classical_sampler, QPUGraph.nodes, QPUGraph.edges)
    pegasus_embedding_sampler = EmbeddingComposite(structured_sampler)
    return pegasus_embedding_sampler
    
def find_embeddings_for_increasing_relations(min_num_relations, max_num_relations):
    graph_types = ["CHAIN", "STAR", "CYCLE"]
    samplesize = 20
    thres = [10]
    num_decimal_pos = 0
    
    pool = Pool()
    
    for graph_type in graph_types:
        for i in range(min_num_relations, max_num_relations+1):
            for j in range(samplesize):
                pool.apply_async(find_embedding, args=(graph_type, i, j, thres, num_decimal_pos, ))
            
    pool.close()
    pool.join()
    
def find_embeddings_for_increasing_precision(max_thres_number, num_decimal_pos):
    graph_types = ["CHAIN", "STAR", "CYCLE"]
    samplesize = 20
    num_relations = 8
    
    pool = Pool()
    
    for graph_type in graph_types:
        thres = []
        for i in range(max_thres_number):
            thres.append(10)
            for j in range(samplesize):
                pool.apply_async(find_embedding, args=(graph_type, num_relations, j, thres.copy(), num_decimal_pos, ))
                
    pool.close()
    pool.join()


# In[9]:


def save_to_csv(data, path, filename):
    
    sd = os.path.abspath(path)
    pathlib.Path(sd).mkdir(parents=True, exist_ok=True) 
    
    f = open(path + '/' + filename, 'a', newline='')
    writer = csv.writer(f)
    writer.writerow(data)
    f.close()

def process_data(embedding_sizes):
    embeddings = []
    for (k, v) in embedding_sizes.items():
        if v != 0:
            embeddings.append(v)
    minimum_size = np.amin(embeddings)
    mean_size = int(np.ceil(np.mean(embeddings)))
    median_size = int(np.ceil(np.median(embeddings)))
    maximum_size = np.amax(embeddings)
    return minimum_size, mean_size, median_size, maximum_size

def parse_QPU_data(aggregate_data = False):
    path = 'ExperimentalAnalysis/DWave/QPUPerformance/Results'
    annealing_times = [10, 100, 1000]
    graph_types = ['CHAIN', 'STAR', 'CYCLE']
    if aggregate_data:
        save_to_csv(['annealing_time', 'query_graph', 'num_relations', 'mean_valid_ratio', 'mean_opt_ratio', 'mean_first_opt_index', 'std_first_opt_index'], path, 'aggregated_QPU_results.txt')
    else:
        save_to_csv(['annealing_time', 'query_graph', 'num_relations', 'sample_index', 'valid_ratio', 'opt_ratio', 'first_opt_index'], path, 'QPU_results.txt')
    for graph_type in graph_types:
        for at in annealing_times:
            for i in range(3, 6):
                if graph_type == 'STAR' and i == 3:
                    continue
                valid_ratios = []
                opt_ratios = []
                first_opt_samples = []           
                for j in range(20):
                    problem_path = 'ExperimentalAnalysis/DWave/QPUPerformance/Problems/IntLogQueries/' + graph_type + '_query/' + str(i) + 'relations'
                    card, pred, pred_sel, thres = load_problem_from_disc(problem_path)
                    response_path = 'ExperimentalAnalysis/DWave/QPUPerformance/Results/Data/IntLogQueries/' + str(at) + '_AT/' + graph_type + '_query/' + str(i) + 'relations/' + str(j) + "/best_embedding_response.txt"
                    response = load_from_path(response_path)
                    best_join_order, best_join_order_costs, valid_ratio, optimal_ratio, first_opt_sample = Postprocessing.postprocess_DWave_response(response, card, pred, pred_sel, thres)
                    if aggregate_data:
                        valid_ratios.append(valid_ratio)
                        opt_ratios.append(optimal_ratio)
                        first_opt_samples.append(first_opt_sample)
                    else:
                        if first_opt_sample is None:
                            first_opt_sample = 'None'
                        save_to_csv([at, graph_type, i, j, get_rounded_val(valid_ratio), get_rounded_val(optimal_ratio), first_opt_sample], path, 'QPU_results.txt')
                if aggregate_data:
                    mean_valid_ratio = get_rounded_val(np.mean(valid_ratios))
                    mean_opt_ratio = get_rounded_val(np.mean(opt_ratios))
                    mean_first_opt_sample = None
                    std_first_opt_sample = None
                    if None in first_opt_samples:
                        mean_first_opt_sample = 'None'
                        std_first_opt_sample = 'None'
                    else:
                        mean_first_opt_sample = np.ceil(np.mean(first_opt_samples))
                        std_first_opt_sample = np.ceil(np.std(first_opt_samples))
                    save_to_csv([at, graph_type, i, mean_valid_ratio, mean_opt_ratio, mean_first_opt_sample, std_first_opt_sample], path, 'aggregated_QPU_results.txt')

def get_randomized_result_batch(sample_size=900, num_vars=28):
    results = np.empty(sample_size).tolist()
    for i in range(sample_size):
        config = np.random.randint(2, size=num_vars).tolist()
        results[i] = (config, 10, 1)
    return results

def add_randomized_batch_to_csv(aggregate_data = True):
    path = 'ExperimentalAnalysis/DWave/QPUPerformance/Results'
    first_opt_samples = []
    for j in range(20):
        problem_path = 'ExperimentalAnalysis/DWave/QPUPerformance/Problems/IntLogQueries/CYCLE_query/3relations'
        card, pred, pred_sel, thres = load_problem_from_disc(problem_path)
        response_path = 'ExperimentalAnalysis/DWave/QPUPerformance/Results/Data/IntLogQueries/Random/CYCLE_query/3relations/' + str(j) + "/best_embedding_response.txt"
        response = get_randomized_result_batch()
        best_join_order, best_join_order_costs, valid_ratio, optimal_ratio, first_opt_sample = Postprocessing.postprocess_DWave_response(response, card, pred, pred_sel, thres)
        if aggregate_data:
            first_opt_samples.append(first_opt_sample)
        else:
            if first_opt_sample is None:
                first_opt_sample = 'None'
            save_to_csv(['Guessing', 'CYCLE', j, first_opt_sample], path, 'opt_index_results_IntLog.txt')
    if aggregate_data:
        mean_first_opt_sample = None
        std_first_opt_sample = None
        if None in first_opt_samples:
            mean_first_opt_sample = 'None'
            std_first_opt_sample = 'None'
        else:
            mean_first_opt_sample = np.ceil(np.mean(first_opt_samples))
            std_first_opt_sample = np.ceil(np.std(first_opt_samples))
        save_to_csv(['Guessing', 'CYCLE', mean_first_opt_sample, std_first_opt_sample], path, 'aggregated_opt_index_results_IntLog.txt')                    
            
def parse_QPU_opt_sample_index(aggregate_data = False, include_guessing = False):
    path = 'ExperimentalAnalysis/DWave/QPUPerformance/Results'

    annealing_times = [0.5, 2.5, 5, 7.5]
    
    graph_types = ['CLIQUE']
    if aggregate_data:
        save_to_csv(['num_relations', 'annealing_time', 'query_graph', 'mean_first_opt_index', 'std_first_opt_index'], path, 'aggregated_opt_index_results_IntLog.txt')
    else:
        save_to_csv(['num_relations', 'annealing_time', 'query_graph', 'sample_index', 'first_opt_index'], path, 'opt_index_results_IntLog.txt')
    for graph_type in graph_types:
        for at in annealing_times:
            for i in range(2, 5):
            #for i in range(3, 4):
                if graph_type == 'STAR' and i == 3:
                    continue
                first_opt_samples = []
                for j in range(20):
                    problem_path = 'ExperimentalAnalysis/DWave/QPUPerformance/Problems/IntLogQueries/' + graph_type + '_query/' + str(i) + 'relations'
                    card, pred, pred_sel, thres = load_problem_from_disc(problem_path)
                    response_path = 'ExperimentalAnalysis/DWave/QPUPerformance/Results/Data/IntLogQueries/' + str(at) + '_AT/' + graph_type + '_query/' + str(i) + 'relations/' + str(j) + "/best_embedding_response.txt"
                    response = load_from_path(response_path)
                    best_join_order, best_join_order_costs, valid_ratio, optimal_ratio, first_opt_sample = Postprocessing.postprocess_DWave_response(response, card, pred, pred_sel, thres)
                    if aggregate_data:
                        first_opt_samples.append(first_opt_sample)
                    else:
                        if first_opt_sample is None:
                            first_opt_sample = 'None'
                        save_to_csv([i, at, graph_type, j, first_opt_sample], path, 'opt_index_results_IntLog.txt')
                if aggregate_data:
                    mean_first_opt_sample = None
                    std_first_opt_sample = None
                    if None in first_opt_samples:
                        mean_first_opt_sample = 'None'
                        std_first_opt_sample = 'None'
                    else:
                        mean_first_opt_sample = np.ceil(np.mean(first_opt_samples))
                        std_first_opt_sample = np.ceil(np.std(first_opt_samples))
                    save_to_csv([i, at, graph_type, mean_first_opt_sample, std_first_opt_sample], path, 'aggregated_opt_index_results_IntLog.txt')                    
    if include_guessing:
        add_randomized_batch_to_csv(aggregate_data)
    
def reliability_threshold_reached(embedding_sizes, samplesize, reliability_threshold = 0.5):
    embeddings_counter = 0
    for (k, v) in embedding_sizes.items():
        if v != 0:
            embeddings_counter += 1
    if embeddings_counter >= get_rounded_val(samplesize*reliability_threshold):
        return True
    return False
    
def parse_embedding_data(aggregate_data = False):
    graph_types = ["CHAIN", "STAR", "CYCLE"]
    decpos_values = [0, 2, 4]
    samplesize = 20
    path = 'ExperimentalAnalysis/DWave/Embeddings/Results'
    if aggregate_data:
        save_to_csv(['num_relations','num_threshold_values','num_decimal_pos','query_graph_type','min_size', 'mean_size', 'med_size', 'max_size'], path, 'aggregated_embeddings.txt')
    else:
        save_to_csv(['num_relations','num_threshold_values','num_decimal_pos','query_graph_type','sample_index','num_phys_qubits'], path, 'embeddings.txt')
    for i in range(4, 21):
        for j in range(1, 28):
            for decpos in decpos_values:
                for graph_type in graph_types:
                    if not embedding_exists(graph_type, i, j, decpos):
                        continue
                    embedding_sizes = {}
                    for k in range(samplesize):
                        embedding = load_result(graph_type, i, j, decpos, k)
                        embedding_sizes[k] = len(embedding)
                    if reliability_threshold_reached(embedding_sizes, samplesize):
                        if aggregate_data: 
                            minimum_size, mean_size, median_size, maximum_size = process_data(embedding_sizes)
                            save_to_csv([i, j, decpos, graph_type, minimum_size, mean_size, median_size, maximum_size], path, 'aggregated_embeddings.txt')
                        else:
                            for (key, val) in embedding_sizes.items():
                                save_to_csv([i, j, decpos, graph_type, key, val], path, 'embeddings.txt')


# In[10]:


if __name__ == '__main__':
    
    #find_embeddings_for_increasing_relations(4, 20)
    #find_embeddings_for_increasing_precision(27, 0)
    #find_embeddings_for_increasing_precision(12, 2)
    #find_embeddings_for_increasing_precision(7, 4)
    #parse_embedding_data()
    
    preembed_problems_for_QPU_experiments()
    ## Experiments to verify how quickly (i.e., at which shot) an optimal solution is typically found
    conduct_QPU_experiments([0.5, 2.5, 5, 7.5], ["CLIQUE"], num_reads=10000, samplesize=20)
    parse_QPU_opt_sample_index()
    ## Experiments to verify how solution quality (i.e., ratios of valid and optimal solution) deteriorates with increasing query size
    conduct_QPU_experiments([10, 100, 1000], ["CHAIN", "STAR", "CYCLE"], num_reads=900, samplesize=20)
    parse_QPU_data()

