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

import boto3
from braket.ocean_plugin import BraketSampler, BraketDWaveSampler
from braket.aws import AwsDevice
import os
os.environ['AWS_DEFAULT_REGION'] = "us-west-2"
   
my_bucket = f"bucketname" # specify the name of the bucket
my_prefix = "foldername" # specify the name of the folder placed in the bucket

s3_folder = (my_bucket, my_prefix)
    
adv_solver = BraketDWaveSampler(s3_folder,'arn:aws:braket:us-west-2::device/qpu/d-wave/Advantage_system6')


# In[4]:


def solve_problem_classical(bqm, num_reads=20):
    return SimulatedAnnealingSampler().sample(bqm, num_reads=num_reads, answer_mode='raw')

def solve_problem(bqm, sampler, num_reads=20, annealing_time=None):
    response = None
    if annealing_time is not None:
        response = sampler.sample(bqm, num_reads=num_reads, annealing_time=annealing_time, answer_mode='raw')    
    else:
        response = sampler.sample(bqm, num_reads=num_reads, answer_mode='raw')
    
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


# In[7]:


def response_to_dict(raw_response, use_classical_solver=True):
    response = []
    for i in range(len(raw_response.record)):
        if use_classical_solver:
            (sample, energy, occ) = raw_response.record[i]
            response.append([sample.tolist(), energy.item(), occ.item()])
        else:
            (sample, energy, occ, chain) = raw_response.record[i]
            response.append([sample.tolist(), energy.item(), occ.item()])
    return response


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


# In[10]:


def load_data(path, filename):
    datafile = os.path.abspath(path + '/' + filename)
    if os.path.exists(datafile):
        with open(datafile, 'rb') as file:
            return json.load(file)

def save_data(data, path, filename):
    datapath = os.path.abspath(path)
    pathlib.Path(datapath).mkdir(parents=True, exist_ok=True) 
    
    datafile = os.path.abspath(path + '/' + filename)
    mode = 'a' if os.path.exists(datafile) else 'w'
    with open(datafile, mode) as file:
        json.dump(data, file)

# Pre-embeds the problems for the actual QPU experiments.
# Thereby determines and persists the best embedding for the problems over the specified number of iterations.
def preembed_problems_for_QPU_experiments(relations, num_decimal_pos, graph_types, problems, problem_path_prefix, iterations=20):
    
    classical_sampler = SimulatedAnnealingSampler()
    QPUGraph = nx.Graph(adv_solver.edgelist)
    structured_sampler = dimod.StructureComposite(classical_sampler, QPUGraph.nodes, QPUGraph.edges)
    pegasus_embedding_sampler = EmbeddingComposite(structured_sampler)
            
    for i in relations:
        for decimal_pos in num_decimal_pos:
            for graph_type in graph_types:
                for j in problems:
                    problem_path = problem_path_prefix + '/' + graph_type + "_query/" + str(i) + "relations/problem" + str(j)
                    card, pred, pred_sel = ProblemGenerator.get_join_ordering_problem(problem_path, generated_problems=False)
                    thres = load_data(problem_path, 'thres.txt')
                    bqm, weight_a = QUBOGenerator.generate_QUBO_for_DWave(card, thres, decimal_pos, pred, pred_sel)
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
                        save_to_path(best_embedding, problem_path + "/embedding/" + str(decimal_pos) + "decpos", 'best_embedding.txt')
                    
## Experiments to verify how quickly (i.e., at which shot) an optimal solution is typically found
def conduct_QPU_experiments(relations, num_decimal_pos, annealing_times, graph_types, problems, problem_path_prefix, result_path_prefix, num_reads=1000, samples = range(0, 20), use_classical_solver=True):
    
    #problem_path_prefix = "ExperimentalAnalysis/DWave/QPUPerformance/Problems/SteinbrunnQueries"
    #result_path_prefix = "ExperimentalAnalysis/DWave/QPUPerformance/Results/Data/SteinbrunnQueries"
    
    if not use_classical_solver:
        sampler = adv_solver
    
    for at in annealing_times:
        for graph_type in graph_types:
            for i in relations:
                for decimal_pos in num_decimal_pos:
                    for j in problems:
                        problem_path_main = graph_type + "_query/" + str(i) + "relations/problem" + str(j)
                        card, pred, pred_sel = ProblemGenerator.get_join_ordering_problem(problem_path_prefix + '/' + problem_path_main, generated_problems=False)
                        thres = load_data(problem_path_prefix + '/' + problem_path_main, 'thres.txt')
                        embedding = format_loaded_embedding(load_from_path(problem_path_prefix + '/' + problem_path_main + "/embedding/" + str(decimal_pos) + "decpos" + '/best_embedding.txt'))                
                        bqm, weight_a = QUBOGenerator.generate_QUBO_for_DWave(card, thres, decimal_pos, pred, pred_sel)
                        if not use_classical_solver:
                            embedded_sampler = embed_sampler(sampler, embedding)
                        else:
                            embedded_sampler = None
                        for s in samples:
                            response = None
                            if use_classical_solver:
                                response = solve_problem_classical(bqm, num_reads=num_reads)
                            else:
                                response = solve_problem(bqm, embedded_sampler, num_reads=num_reads, annealing_time=at)
                            result_path_suffix = str(decimal_pos) + 'decpos/sample' + str(s)
                            result_path = result_path_prefix + '/' + str(at) + '_AT/' + problem_path_main + '/' + result_path_suffix
                            save_to_path(response_to_dict(response, use_classical_solver), result_path, "response.txt")


def parse_QPU_opt_sample_index(relations, num_decimal_pos, annealing_times, graph_types, problems, problem_path_prefix, result_path_prefix, query_generation_name, samples = range(0, 20), aggregate_data = False):
    path = 'ExperimentalAnalysis/DWave/QPUPerformance/Results'
    
    if aggregate_data:
        save_to_csv(['query_generation', 'num_relations', 'num_decimal_pos', 'annealing_time', 'query_graph', 'problem_index', 'mean_first_opt_index', 'std_first_opt_index'], path, 'aggregated_opt_index_results_benchmark.txt')
    else:
        save_to_csv(['query_generation', 'num_relations', 'num_decimal_pos', 'annealing_time', 'query_graph', 'problem_index', 'sample_index', 'first_opt_index'], path, 'opt_index_results_benchmark.txt')
    
    #problem_path_prefix = "ExperimentalAnalysis/DWave/QPUPerformance/Problems/SteinbrunnQueries"
    #result_path_prefix = "ExperimentalAnalysis/DWave/QPUPerformance/Results/Data/SteinbrunnQueries"
    for graph_type in graph_types:
        for decimal_pos in num_decimal_pos:
            for at in annealing_times:
                for i in relations:
                    if graph_type == 'STAR' and i == 3:
                        continue
                    for p in problems:
                        first_opt_samples = []
                        for j in samples:
                            problem_path_main = graph_type + "_query/" + str(i) + "relations/problem" + str(p)
                            card, pred, pred_sel = ProblemGenerator.get_join_ordering_problem(problem_path_prefix + '/' + problem_path_main, generated_problems=False)
                            thres = load_data(problem_path_prefix + '/' + problem_path_main, 'thres.txt')
                            result_path_suffix = str(decimal_pos) + 'decpos/sample' + str(j)
                            response = load_from_path(result_path_prefix + '/' + str(at) + '_AT/' + problem_path_main + '/' + result_path_suffix + '/response.txt')
                            best_join_order, best_join_order_costs, valid_ratio, optimal_ratio, first_opt_sample = Postprocessing.postprocess_DWave_response(response, card, pred, pred_sel, thres)
                            if aggregate_data:
                                first_opt_samples.append(first_opt_sample)
                            else:
                                if first_opt_sample is None:
                                    first_opt_sample = 'None'
                                save_to_csv([query_generation_name, i, decimal_pos, at, graph_type, p, j, first_opt_sample], path, 'opt_index_results_benchmark.txt')
                        if aggregate_data:
                            mean_first_opt_sample = None
                            std_first_opt_sample = None
                            if None in first_opt_samples:
                                mean_first_opt_sample = 'None'
                                std_first_opt_sample = 'None'
                            else:
                                mean_first_opt_sample = np.ceil(np.mean(first_opt_samples))
                                std_first_opt_sample = np.ceil(np.std(first_opt_samples))
                            save_to_csv([query_generation_name, i, decimal_pos, at, graph_type, p, mean_first_opt_sample, std_first_opt_sample], path, 'aggregated_opt_index_results_benchmark.txt')                    
                            
if __name__ == '__main__':
    
    relations = [2, 3, 4]
    decimal_positions = [2]
    query_graph_types = ["CLIQUE"]
    problems = range(0, 20)
    samples = range(0, 20)
    annealing_times = [0.5]
    
    problem_path_prefix = "ExperimentalAnalysis/DWave/QPUPerformance/Problems/SteinbrunnQueries"
    result_path_prefix = "ExperimentalAnalysis/DWave/QPUPerformance/Results/Data/SteinbrunnQueries"

    preembed_problems_for_QPU_experiments(relations, decimal_positions, query_graph_types, problems, problem_path_prefix, iterations=20)
    conduct_QPU_experiments(relations, decimal_positions, annealing_times, query_graph_types, problems, problem_path_prefix, result_path_prefix, num_reads=5000, samples = samples, use_classical_solver=True)
    parse_QPU_opt_sample_index(relations, decimal_positions, annealing_times, query_graph_types, problems, problem_path_prefix, result_path_prefix, 'Steinbrunn', samples = samples, aggregate_data = False)
    
    relations = [2, 3, 4]
    decimal_positions = [2]
    query_graph_types = ["CLIQUE"]
    problems = range(0, 20)
    samples = range(0, 10)
    annealing_times = [0.5]
    
    problem_path_prefix = "ExperimentalAnalysis/DWave/QPUPerformance/Problems/ManciniQueries"
    result_path_prefix = "ExperimentalAnalysis/DWave/QPUPerformance/Results/Data/ManciniQueries"

    preembed_problems_for_QPU_experiments(relations, decimal_positions, query_graph_types, problems, problem_path_prefix, iterations=20)
    conduct_QPU_experiments(relations, decimal_positions, annealing_times, query_graph_types, problems, problem_path_prefix, result_path_prefix, num_reads=100, samples = samples, use_classical_solver=False)
    parse_QPU_opt_sample_index(relations, decimal_positions, annealing_times, query_graph_types, problems, problem_path_prefix, result_path_prefix, 'Mancini', samples = samples, aggregate_data = False)

