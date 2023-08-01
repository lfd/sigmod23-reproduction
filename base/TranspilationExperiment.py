#!/usr/bin/env python
# coding: utf-8

# In[1]:


import json
import os
import pathlib 
import Scripts.ProblemGenerator as ProblemGenerator
import Scripts.QUBOGenerator as QUBOGenerator
import Scripts.CircuitGeneration as CircuitGenerator
import Scripts.TopologyGenerator as TopologyGenerator
from multiprocessing import Pool
#from multiprocessing_on_dill import Pool
import numpy as np

import csv
import config as config
import Scripts.SteinbrunnQueryGenerator as SteinbrunnQueryGenerator


# In[2]:


def save_results(vendor, optimizer, querygraph, density, restrict_to_native_gates, relations, sample, depths):
    
    if optimizer == 'tket':
        circuit_transpilation = config.configuration["tket-circuit-transpilation"]
    else:
        circuit_transpilation = config.configuration["qiskit-circuit-transpilation"]
    
    if circuit_transpilation == "retranspile":
        savings_dir = 'TranspilationExperiment/Results/Retranspiled_Data'
    else:
        savings_dir = 'TranspilationExperiment/Results/Collected_Data'
       
    gateset = None
    if restrict_to_native_gates:
        gateset = 'Native_gates'
    else:
        gateset = 'Any_gates'
    path_string = savings_dir + '/' + vendor + '/' + optimizer + '/' + querygraph + '_graph/' + str(density) + '_density/' + gateset + '/' + str(relations) + '_relations/sample_' + str(sample)
    print("Saving " + path_string)

    datapath = os.path.abspath(path_string)
    pathlib.Path(datapath).mkdir(parents=True, exist_ok=True) 
    
    datafile = os.path.abspath(path_string + '/depth.txt')
    mode = 'a' if os.path.exists(datafile) else 'w'
    with open(datafile, mode) as file:
        json.dump(depths, file)
        
def save_to_csv(data, path, filename):
    sd = os.path.abspath(path)
    pathlib.Path(sd).mkdir(parents=True, exist_ok=True) 
    
    f = open(path + '/' + filename, 'a', newline='')
    writer = csv.writer(f)
    writer.writerow(data)
    f.close()
        
def load_data(path, filename):
    datafile = os.path.abspath(path + '/' + filename)
    if os.path.exists(datafile):
        with open(datafile, 'rb') as file:
            return json.load(file)
        
def get_optimizer_string(tket_optimizer, optimization_level):
    if tket_optimizer:
        return 'Tket_optimizer'
    else:
        return 'Qiskit_Lv_' + str(optimization_level) + '_optimizer'
    
def get_qiskit_IBMQ_gateset(restrict_to_native_gates):
    if restrict_to_native_gates:
        return CircuitGenerator.get_IBMQ_basis_gates()
    else:
        return None
    
def get_qiskit_Rigetti_gateset(restrict_to_native_gates):
    if restrict_to_native_gates:
        return CircuitGenerator.get_Rigetti_basis_gates()
    else:
        return None
    
def get_qiskit_IonQ_gateset(restrict_to_native_gates):
    if restrict_to_native_gates:
        return CircuitGenerator.get_IonQ_basis_gates()
    else:
        return None

def get_tket_IBMQ_gateset_pass(restrict_to_native_gates):
    if restrict_to_native_gates:
        return CircuitGenerator.fetch_IBMQ_rebase_pass()
    else:
        return None
    
def get_tket_IonQ_gateset_pass(restrict_to_native_gates):
    if restrict_to_native_gates:
        return CircuitGenerator.fetch_IonQ_rebase_pass()
    else:
        return None
    
def get_tket_Rigetti_gateset_pass(restrict_to_native_gates):
    if restrict_to_native_gates:
        return CircuitGenerator.fetch_Rigetti_rebase_pass()
    else:
        return None
    
def get_join_ordering_problem(num_relations, graph_type, sample_index):
    codesign_queries = config.configuration["codesign-queries"]
    if codesign_queries == 'collected':
        problem_path_prefix = 'TranspilationExperiment/CollectedProblems/JSON'
    else:
        problem_path_prefix = 'TranspilationExperiment/Problems/JSON'
    card, pred, pred_sel = ProblemGenerator.get_join_ordering_problem(problem_path_prefix + '/' + str(num_relations) + 'relations/' + str(graph_type) + '_graph/' + str(sample_index))
    thres = [100, 10000]
    num_decimal_pos = 0
    return card, thres, num_decimal_pos, pred, pred_sel
        
def get_QUBO(num_relations, graph_type, sample_index, load_saved_qubo = True):
    codesign_queries = config.configuration["codesign-queries"]
    if codesign_queries == 'collected':
        problem_path_prefix = 'TranspilationExperiment/CollectedProblems/QUBO'
    else:
        problem_path_prefix = 'TranspilationExperiment/Problems/JSON'
    qubo_path = problem_path_prefix + '/' + str(num_relations) + 'relations/' + str(graph_type) + '_graph/' + str(sample_index)
    if load_saved_qubo and os.path.exists(qubo_path + '/qubo.txt'):
        return ProblemGenerator.get_join_ordering_qubo(qubo_path)
    else:
        card, thres, num_decimal_pos, pred, pred_sel = get_join_ordering_problem(num_relations, graph_type, sample_index)
        qubo, weight_a =  QUBOGenerator.generate_QUBO_for_IBMQ(card, thres, num_decimal_pos, pred, pred_sel)
        return qubo
    
def get_QAOA_circuit(qubo):
    circuit = CircuitGenerator.create_QAOA_circuit(qubo)
    circuit = CircuitGenerator.decompose_circuit(circuit)
    return circuit

def conduct_IBMQ_experiment(tket_optimizer, optimization_level, restrict_to_native_gates, graph_type, density, num_relations, sample_index):
    qubo = get_QUBO(num_relations, graph_type, sample_index)
    circuit = get_QAOA_circuit(qubo)
    coupling_map = TopologyGenerator.get_IBM_Washington_topology()
    coupling_map = TopologyGenerator.expand_IBM_Washington_topology(coupling_map, len(qubo.variables))
    coupling_map = TopologyGenerator.increase_coupling_density(coupling_map, density)
    coupling_map = TopologyGenerator.format_qubit_labels_to_consecutive_integers(coupling_map)
    if tket_optimizer:
        circuit = CircuitGenerator.transpile_circuit_with_tket(circuit, coupling_map, get_tket_IBMQ_gateset_pass(restrict_to_native_gates))
    else:
        circuit = CircuitGenerator.transpile_circuit_with_qiskit(circuit, coupling_map, optimization_level, get_qiskit_IBMQ_gateset(restrict_to_native_gates))
    depths = []
    depths.append(circuit.depth())
    save_results('IBMQ', get_optimizer_string(tket_optimizer, optimization_level), graph_type, density, restrict_to_native_gates, num_relations, sample_index, depths)
    
def conduct_Rigetti_experiment(tket_optimizer, optimization_level, restrict_to_native_gates, graph_type, density, num_relations, sample_index):
    qubo = get_QUBO(num_relations, graph_type, sample_index)
    circuit = get_QAOA_circuit(qubo)
    coupling_map = TopologyGenerator.get_Rigetti_Aspen_M_topology()
    coupling_map = TopologyGenerator.expand_Rigetti_Aspen_M_topology(coupling_map, len(qubo.variables))
    coupling_map = TopologyGenerator.increase_coupling_density(coupling_map, density)
    coupling_map = TopologyGenerator.format_qubit_labels_to_consecutive_integers(coupling_map)
    if tket_optimizer:
        circuit = CircuitGenerator.transpile_circuit_with_tket(circuit, coupling_map, get_tket_Rigetti_gateset_pass(restrict_to_native_gates))
    else:
        circuit = CircuitGenerator.transpile_circuit_with_qiskit(circuit, coupling_map, optimization_level, get_qiskit_Rigetti_gateset(restrict_to_native_gates))
    depths = []
    depths.append(circuit.depth())
    save_results('Rigetti', get_optimizer_string(tket_optimizer, optimization_level), graph_type, density, restrict_to_native_gates, num_relations, sample_index, depths)
    
def conduct_IonQ_experiment(tket_optimizer, optimization_level, restrict_to_native_gates, graph_type, num_relations, sample_index):
    qubo = get_QUBO(num_relations, graph_type, sample_index)
    circuit = get_QAOA_circuit(qubo)
    coupling_map = TopologyGenerator.get_IonQ_topology(len(qubo.variables))
    if tket_optimizer:
        circuit = CircuitGenerator.transpile_circuit_with_tket(circuit, coupling_map, get_tket_IonQ_gateset_pass(restrict_to_native_gates))
    else:
        circuit = CircuitGenerator.transpile_circuit_with_qiskit(circuit, coupling_map, optimization_level, get_qiskit_IonQ_gateset(restrict_to_native_gates))
    depths = []
    depths.append(circuit.depth())
    save_results('IonQ', get_optimizer_string(tket_optimizer, optimization_level), graph_type, 1.0, restrict_to_native_gates, num_relations, sample_index, depths)


# In[3]:


def conduct_IBMQ_experiments(tket_optimizer, optimization_level, relations, graph_types, densities, sample_size):
    print('Begin IBMQ co-design experiments.')
    
    pool = Pool()

    for graph_type in graph_types:
        for density in densities:
            for j in relations:
                for k in range(sample_size):
                    #conduct_IBMQ_experiment(tket_optimizer, optimization_level, True, graph_type, density, j,  k)
                    #conduct_IBMQ_experiment(tket_optimizer, optimization_level, False, graph_type, density, j,  k)
                    response = pool.apply_async(conduct_IBMQ_experiment, args=(tket_optimizer, optimization_level, True, graph_type, density, j,  k, ))
                    response = pool.apply_async(conduct_IBMQ_experiment, args=(tket_optimizer, optimization_level, False, graph_type, density, j,  k, ))
    pool.close()
    pool.join()
    print('IBMQ experiments finished.')
    
def conduct_Rigetti_experiments(tket_optimizer, optimization_level, relations, graph_types, densities, sample_size):
    print('Begin Rigetti co-design experiments.')
    
    pool = Pool()

    for graph_type in graph_types:
        for density in densities:
            for j in relations:
                for k in range(sample_size):
                    response = pool.apply_async(conduct_Rigetti_experiment, args=(tket_optimizer, optimization_level, True, graph_type, density, j,  k, ))
                    response = pool.apply_async(conduct_Rigetti_experiment, args=(tket_optimizer, optimization_level, False, graph_type, density, j,  k, ))
                    #conduct_Rigetti_experiment(tket_optimizer, optimization_level, True, graph_type, density, j,  k, )
                    #conduct_Rigetti_experiment(tket_optimizer, optimization_level, False, graph_type, density, j,  k, )
    pool.close()
    pool.join()
    print('Rigetti experiments finished.')
    
def conduct_IonQ_experiments(tket_optimizer, optimization_level, relations, graph_types, sample_size):
    print('Begin IonQ co-design experiments.')
    
    pool = Pool()

    for graph_type in graph_types:
        for j in relations:
            for k in range(sample_size):
                response = pool.apply_async(conduct_IonQ_experiment, args=(tket_optimizer, optimization_level, True, graph_type, j,  k, ))
                response = pool.apply_async(conduct_IonQ_experiment, args=(tket_optimizer, optimization_level, False, graph_type, j,  k, ))
                #conduct_IonQ_experiment(tket_optimizer, optimization_level, True, graph_type, j,  k, )
                #conduct_IonQ_experiment(tket_optimizer, optimization_level, False, graph_type, j,  k, )
    pool.close()
    pool.join()
    print('IonQ experiments finished.')


# In[4]:


def parse_transpilation_data(graph_types, relations, densities, vendors, optimisers, gatesets, samples, include_header=True):

    if include_header:
        save_to_csv(['vendor','optimizer','opt_level','query_graph_type','density','gateset','num_relations','sample_index','depth'], 'TranspilationExperiment/Results', 'transpilation_results.txt')
    
    qiskit_circuit_transpilation = config.configuration["qiskit-circuit-transpilation"]
    tket_circuit_transpilation = config.configuration["tket-circuit-transpilation"]
    if qiskit_circuit_transpilation == "retranspile":
        qiskit_result_path_prefix = 'TranspilationExperiment/Results/Retranspiled_Data'
    else:
        qiskit_result_path_prefix = 'TranspilationExperiment/Results/Collected_Data'
    if tket_circuit_transpilation == "retranspile":
        tket_result_path_prefix = 'TranspilationExperiment/Results/Retranspiled_Data'
    else:
        tket_result_path_prefix = 'TranspilationExperiment/Results/Collected_Data'
    
    for vendor in vendors:
        for (optimiser, level) in optimisers:
            for graph_type in graph_types:
                for density in densities:
                    if vendor == "IonQ" and density < 1:
                        continue
                    for gateset in gatesets:
                        for num_relations in relations:
                            for sample in samples:
                                if optimiser == "Tket":
                                    opt_string = get_optimizer_string(True, 0)
                                    result_path_prefix = tket_result_path_prefix
                                else:
                                    opt_string = get_optimizer_string(False, level)
                                    result_path_prefix = qiskit_result_path_prefix
                                path = result_path_prefix + '/' + vendor + '/' + opt_string + '/' + graph_type + '_graph/' + str(density) + '_density/' + gateset + '/' + str(num_relations) + '_relations/sample_' + str(sample)
                                if not os.path.exists(path):
                                    continue
                                depth = load_data(path, 'depth.txt')[0]
                                save_to_csv([vendor,optimiser,level,graph_type,density,gateset,num_relations,sample,depth], 'TranspilationExperiment/Results', 'transpilation_results.txt')


if __name__ == '__main__':      
    
    #graph_types = ['CHAIN', 'STAR', 'CYCLE']
    graph_types = ['CHAIN']
    relations = [4, 5, 6, 7]
    densities = [0.0, 0.05, 0.15, 0.25, 0.5, 0.75, 1.0]
    
    codesign_queries = config.configuration["codesign-queries"]
    if codesign_queries == "generate":
        problem_path_prefix = 'TranspilationExperiment/Problems/JSON'
        SteinbrunnQueryGenerator.produce_steinbrunn_queries_for_codesign(relations, graph_types, range(20), problem_path_prefix)
    
    qiskit_circuit_transpilation = config.configuration["qiskit-circuit-transpilation"]
    tket_circuit_transpilation = config.configuration["tket-circuit-transpilation"]
    
    if qiskit_circuit_transpilation == "retranspile":
        conduct_IBMQ_experiments(False, 3, relations, graph_types, densities, 20)
        conduct_Rigetti_experiments(False, 3, relations, graph_types, densities, 20)
        conduct_IonQ_experiments(False, 3, relations, graph_types, 20)
        
    if tket_circuit_transpilation == "retranspile":
        conduct_IBMQ_experiments(True, 1, relations, graph_types, densities, 20)
        conduct_Rigetti_experiments(True, 1, relations, graph_types, densities, 20)
        conduct_IonQ_experiments(True, 1, relations, graph_types, 20)
    
    vendors = ["IBMQ", "Rigetti", "IonQ"]
    optimisers = [("Tket", 1), ("Qiskit", 3)]
    gatesets = ["Any_gates", "Native_gates"]
    samples = range(20)
    parse_transpilation_data(graph_types, relations, densities, vendors, optimisers, gatesets, samples)

