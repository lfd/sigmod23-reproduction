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
import numpy as np


# In[2]:


def save_results(vendor, optimizer, querygraph, density, restrict_to_native_gates, relations, sample, depths, savings_dir = 'TranspilationExperiment/Results/Data'):
    gateset = None
    if restrict_to_native_gates:
        gateset = 'Native_gates'
    else:
        gateset = 'Any_gates'
    path_string = savings_dir + '/' + vendor + '/' + optimizer + '/' + querygraph + '_graph/' + str(density) + '_density/' + gateset + '/' + str(relations) + '_relations/sample_' + str(sample)

    datapath = os.path.abspath(path_string)
    pathlib.Path(datapath).mkdir(parents=True, exist_ok=True) 
    
    datafile = os.path.abspath(path_string + '/depth.txt')
    mode = 'a' if os.path.exists(datafile) else 'w'
    with open(datafile, mode) as file:
        json.dump(depths, file)
        
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
    card, pred, pred_sel = ProblemGenerator.get_join_ordering_problem('TranspilationExperiment/Problems/JSON/' + str(num_relations) + 'relations/' + str(graph_type) + '_graph/' + str(sample_index))
    thres = [100, 10000]
    num_decimal_pos = 0
    return card, thres, num_decimal_pos, pred, pred_sel
        
def get_QUBO(num_relations, graph_type, sample_index, load_saved_qubo = True):
    if load_saved_qubo:
        return ProblemGenerator.get_join_ordering_qubo('TranspilationExperiment/Problems/QUBO/' + str(num_relations) + 'relations/' + str(graph_type) + '_graph/' + str(sample_index))
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


def conduct_IBMQ_experiments(tket_optimizer, optimization_level, relations, graph_types, densities, sample_size = 20):
    print('Begin IBMQ experiments.')
    
    pool = Pool()

    for graph_type in graph_types:
        for density in densities:
            for j in relations:
                for k in range(sample_size):
                    pool.apply_async(conduct_IBMQ_experiment, args=(tket_optimizer, optimization_level, True, graph_type, density, j,  k, ))
                    pool.apply_async(conduct_IBMQ_experiment, args=(tket_optimizer, optimization_level, False, graph_type, density, j,  k, ))
    
    pool.close()
    pool.join()
    print('IBMQ experiments finished.')
    
def conduct_Rigetti_experiments(tket_optimizer, optimization_level, relations, graph_types, densities, sample_size = 20):
    print('Begin Rigetti experiments.')
    
    pool = Pool()

    for graph_type in graph_types:
        for density in densities:
            for j in relations:
                for k in range(sample_size):
                    pool.apply_async(conduct_Rigetti_experiment, args=(tket_optimizer, optimization_level, True, graph_type, density, j,  k, ))
                    pool.apply_async(conduct_Rigetti_experiment, args=(tket_optimizer, optimization_level, False, graph_type, density, j,  k, ))
                   
    pool.close()
    pool.join()
    print('Rigetti experiments finished.')
    
def conduct_IonQ_experiments(tket_optimizer, optimization_level, relations, graph_types, sample_size = 20):
    print('Begin IonQ experiments.')
    
    pool = Pool()

    for graph_type in graph_types:
        for j in relations:
            for k in range(sample_size):
                pool.apply_async(conduct_IonQ_experiment, args=(tket_optimizer, optimization_level, True, graph_type, j,  k, ))
                pool.apply_async(conduct_IonQ_experiment, args=(tket_optimizer, optimization_level, False, graph_type, j,  k, ))
                
    pool.close()
    pool.join()
    print('IonQ experiments finished.')


# In[4]:


if __name__ == '__main__':      
    
    graph_types = ['CHAIN', 'STAR', 'CYCLE']
    relations = [4, 5, 6, 7]
    densities = [0.0, 0.05, 0.15, 0.25, 0.5, 0.75, 1.0]
    
    conduct_IBMQ_experiments(False, 1, relations, graph_types, densities)
    conduct_IBMQ_experiments(False, 3, min_number_of_relations, max_number_of_relations, graph_types, densities)
    conduct_Rigetti_experiments(True, 1, min_number_of_relations, max_number_of_relations, graph_types, densities)
    conduct_Rigetti_experiments(False, 3, min_number_of_relations, max_number_of_relations, graph_types, densities)
    conduct_IonQ_experiments(True, 1, min_number_of_relations, max_number_of_relations, graph_types)
    conduct_IonQ_experiments(False, 3, min_number_of_relations, max_number_of_relations, graph_types)

