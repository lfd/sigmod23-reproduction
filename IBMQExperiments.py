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
import Scripts.Postprocessing as Postprocessing
from multiprocessing import Pool
import csv
import numpy as np
import pickle
from decimal import *

from qiskit.algorithms.optimizers import AQGD
from qiskit.algorithms import QAOA
from qiskit_optimization.algorithms import MinimumEigenOptimizer
from qiskit.utils import QuantumInstance
from qiskit import IBMQ


# In[2]:


def get_rounded_val(val, decimal_pos = 4):
    return Decimal(val).quantize(Decimal(10) ** -decimal_pos, rounding=ROUND_HALF_EVEN)

def save_results(path_string, depths):

    datapath = os.path.abspath(path_string)
    pathlib.Path(datapath).mkdir(parents=True, exist_ok=True) 
    
    datafile = os.path.abspath(path_string + '/depth.txt')
    mode = 'a' if os.path.exists(datafile) else 'w'
    with open(datafile, mode) as file:
        json.dump(depths, file)
        
def pickle_results(path_string, results):

    datapath = os.path.abspath(path_string)
    pathlib.Path(datapath).mkdir(parents=True, exist_ok=True) 
    
    datafile = os.path.abspath(path_string + '/results.txt')
    with open(datafile, 'wb') as file:
        pickle.dump(results, file)
        
def load_pickled_result(path_string):
    datapath = os.path.abspath(path_string)
    datafile = os.path.abspath(path_string + '/results.txt')
    results = None
    with open(datafile, 'rb') as file:
        results = pickle.load(file)
    return results

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

def get_QAOA_circuit(qubo):
    circuit = CircuitGenerator.create_QAOA_circuit(qubo)
    circuit = CircuitGenerator.decompose_circuit(circuit)
    return circuit

def determine_circuit_depth(qubo, coupling_map, tket_optimizer, optimization_level):
    circuit = get_QAOA_circuit(qubo)
    if tket_optimizer:
        circuit = CircuitGenerator.transpile_circuit_with_tket(circuit, coupling_map, get_tket_IBMQ_gateset_pass(True))
    else:
        circuit = CircuitGenerator.transpile_circuit_with_qiskit(circuit, coupling_map, optimization_level, get_qiskit_IBMQ_gateset(True))
    return circuit.depth()


# In[3]:


# Returns a simulator backend for testing the implementation
def get_IBMQ_QASM_backend():
    provider = IBMQ.get_provider(hub='ibm-q', group='open', project='main')
    backend = provider.get_backend('ibmq_qasm_simulator')
    quantum_instance = QuantumInstance(backend=backend)
    return quantum_instance

def get_IBMQ_backend():
    # The provider needs to enable access to the ibm_auckland, or a similar 27-qubit, QPU (this is not the case for the default "open" group)
    provider = IBMQ.get_provider(hub='ibm-q', group='open', project='main')
    backend = provider.get_backend('ibm_auckland')
    quantum_instance = QuantumInstance(backend=backend)
    return quantum_instance

def solve_with_QAOA(qubo, iterations, reps=1):
    optimizer = AQGD(maxiter=iterations)
    qaoa_meas = QAOA(optimizer=optimizer, quantum_instance=get_IBMQ_backend(), reps=reps, initial_point=[0., 0.])
    qaoa = MinimumEigenOptimizer(qaoa_meas)
    qaoa_result = qaoa.solve(qubo)
    return qaoa_result

def conduct_IBMQ_QPU_experiments():
    
    IBMQ.load_account()
    
    iterations_categories = [20, 50]
    thres = [10]
    num_decimal_pos = 0
    optimal_solution = 0
    
    for iterations in iterations_categories:
        for i in range(4):
            
            card, pred, pred_sel = ProblemGenerator.get_join_ordering_problem('ExperimentalAnalysis/IBMQ/QPUPerformance/Problems/JSON/' + str(i) + '_predicates', generated_problems=False)
            qubo, weight_a = QUBOGenerator.generate_QUBO_for_IBMQ(card, thres, num_decimal_pos, pred, pred_sel)

            response = solve_with_QAOA(qubo, iterations)
            pickle_results('ExperimentalAnalysis/IBMQ/QPUPerformance/Results/Data/' + str(iterations) + '_Iterations/' + str(i) + '_predicates', response)

def conduct_IBMQ_transpilation_experiments(tket_optimizer, optimization_level, sample_size = 20):
    for i in range(4):
        for k in range(sample_size):
            ## Experiments for varying predicate numbers, for the IBMQ Auckland topology
            qubo = ProblemGenerator.get_join_ordering_qubo('ExperimentalAnalysis/IBMQ/Embeddings/Problems/QUBO/predicate_variation/' + str(i) + '_predicates')
            coupling_map = TopologyGenerator.get_IBM_Mumbai_topology()
            depth = determine_circuit_depth(qubo, coupling_map, tket_optimizer, optimization_level)
            path_string = 'ExperimentalAnalysis/IBMQ/Embeddings/Results/Data/predicate_variation/Auckland_topology/' + get_optimizer_string(tket_optimizer, optimization_level) + '/' + str(i) + '_predicates/sample_' + str(k)
            save_results(path_string, [depth])
            
            ## Experiments for varying cont. variable discretization precisions
            qubo = ProblemGenerator.get_join_ordering_qubo('ExperimentalAnalysis/IBMQ/Embeddings/Problems/QUBO/precision_variation/' + str(i) + '_decpos')
            coupling_map = TopologyGenerator.get_IBM_Mumbai_topology()
            depth = determine_circuit_depth(qubo, coupling_map, tket_optimizer, optimization_level)
            path_string = 'ExperimentalAnalysis/IBMQ/Embeddings/Results/Data/precision_variation/Auckland_topology/' + get_optimizer_string(tket_optimizer, optimization_level) + '/' + str(i) + '_decpos/sample_' + str(k)
            save_results(path_string, [depth])
            
            ## Experiments for varying predicate numbers, for the IBMQ Auckland topology
            qubo = ProblemGenerator.get_join_ordering_qubo('ExperimentalAnalysis/IBMQ/Embeddings/Problems/QUBO/predicate_variation/' + str(i) + '_predicates')
            coupling_map = TopologyGenerator.get_IBM_Washington_topology()
            depth = determine_circuit_depth(qubo, coupling_map, tket_optimizer, optimization_level)
            path_string = 'ExperimentalAnalysis/IBMQ/Embeddings/Results/Data/predicate_variation/Washington_topology/' + get_optimizer_string(tket_optimizer, optimization_level) + '/' + str(i) + '_predicates/sample_' + str(k)
            save_results(path_string, [depth])
            
            ## Experiments for varying cont. variable discretization precisions
            qubo = ProblemGenerator.get_join_ordering_qubo('ExperimentalAnalysis/IBMQ/Embeddings/Problems/QUBO/precision_variation/' + str(i) + '_decpos')
            coupling_map = TopologyGenerator.get_IBM_Washington_topology()
            depth = determine_circuit_depth(qubo, coupling_map, tket_optimizer, optimization_level)
            path_string = 'ExperimentalAnalysis/IBMQ/Embeddings/Results/Data/precision_variation/Washington_topology/' + get_optimizer_string(tket_optimizer, optimization_level) + '/' + str(i) + '_decpos/sample_' + str(k)
            save_results(path_string, [depth])


# In[4]:


def save_to_csv(data, path, filename):
    
    sd = os.path.abspath(path)
    pathlib.Path(sd).mkdir(parents=True, exist_ok=True) 
    
    f = open(path + '/' + filename, 'a', newline='')
    writer = csv.writer(f)
    writer.writerow(data)
    f.close()

def process_data(depths):
    minimum_depth = np.amin(depths)
    mean_depth = int(np.ceil(np.mean(depths)))
    median_depth = int(np.ceil(np.median(depths)))
    maximum_depth = np.amax(depths)
    return minimum_depth, mean_depth, median_depth, maximum_depth

def parse_QPU_data():
    iterations_categories = [20, 50]
    thres_vals = {0: [100], 1: [10], 2: [10], 3: [10]}
    
    for iterations in iterations_categories:
        for i in range(4):
            card, pred, pred_sel = ProblemGenerator.get_join_ordering_problem('ExperimentalAnalysis/IBMQ/QPUPerformance/Problems/JSON/' + str(i) + '_predicates', generated_problems=False)
            response = load_pickled_result('ExperimentalAnalysis/IBMQ/QPUPerformance/Results/Data/' + str(iterations) + '_Iterations/' + str(i) + '_predicates')
            best_join_order, best_join_order_costs, valid_ratio, optimal_ratio = Postprocessing.postprocess_IBMQ_response(response, card, pred, pred_sel, thres_vals[i])
            save_to_csv([iterations, i, get_rounded_val(valid_ratio), get_rounded_val(optimal_ratio)], 'ExperimentalAnalysis/IBMQ/QPUPerformance/Results', 'results.txt')

            
def parse_transpilation_data(aggregate_results = False):
    optimizers = ["Tket", "Qiskit"]
    topologies = ["Auckland", "Washington"]
    opt_levels = [1, 2, 3]
    qubits = [18, 21, 24, 27]
    for optimizer in optimizers:
        for i in range(4):
            for opt_level in opt_levels:
                optimizer_string = None
                samplesize = 0
                if optimizer == "Tket" and opt_level != 1:
                    continue
                if optimizer == "Tket":
                    optimizer_string = get_optimizer_string(True, 1)
                    samplesize = 1
                else:
                    optimizer_string = get_optimizer_string(False, opt_level)
                    samplesize = 20
                    
                for topology in topologies:
                    depths = []
                    for j in range(samplesize):
                        path_string = 'ExperimentalAnalysis/IBMQ/Embeddings/Results/Data/predicate_variation/' + topology + '_topology/' + optimizer_string + '/' + str(i) + '_predicates/sample_' + str(j)
                        depth = load_result(path_string)[0]
                        if aggregate_results:
                            depths.append(depth)
                        else:
                            save_to_csv([optimizer, opt_level, i, qubits[i], topology, 'predicates', j, depth], 'ExperimentalAnalysis/IBMQ/Embeddings/Results', 'depths.txt')

                    if aggregate_results:
                        minimum_depth, mean_depth, median_depth, maximum_depth = process_data(depths)
                        save_to_csv([optimizer, opt_level, i, qubits[i], topology, 'predicates', minimum_depth, mean_depth, median_depth, maximum_depth], 'ExperimentalAnalysis/IBMQ/Embeddings/Results', 'aggregated_depths.txt')

                    depths = []
                    for j in range(samplesize):
                        path_string = 'ExperimentalAnalysis/IBMQ/Embeddings/Results/Data/precision_variation/' + topology + '_topology/' + optimizer_string + '/' + str(i) + '_decpos/sample_' + str(j)
                        depth = load_result(path_string)[0]
                        if aggregate_results:
                            depths.append(depth)
                        else:
                            save_to_csv(aggregate_results, [optimizer, opt_level, i, qubits[i], topology, 'precision', j, depth])

                    if aggregate_results:
                        minimum_depth, mean_depth, median_depth, maximum_depth = process_data(depths)
                        save_to_csv(aggregate_results, [optimizer, opt_level, i, qubits[i], topology, 'precision', minimum_depth, mean_depth, median_depth, maximum_depth])


# In[5]:


#conduct_IBMQ_experiments(True, 1, sample_size = 1)
#conduct_IBMQ_experiments(False, 1, sample_size = 20)
#conduct_IBMQ_experiments(False, 2, sample_size = 20)
#conduct_IBMQ_experiments(False, 3, sample_size = 20)
conduct_IBMQ_transpilation_experiments(False, 1, sample_size = 20)
parse_transpilation_data()
conduct_IBMQ_QPU_experiments()
parse_QPU_data()

