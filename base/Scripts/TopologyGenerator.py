#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import itertools
import json
import os
import pathlib 
import pickle
from iteration_utilities import duplicates, unique_everseen

import networkx as nx
import pyquil


# In[2]:


def load_data(filename, directory):
    data_file = os.path.abspath(directory + "/" + filename)
    
    cd = os.path.abspath(directory)
    pathlib.Path(cd).mkdir(parents=True, exist_ok=True)
    
    data = None
    with open(data_file, 'rb') as file:
        data = pickle.load(file)
    return data

def load_coupling(filename, vendor):
    return load_data(filename, 'couplings/' + vendor)


# In[3]:


def get_qubits_for_coupling_map(coupling_map):
    return set(sum(coupling_map, []))

def format_qubit_labels_to_consecutive_integers(coupling_map):
    g = get_nx_graph(coupling_map)
    g = nx.convert_node_labels_to_integers(g)
    coupling_map = []
    for (i, j) in g.edges():
        coupling_map.append([i, j])
        coupling_map.append([j, i])
    return coupling_map

def get_nx_graph(base_map):
    qubits = get_qubits_for_coupling_map(base_map)
    g = nx.Graph()
    g.add_nodes_from(list(qubits))
    g.add_edges_from(base_map)
    return g

def get_eligible_edges(number_of_nodes, coupling_map, coupling_map_distances, distance):
    qubits = set(sum(coupling_map, []))
    eligible_edges = []
    for (i, j) in itertools.combinations(qubits, 2):
        if coupling_map_distances[i][j] == distance:
            eligible_edges.append((i, j))
    return eligible_edges

def add_qubit_connection(coupling_map, index1, index2):
    coupling_map.append([index1, index2])
    coupling_map.append([index2, index1])
    return coupling_map

def repair_IBMQ_Washington_topology(coupling_map):
    ## Repairing the broken connections for the IBM-Q Washington QPU
    coupling_map = add_qubit_connection(coupling_map, 8, 9)
    coupling_map = add_qubit_connection(coupling_map, 109, 114)
    return coupling_map

# The heavy hex lattice topology consists of a sequence of rows with 3 qubit cells. These cells can be decomposed in an top and bottom line of qubits,
# as well as intermediate qubits connecting the top and bottom row, and thereby serving as the borders of the unit cells.
# The index of qubits in the top and bottom row, which the intermediate qubits connect to, change in an alternating sequence.
# The shift offset variable accounts for this change in accordance to the required qubit connections.
def add_intermediate_qubits_connecting_to_next_line(coupling_map, leftmost_qubit_index, new_qubit_index, shift_offset, starting_index=0):
    for i in range(starting_index, 4):
        # The index for the qubit in the row above
        upper_adj_index = leftmost_qubit_index + (4*i) + shift_offset
        coupling_map = add_qubit_connection(coupling_map, upper_adj_index, new_qubit_index)
        # The index for the qubit in the row below
        lower_adj_index = new_qubit_index + (3-i) + (i*4) + 1 + shift_offset
        coupling_map = add_qubit_connection(coupling_map, new_qubit_index, lower_adj_index)
        new_qubit_index += 1
    return coupling_map

def add_new_line_of_qubits(coupling_map, leftmost_qubit_index, qubit_line_length = 15):
    current_qubit = leftmost_qubit_index
    for i in range(qubit_line_length-1):
        coupling_map = add_qubit_connection(coupling_map, current_qubit, current_qubit+1)
        current_qubit += 1
    return coupling_map

def add_IBMQ_Washington_heavy_hex_rows(coupling_map, extra_rows):
    repair_IBMQ_Washington_topology(coupling_map)
    
    ## Finish bottom line by adding missing qubit to the left
    coupling_map = add_qubit_connection(coupling_map, 113, 127)
    
    new_qubit_index = 128
    
    ## Add edges connecting to the next line of qubits
    
    # Since the index of the newly added qubit is an exception, the connections to the intermediate qubit connecting to the
    # added qubit and the next line of qubits needs to be added "manually"
    coupling_map = add_qubit_connection(coupling_map, 127, new_qubit_index)
    coupling_map = add_qubit_connection(coupling_map, new_qubit_index, new_qubit_index+4)
    new_qubit_index += 1
    
    ## For the addition of the first new row, we just assume the leftmost qubit of the current bottom line has index 112
    ## (in reality, the qubit corresponds to the one that was just added, which has index 127)
    leftmost_qubit_index = 112
    shift_offset = 0
    # First, we add the connections from the current bottom row to the set of intermediate qubits
    coupling_map = add_intermediate_qubits_connecting_to_next_line(coupling_map, leftmost_qubit_index, new_qubit_index, shift_offset, starting_index=1)
    new_qubit_index = new_qubit_index + 3
    # Next, we add the next line of qubits, which becomes the new bottom line
    leftmost_qubit_index = new_qubit_index
    coupling_map = add_new_line_of_qubits(coupling_map, new_qubit_index)
    new_qubit_index = new_qubit_index + 15
    
    # After the first row addition, which has to account for the exceptional index of the added qubit, the following code
    # adds an arbitrary number of additional rows to the coupling map
    for i in range(extra_rows-1):
        if shift_offset == 0:
            shift_offset = 2
        else:
            shift_offset = 0
        coupling_map = add_intermediate_qubits_connecting_to_next_line(coupling_map, leftmost_qubit_index, new_qubit_index, shift_offset)
        new_qubit_index = new_qubit_index + 4
        leftmost_qubit_index = new_qubit_index
        coupling_map = add_new_line_of_qubits(coupling_map, new_qubit_index)
        new_qubit_index = new_qubit_index + 15
        
    ## Finally, since the bottom (and top) lines typically have one qubit less than the intermediate lines, one qubit needs to be
    ## removed from the coupling map. Depending on the current position in the row sequence, either the leftmost or the rightmost
    ## qubit needs to be removed.
    if shift_offset == 0:
        coupling_map.remove([new_qubit_index-2, new_qubit_index-1])
        coupling_map.remove([new_qubit_index-1, new_qubit_index-2])
    else: 
        coupling_map.remove([leftmost_qubit_index, leftmost_qubit_index+1])
        coupling_map.remove([leftmost_qubit_index+1, leftmost_qubit_index])
    return coupling_map

# Extends the given Rigetti couplings by the specified number of extra columns.
# Note: This method expects the nodes to be labeled in accordance to the scheme used by Rigetti.
# Relabeling the nodes (for qiskit transpilation) should therefore be done after calling this function.
def add_additional_Rigetti_columns(coupling_map, extra_columns):
    num_qubits = len(get_qubits_for_coupling_map(coupling_map))
    num_lines = 2
    row_offset = 100
    col_offset = 10
    num_octagons_per_line = int(num_qubits / (8*num_lines))
    dec_counter = 0
    
    lower_pred_id = col_offset * (num_octagons_per_line-1) + 1
    upper_pred_id = col_offset * (num_octagons_per_line-1) + 2
        
    for j in range(extra_columns):
        if (j + num_octagons_per_line) % 10 == 0:
            dec_counter += 1
        offset = col_offset * ((num_octagons_per_line+j) % 10) + 1000 * dec_counter
        
        add_qubit_connection(coupling_map, lower_pred_id, offset+6)
        add_qubit_connection(coupling_map, upper_pred_id, offset+5)
        add_qubit_connection(coupling_map, lower_pred_id + row_offset, offset+6+row_offset)
        add_qubit_connection(coupling_map, upper_pred_id + row_offset, offset+5+row_offset)
         
        # Add 2 new octagons representing one new column
        for i in range(offset, offset+7):
            add_qubit_connection(coupling_map, i, i+1)
            add_qubit_connection(coupling_map, i+row_offset, i+1+row_offset)
            
        lower_pred_id = offset + 1
        upper_pred_id = offset + 2

        # Add connections between the new octagons
        add_qubit_connection(coupling_map, offset, offset+3+row_offset)
        add_qubit_connection(coupling_map, offset+7, offset+4+row_offset)
    
    return coupling_map


# In[4]:


def increase_coupling_density(base_map, density):
    
    if density == 0:
        return base_map
    
    number_of_nodes = len(set(sum(base_map, [])))
    ## Note: IBM-Q requires a list of directed edges as a coupling map
    max_edges = (number_of_nodes)*(number_of_nodes-1)
    coupling_map = base_map.copy()
    
    g = get_nx_graph(coupling_map)
    coupling_map_distances = dict(nx.all_pairs_shortest_path_length(g))
    distance = 2 # Start with edges connecting nodes of distance 2
    eligible_edges = get_eligible_edges(number_of_nodes, base_map, coupling_map_distances, distance)
    
    while (len(coupling_map)/max_edges) < density:
        nodes = eligible_edges[np.random.choice(np.arange(len(eligible_edges)), size=1, replace=False)[0]]
        coupling_map.append([nodes[0], nodes[1]])
        coupling_map.append([nodes[1], nodes[0]])
        eligible_edges.remove(nodes)
        # Check if no more elements of the current minimum distance exist
        if not eligible_edges:
            # If so, increase the considered distance and fetch a new list of eligible edges
            distance = distance + 1
            eligible_edges = get_eligible_edges(number_of_nodes, base_map, coupling_map_distances, distance)
        
    return coupling_map

def get_IonQ_topology(num_of_qubits):
    coupling_map = []
    for (i, j) in itertools.combinations(range(num_of_qubits), 2):
        coupling_map.append([i, j])
        coupling_map.append([j, i])
    return coupling_map

def get_Rigetti_Aspen_M_topology(density=0, extra_columns=0):
    coupling_map = load_coupling('AspenM1.txt', 'Rigetti')
    return coupling_map

def get_IBMQ_topology(filename, vendor, density, extra_rows):
    coupling_map = load_coupling(filename, vendor)
    
    if extra_rows > 0:
        coupling_map = add_IBMQ_Washington_heavy_hex_rows(coupling_map, extra_rows)
        
    if density > 0:
        coupling_map = increase_coupling_density(coupling_map, density)
        
    return coupling_map
    
def get_IBM_Washington_topology(density=0, extra_rows=0):
    return get_IBMQ_topology('washington_coupling.txt', 'IBMQ', density, extra_rows)

def get_IBM_Mumbai_topology(density=0, extra_rows=0):
    return get_IBMQ_topology('mumbai_coupling.txt', 'IBMQ', density, extra_rows)

def expand_Rigetti_Aspen_M_topology(base_map, required_qubits):
    extra_columns = 0
    coupling_map = base_map.copy()
    while len(get_qubits_for_coupling_map(coupling_map)) < required_qubits:
        extra_columns += 1
        coupling_map = add_additional_Rigetti_columns(base_map.copy(), extra_columns)
    return coupling_map

def expand_IBM_Washington_topology(base_map, required_qubits):
    extra_rows = 0
    coupling_map = base_map.copy()
    while len(get_qubits_for_coupling_map(coupling_map)) < required_qubits:
        extra_rows += 1
        coupling_map = add_IBMQ_Washington_heavy_hex_rows(base_map.copy(), extra_rows)
    return coupling_map

