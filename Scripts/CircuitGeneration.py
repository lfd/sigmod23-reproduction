#!/usr/bin/env python
# coding: utf-8

# In[1]:


from docplex.mp.model import Model
import qiskit
from qiskit.algorithms import QAOA
from qiskit.circuit.library.n_local.qaoa_ansatz import QAOAAnsatz
from qiskit.compiler import transpile
from qiskit_optimization.translators import from_docplex_mp
from qiskit.transpiler import CouplingMap

from pytket import Circuit, OpType
from pytket.architecture import Architecture
from pytket.extensions.qiskit import qiskit_to_tk, IBMQBackend
from pytket.passes import FullPeepholeOptimise, DefaultMappingPass, RemoveRedundancies
from pytket.extensions.ionq import IonQBackend
from pytket.extensions.ionq.backends.config import set_ionq_config
from pytket.extensions.qiskit.backends.config import set_ibmq_config

import pyquil
from pytket.extensions.pyquil import ForestBackend


# In[2]:


def create_QAOA_circuit(qubo, reps=1):
    qaoa = qiskit.algorithms.QAOA(initial_point=[0., 0.])
    op, offset = qubo.to_ising()
    ansatz = QAOAAnsatz(op, 1).decompose()
    circuit = qaoa.construct_circuit(ansatz.parameters, op)[0]
    return circuit

# Decompose the circuit to its basis gates
def decompose_circuit(circuit, max_unchanged_repetitions=3):
    unchanged_counter = 0
    while unchanged_counter < max_unchanged_repetitions:
        old_depth = circuit.depth()
        circuit = circuit.decompose()
        if old_depth == circuit.depth():
            # Increase the counter if the circuit has not changed
            unchanged_counter = unchanged_counter + 1
        else:
            # Otherwise reset the counter
            unchanged_counter = 0
    return circuit


# In[3]:


def get_IBMQ_basis_gates():
    return ['id', 'rz', 'sx', 'x', 'cx']

def get_Rigetti_basis_gates():
    return ['rx', 'rz', 'cz']

def get_IonQ_basis_gates():
    return ['rxx', 'ryy', 'z', 'rzz', 'x', 'y', 's', 'sdg', 't', 'tdg', 'h', 'rx', 'ry', 'rz', 'cx', 'swap']


# In[4]:


def apply_tket_passes(circuit, coupling_map, rebase_pass, architecture):
        
    FullPeepholeOptimise().apply(circuit)

    DefaultMappingPass(architecture).apply(circuit)

    if rebase_pass is not None:
        rebase_pass.apply(circuit)

    RemoveRedundancies().apply(circuit)
    
    return circuit

def create_pytket_architecture(qpu_coupling):
    pytket_coupling = []
    for qubitlist in qpu_coupling:
        pytket_coupling.append((qubitlist[0], qubitlist[1]))
    return Architecture(pytket_coupling)

def transpile_circuit_with_tket(circuit, coupling_map, rebase_pass):
    circuit = qiskit_to_tk(circuit)
    architecture = create_pytket_architecture(coupling_map)
    return apply_tket_passes(circuit.copy(), coupling_map, rebase_pass, architecture)

def transpile_circuit_with_qiskit(circuit, coupling_map, optimization_level, basis_gates):
    if basis_gates == None:
        return transpile(circuit, coupling_map=coupling_map, optimization_level=optimization_level)
    else:
        return transpile(circuit, coupling_map=coupling_map, optimization_level=optimization_level, basis_gates=basis_gates)
    
def fetch_Rigetti_rebase_pass():
    qc = pyquil.get_qc('Aspen-M-1')
    backend = ForestBackend(qc)
    rigetti_rebase_pass = backend.rebase_pass()
    return rigetti_rebase_pass
    
def fetch_IonQ_rebase_pass():
    set_ionq_config('abcd') # Set IonQ API token
    backend = IonQBackend(device_name='IonQBackend')
    ionq_rebase_pass = backend.rebase_pass()
    return ionq_rebase_pass
    
def fetch_IBMQ_rebase_pass():
    set_ibmq_config(ibmq_api_token='abcd') # Set IBMQ API token
    backend = IBMQBackend(backend_name='ibmq_manila', hub='ibm-q', group='open', project='main')
    ibmq_rebase_pass = backend.rebase_pass()
    return ibmq_rebase_pass

