"""
Qiskit Trotter-Suzuki Quantum Algorithm Module
----------------------------------------------
Author: Michał Ślusarczyk
Date: July 2026

Synthesizes explicit Trotter-Suzuki quantum circuits that map weighted edge impedances
w_{uv} = (d_u * d_v)^(-q) directly onto native multi-qubit phase rotation gates without
requiring classical matrix exponentiation. Designed for IBM Quantum platforms.
"""

import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import SparsePauliOp, Statevector
from qiskit.circuit.library import PauliEvolutionGate
from qiskit.synthesis import LieTrotter, SuzukiTrotter

def decompose_hamiltonian_to_paulis(H):
    """
    Decomposes a Hermitian Hamiltonian matrix into a sum of Pauli string operators.
    
    Args:
        H (np.ndarray): Hermitian matrix.
        
    Returns:
        SparsePauliOp: Qiskit operator representation for VQE and Trotter simulation.
    """
    return SparsePauliOp.from_operator(H)

def design_trotterized_quantum_circuit(op, start_node, t, num_qubits, reps=2, order=2):
    """
    Synthesizes a scalable quantum circuit implementing U(t) = exp(-i * H_q * t)
    via Suzuki-Trotter product formula decomposition.
    
    In this circuit design, physical edge weights w_{uv} dictate the exact phase
    rotation angles theta_{uv} = -2 * w_{uv} * dt of multi-qubit Pauli rotation gates.
    
    Args:
        op (SparsePauliOp): Pauli Hamiltonian representation.
        start_node (int): Initial vertex index for the quantum walker.
        t (float): Evolution time.
        num_qubits (int): Number of qubits (N = 2^num_qubits).
        reps (int): Number of Trotter repetitions (depth parameter M).
        order (int): Suzuki-Trotter formula order (1 for LieTrotter, 2 for SuzukiTrotter).
        
    Returns:
        tuple: (probs, circuit_depth, gate_counts, full_circuit)
    """
    evol_gate = PauliEvolutionGate(op, time=t)
    synth = SuzukiTrotter(order=order, reps=reps) if order > 1 else LieTrotter(reps=reps)
    
    qc_decomp = synth.synthesize(evol_gate)
    
    # Assemble full circuit with initial state preparation
    qc_full = QuantumCircuit(num_qubits)
    bin_str = format(start_node, f'0{num_qubits}b')
    for idx, bit in enumerate(reversed(bin_str)):
        if bit == '1':
            qc_full.x(idx)
            
    qc_full.compose(qc_decomp, inplace=True)
    
    sv = Statevector(qc_full)
    probs = np.abs(sv.data) ** 2
    return probs, qc_full.depth(), qc_decomp.count_ops(), qc_full
