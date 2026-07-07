"""
Google Cirq Simulation Module
-----------------------------
Author: Michał Ślusarczyk
Date: July 2026

Implements continuous-time quantum walks using Google's Cirq framework,
executing high-precision (complex128) state vector simulations for cross-platform
verification against Qiskit and exact analytical benchmarks.
"""

import numpy as np
import scipy.linalg as la
import cirq

def simulate_cirq_walk(H, start_node, t, num_qubits):
    """
    Simulates unitary evolution U(t) = exp(-i * H * t) within the Cirq ecosystem.
    
    Args:
        H (np.ndarray): Hermitian Hamiltonian matrix.
        start_node (int): Initial vertex index for the walker.
        t (float): Evolution time.
        num_qubits (int): Number of qubits.
        
    Returns:
        tuple: (probs, circuit) where probs is the real probability vector over all vertices.
    """
    U_matrix = la.expm(-1j * H * t)
    qubits = cirq.LineQubit.range(num_qubits)
    ops = []
    
    # Initialize start_node in MSB bit representation
    bin_str = format(start_node, f'0{num_qubits}b')
    for idx, bit in enumerate(bin_str):
        if bit == '1':
            ops.append(cirq.X(qubits[idx]))
            
    # Apply unitary matrix evolution gate
    ops.append(cirq.MatrixGate(U_matrix).on(*qubits))
    
    circuit = cirq.Circuit(ops)
    sim = cirq.Simulator(dtype=np.complex128)
    result = sim.simulate(circuit)
    probs = np.abs(result.final_state_vector) ** 2
    return probs, circuit
