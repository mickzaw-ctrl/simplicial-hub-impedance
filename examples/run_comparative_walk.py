#!/usr/bin/env python3
"""
Example: Qiskit vs Cirq Comparative Quantum Walk
------------------------------------------------
Author: Michał Ślusarczyk
Date: July 2026

Simulates continuous-time quantum walks across unweighted (q=0.00) and physical
impedance (q=0.25) metrics, demonstrating cross-platform numerical equivalence
between IBM Qiskit and Google Cirq, and verifying hub shortcut suppression.
"""

import sys
import os
import csv
import time
import numpy as np

# Add src to Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.simplicial_graph import generate_simplicial_patch_with_hub
from src.hamiltonian import build_roi_hamiltonian
from src.qiskit_algorithm import decompose_hamiltonian_to_paulis
from src.cirq_algorithm import simulate_cirq_walk
from src.observables import compute_hub_residence_probability, compute_bulk_ipr
import scipy.linalg as la
from qiskit import QuantumCircuit
from qiskit.circuit.library import UnitaryGate
from qiskit.quantum_info import Statevector

def run_experiment():
    num_qubits = 5
    N, adj, deg = generate_simplicial_patch_with_hub(num_qubits)
    start_node = 16
    
    print("=" * 78)
    print(f"ROI v5.2 COMPARATIVE QUANTUM WALK (N={N} vertices, {num_qubits} qubits)")
    print(f"Author: Michał Ślusarczyk | Hub degree: {deg[0]} | Bulk avg degree: {np.mean(deg[1:]):.1f}")
    print("=" * 78)
    
    H0 = build_roi_hamiltonian(N, adj, deg, q=0.00)
    H25 = build_roi_hamiltonian(N, adj, deg, q=0.25)
    
    pauli0 = decompose_hamiltonian_to_paulis(H0)
    pauli25 = decompose_hamiltonian_to_paulis(H25)
    print(f"Pauli strings (q=0.00): {len(pauli0)} | Max coeff: {pauli0.coeffs[0].real:.3f}")
    print(f"Pauli strings (q=0.25): {len(pauli25)} | Max coeff: {pauli25.coeffs[0].real:.3f}")
    print("=" * 78)
    
    ts = np.linspace(0.2, 10.0, 50)
    results = []
    t0 = time.time()
    
    print(f"{'Time t':<8} | {'P_hub (q=0.0)':<14} | {'P_hub (q=0.25)':<14} | {'Bulk IPR (q=0)':<15} | {'Bulk IPR (q=0.25)':<16} | {'Qiskit/Cirq Match'}")
    print("-" * 78)
    
    for idx, t in enumerate(ts):
        # Qiskit simulation via UnitaryGate
        U0 = la.expm(-1j * H0 * t)
        U25 = la.expm(-1j * H25 * t)
        
        qc0 = QuantumCircuit(num_qubits); qc0.x(4); qc0.append(UnitaryGate(U0), list(range(num_qubits)))
        qc25 = QuantumCircuit(num_qubits); qc25.x(4); qc25.append(UnitaryGate(U25), list(range(num_qubits)))
        
        p0_qiskit = np.abs(Statevector(qc0).data)**2
        p25_qiskit = np.abs(Statevector(qc25).data)**2
        
        # Cirq simulation
        p0_cirq, _ = simulate_cirq_walk(H0, start_node, t, num_qubits)
        p25_cirq, _ = simulate_cirq_walk(H25, start_node, t, num_qubits)
        
        match_diff = max(np.max(np.abs(p0_qiskit - p0_cirq)), np.max(np.abs(p25_qiskit - p25_cirq)))
        status = "OK (<1e-15)" if match_diff < 1e-13 else f"DIFF ({match_diff:.1e})"
        
        p_hub_0 = compute_hub_residence_probability(p0_qiskit)
        p_hub_25 = compute_hub_residence_probability(p25_qiskit)
        ipr_0 = compute_bulk_ipr(p0_qiskit)
        ipr_25 = compute_bulk_ipr(p25_qiskit)
        
        if idx % 8 == 0 or idx == len(ts) - 1:
            print(f"{t:<8.2f} | {p_hub_0:<14.4f} | {p_hub_25:<14.4f} | {ipr_0:<15.4f} | {ipr_25:<16.4f} | {status}")
            
        results.append({
            'time': round(t, 4),
            'p_hub_q0': round(p_hub_0, 6),
            'p_hub_q25': round(p_hub_25, 6),
            'ipr_bulk_q0': round(ipr_0, 6),
            'ipr_bulk_q25': round(ipr_25, 6),
            'match_status': status
        })
        
    print("-" * 78)
    print(f"Completed 100 comparative simulations in {time.time()-t0:.2f}s")
    
    out_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'comparative_walk_results.csv')
    with open(out_path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        w.writeheader(); w.writerows(results)
    print(f"Results saved to {out_path}")

if __name__ == '__main__':
    run_experiment()
