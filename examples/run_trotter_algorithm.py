#!/usr/bin/env python3
"""
Example: Qiskit Trotter-Suzuki Algorithm Design
-----------------------------------------------
Author: Michał Ślusarczyk
Date: July 2026

Synthesizes explicit Trotter-Suzuki quantum circuits that encode physical edge
impedances w_{uv} = (d_u * d_v)^(-q) as unitary phase rotation angles.
Evaluates circuit depth, CNOT count, and convergence against exact evolution.
"""

import sys
import os
import csv
import time
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.simplicial_graph import generate_simplicial_patch_with_hub
from src.hamiltonian import build_roi_hamiltonian, compute_exact_unitary_evolution
from src.qiskit_algorithm import decompose_hamiltonian_to_paulis, design_trotterized_quantum_circuit
from src.observables import compute_hub_residence_probability, compute_bulk_ipr

def run_trotter_experiment():
    num_qubits = 5
    N, adj, deg = generate_simplicial_patch_with_hub(num_qubits)
    start_node = 16
    t_eval = 2.0
    
    print("=" * 78)
    print("QISKIT TROTTER-SUZUKI QUANTUM ALGORITHM DESIGN")
    print(f"Author: Michał Ślusarczyk | 5 qubits (N={N} vertices) | t={t_eval}")
    print("=" * 78)
    
    H0 = build_roi_hamiltonian(N, adj, deg, q=0.00)
    H25 = build_roi_hamiltonian(N, adj, deg, q=0.25)
    
    op0 = decompose_hamiltonian_to_paulis(H0)
    op25 = decompose_hamiltonian_to_paulis(H25)
    
    U0_ex = compute_exact_unitary_evolution(H0, t_eval)
    U25_ex = compute_exact_unitary_evolution(H25, t_eval)
    
    p0_exact = np.abs((U0_ex @ np.eye(N)[:, start_node])[0])**2
    p25_exact = np.abs((U25_ex @ np.eye(N)[:, start_node])[0])**2
    
    print(f"Analytical benchmark P_hub -> q=0.00: {p0_exact:.6f} | q=0.25: {p25_exact:.6f}")
    print("-" * 78)
    print(f"{'Metric':<14} | {'Trotter Steps':<14} | {'Algorithm P_hub':<18} | {'Depth':<8} | {'Est. CNOTs'}")
    print("-" * 78)
    
    results = []
    t0 = time.time()
    
    for label, op, p_ex in [("q=0.00 (Unw.)", op0, p0_exact), ("q=0.25 (ROI)", op25, p25_exact)]:
        for reps in [1, 2, 4]:
            probs, depth, ops_count, _ = design_trotterized_quantum_circuit(
                op, start_node, t_eval, num_qubits, reps=reps, order=2
            )
            p_hub = compute_hub_residence_probability(probs)
            ipr = compute_bulk_ipr(probs)
            cnots = ops_count.get('cx', 0) + ops_count.get('ecr', 0) + int(depth * 1.4)
            
            print(f"{label:<14} | {reps:<14d} | {p_hub:<18.6f} | {depth:<8d} | ~{cnots}")
            results.append({
                'metric': label,
                'trotter_steps': reps,
                'p_hub_trotter': round(p_hub, 6),
                'p_hub_exact': round(p_ex, 6),
                'bulk_ipr': round(ipr, 6),
                'circuit_depth': depth,
                'total_gates': sum(ops_count.values())
            })
            
    print("-" * 78)
    print(f"Completed Trotter algorithm synthesis in {time.time()-t0:.2f}s")
    
    out_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'trotter_algorithm_results.csv')
    with open(out_path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        w.writeheader(); w.writerows(results)
    print(f"Results saved to {out_path}")

if __name__ == '__main__':
    run_trotter_experiment()
