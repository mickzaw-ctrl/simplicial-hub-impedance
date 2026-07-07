"""
Weighted Physical Impedance Hamiltonian Module
----------------------------------------------
Author: Michał Ślusarczyk
Date: July 2026

Constructs the ROI v5.2 weighted physical edge impedance Hamiltonian matrix and
evaluates exact continuous-time quantum walk (CTQW) unitary evolution operators.
"""

import numpy as np
import scipy.linalg as la

def build_roi_hamiltonian(N, adj, deg, q=0.25):
    """
    Constructs the Hermitian Hamiltonian matrix H_q for continuous-time quantum walks:
    
        H_{uv} = - (d_u * d_v)^(-q)   for (u, v) in E
        
    Args:
        N (int): Total number of vertices in the graph.
        adj (dict): Adjacency dictionary mapping node index to neighbor indices.
        deg (list): Vertex degree list.
        q (float): Physical impedance exponent (default: 0.25 for ROI v5.2 concrete solution).
        
    Returns:
        np.ndarray: N x N complex Hermitian Hamiltonian matrix.
    """
    H = np.zeros((N, N), dtype=complex)
    for u in range(N):
        for v in adj[u]:
            w = (max(1, deg[u]) * max(1, deg[v])) ** (-q)
            H[u, v] = -w
            H[v, u] = -w
    return H

def compute_exact_unitary_evolution(H, t):
    """
    Evaluates the exact time evolution unitary operator U(t) = exp(-i * H * t)
    using classical dense matrix exponential.
    
    Args:
        H (np.ndarray): Hermitian Hamiltonian matrix.
        t (float): Evolution time.
        
    Returns:
        np.ndarray: N x N unitary transition matrix U(t).
    """
    return la.expm(-1j * H * t)

def simulate_exact_walk(H, start_node, t):
    """
    Simulates exact CTQW evolution starting from a localized bulk vertex.
    
    Returns:
        np.ndarray: Real probability distribution vector P(v, t) over all vertices.
    """
    N = H.shape[0]
    psi0 = np.zeros(N, dtype=complex)
    psi0[start_node] = 1.0
    U = compute_exact_unitary_evolution(H, t)
    psi_t = U @ psi0
    return np.abs(psi_t) ** 2
