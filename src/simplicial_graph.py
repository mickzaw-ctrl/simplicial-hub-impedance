"""
Simplicial Triangulation Graph Generator
----------------------------------------
Author: Michał Ślusarczyk
Date: July 2026

Generates 1-skeleton dual graphs representing discrete simplicial geometry patches
with embedded topological coordination anomalies (hubs).
"""

import numpy as np

def generate_simplicial_patch_with_hub(num_qubits=5):
    """
    Generates a 1-skeleton graph representing a simplicial patch with N = 2^num_qubits vertices.
    
    Structure:
    - Vertex 0 acts as a topological coordination anomaly (hub), connected to all bulk vertices.
    - Vertices 1..(N-1) form a regular 1D/2D lattice ring with local degree ~4 representing
      smooth macroscopic 4D spacetime triangulation bulk geometry.
      
    Args:
        num_qubits (int): Number of qubits defining the Hilbert space dimension (N = 2^num_qubits).
        
    Returns:
        tuple: (N, adj, deg) where N is total node count, adj is adjacency dictionary, and deg is degree list.
    """
    N = 1 << num_qubits
    adj = {i: [] for i in range(N)}
    
    # 1. Hub connections (Vertex 0 connected to all bulk vertices)
    for i in range(1, N):
        adj[0].append(i)
        adj[i].append(0)
        
    # 2. Regular bulk geometry (1st and 2nd nearest neighbor lattice ring)
    for i in range(1, N):
        for offset in [1, 2]:
            nxt = ((i - 1 + offset) % (N - 1)) + 1
            adj[i].append(nxt)
            adj[nxt].append(i)
            
    # Remove duplicates and sort
    for i in range(N):
        adj[i] = sorted(list(set(adj[i])))
        
    deg = [len(adj[i]) for i in range(N)]
    return N, adj, deg

def compute_graph_degree_stats(deg):
    """
    Computes statistical moments and percentiles of the degree distribution.
    """
    deg_arr = np.array(deg, dtype=float)
    return {
        'max': float(np.max(deg_arr)),
        'mean': float(np.mean(deg_arr)),
        'std': float(np.std(deg_arr)),
        'q99': float(np.quantile(deg_arr, 0.99))
    }
