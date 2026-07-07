"""
Simplicial & Quantum Observables Module
---------------------------------------
Author: Michał Ślusarczyk
Date: July 2026

Evaluates topological, spectral, and quantum localization observables:
- Hub Residence Probability (P_hub)
- Inverse Participation Ratio (IPR) for localization diagnosis
- Hub Transport Share (Phi_hub)
- Scale-dependent Random Walk Spectral Dimension D_s(tau)
"""

import numpy as np

def compute_hub_residence_probability(probs, hub_indices=[0]):
    """
    Computes total quantum probability residing in the topological hub vertices.
    """
    return float(np.sum([probs[idx] for idx in hub_indices]))

def compute_bulk_ipr(probs, hub_indices=[0]):
    """
    Computes the Inverse Participation Ratio (IPR) over bulk vertices:
    
        IPR = sum_{v in Bulk} |P(v)|^2 / (sum_{v in Bulk} P(v))^2
        
    A high IPR (~1) indicates strong spatial localization/trapping, while a low IPR
    indicates uniform delocalization and smooth propagation across bulk geometry.
    """
    bulk_probs = np.array([p for idx, p in enumerate(probs) if idx not in hub_indices], dtype=float)
    total_bulk = np.sum(bulk_probs)
    if total_bulk <= 0:
        return 0.0
    return float(np.sum(bulk_probs ** 2) / (total_bulk ** 2))

def compute_hub_transport_share(adj, deg, q=0.25, hub_cutoff_quantile=0.99):
    """
    Computes the share of total network conductance mediated by hub-incident edges:
    
        Phi_hub(q) = sum_{(u,v) in E_hub} w_{uv} / sum_{(u,v) in E} w_{uv}
    """
    N = len(deg)
    cutoff = np.quantile(np.array(deg, dtype=float), hub_cutoff_quantile)
    top_nodes = {u for u, d in enumerate(deg) if d >= cutoff}
    
    conductance_sum = 0.0
    top_edge_conductance = 0.0
    
    for u in range(N):
        for v in adj[u]:
            w = (max(1, deg[u]) * max(1, deg[v])) ** (-q)
            conductance_sum += w
            if u in top_nodes or v in top_nodes:
                top_edge_conductance += w
                
    return float(top_edge_conductance / conductance_sum) if conductance_sum > 0 else 0.0

def measure_spectral_dimension(ts, return_probs, t_lo, t_hi):
    """
    Extracts effective scale-dependent spectral dimension D_s by fitting
    logarithmic derivative of heat kernel return probability:
    
        D_s = -2 * d(log P_r(tau)) / d(log tau)
    """
    ts_arr = np.array(ts, dtype=float)
    pr_arr = np.array(return_probs, dtype=float)
    
    mask = (ts_arr >= t_lo) & (ts_arr <= t_hi) & (pr_arr > 0)
    if mask.sum() < 3:
        return float('nan')
        
    slope, _ = np.polyfit(np.log(ts_arr[mask]), np.log(pr_arr[mask]), 1)
    return float(-2.0 * slope)
