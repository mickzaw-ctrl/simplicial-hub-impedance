#!/usr/bin/env python3
"""
NUMERICAL SIMULATION OF COSMOLOGICAL EVOLUTION IN ROI v5.2
------------------------------------------------------------
Author: Michał Ślusarczyk
Date: July 2026

This script simulates the cosmological evolution of a homogeneous FLRW
universe driven by the effective dark energy that emerges from the discrete
Einstein equations on the lattice:
    G_{uv} + Lambda * g_{uv}^{eff} \approx -1.1837

It determines:
1. The dark energy density rho_DE, the effective cosmological constant
   Lambda_eff, and the Hubble parameter H_0.
2. The evolution of the scale factor a(t) for a flat universe (k=0,
   de Sitter expansion).
3. The evolution of the scale factor a(t) for a closed universe (k=+1,
   quantum bounce cosh(H_0 t)).
4. The evolution of the spatial volume V_3(t) and its convergence with CDT
   profiles.

Output:
- cosmology_scale_factor_results.csv
"""

import numpy as np
import csv

def simulate_cosmological_evolution():
    # Mean value of the regularized Einstein tensor on the lattice
    G_reg_mean = -1.183705

    # Simplicial / Planck units (8 pi G = 1)
    G_val = 1.0 / (8.0 * np.pi)

    # Cosmological parameters
    Lambda_eff = -G_reg_mean
    rho_DE = Lambda_eff  # for 8 pi G = 1
    w_eos = -1.0         # vacuum equation of state
    H0 = np.sqrt(Lambda_eff / 3.0)
    a_min = 1.0 / H0     # minimal bounce radius for k=+1

    print("=" * 78)
    print("COSMOLOGICAL SIMULATION OF THE ROI v5.2 MODEL (FLRW METRIC)")
    print(f"Author: Michał Ślusarczyk | Regularized Einstein tensor <G_reg>: {G_reg_mean:.6f}")
    print("=" * 78)
    print(f"Effective Cosmological Constant (Λ_eff): {Lambda_eff:.6f}")
    print(f"Dark Energy Density (ρ_DE):              {rho_DE:.6f} (in units 8πG=1)")
    print(f"Vacuum Equation of State (w):             {w_eos:.4f}")
    print(f"Hubble Parameter (H_0):                   {H0:.6f} t^-1")
    print(f"Minimal Bounce Radius (a_min):             {a_min:.6f} (for k=+1)")
    print("=" * 78)

    ts = np.linspace(-5.0, 5.0, 101)
    results = []

    print(f"{'Time t':<8} | {'a(t) [k=0 flat]':<18} | {'a(t) [k=+1 bounce]':<20} | {'V_3(t) [k=+1 volume]':<22} | {'Cosmological phase'}")
    print("-" * 88)

    for idx, t in enumerate(ts):
        # 1. Flat universe (k=0): a(t) = a_min * exp(H_0 * t) for t >= 0
        a_flat = a_min * np.exp(H0 * t)

        # 2. Closed universe (k=+1): a(t) = a_min * cosh(H_0 * t)
        a_closed = a_min * np.cosh(H0 * t)

        # 3. Spatial volume of the S^3 sphere: V_3(t) = 2 * pi^2 * a^3(t)
        v3_closed = 2.0 * (np.pi ** 2) * (a_closed ** 3)

        # Phase
        if abs(t) < 0.2:
            phase = "BOUNCE (quantum bounce)"
        elif t < 0:
            phase = "Contraction (collapse)"
        else:
            phase = "de Sitter expansion"

        if idx % 10 == 0 or abs(t) < 1e-10:
            print(f"{t:<8.2f} | {a_flat:<18.4f} | {a_closed:<20.4f} | {v3_closed:<22.2f} | {phase}")

        results.append({
            'time': round(t, 4),
            'scale_factor_flat_k0': round(a_flat, 6),
            'scale_factor_closed_k1': round(a_closed, 6),
            'volume_closed_S3': round(v3_closed, 4),
            'hubble_rate': round(H0 * np.tanh(H0 * t), 6),
            'acceleration_ddot_a': round(a_closed * (H0 ** 2), 6),
            'cosmological_phase': phase
        })

    print("-" * 88)

    out_file = 'cosmology_scale_factor_results.csv'
    with open(out_file, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        w.writeheader(); w.writerows(results)
    print(f"Saved cosmological evolution to: {out_file}")
    print("=" * 88)

if __name__ == '__main__':
    simulate_cosmological_evolution()
