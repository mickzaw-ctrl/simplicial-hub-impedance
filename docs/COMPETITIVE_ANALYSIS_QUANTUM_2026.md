# Competitive Comparison: Willow / Willow-compatible 1M / QPU Market

Analysis date: 2026-06-29

## 1. Reference Point: Google Willow

**Google Willow** is a publicly described 105-qubit superconducting transmon processor designed for surface-code error correction. Public Willow metrics include:

- **105 physical qubits**.
- **Square-grid / surface-code layout**.
- **T1 ≈ 68 µs**, **T2,CPMG ≈ 89 µs** in the Nature publication.
- Demonstration of **below-threshold QEC**, i.e., a situation where increasing the code distance reduces the logical error.
- **Random Circuit Sampling** benchmark executed by Willow in under 5 minutes according to Google; Google compares this with an estimate of approx. `10^25` years for a classical supercomputer.

Sources: Nature `Quantum error correction below the surface code threshold`; Google Blog `Meet Willow, our state-of-the-art quantum chip`.

---

## 2. Our Architecture: Willow-compatible Neuro-Qubit System 1M

Our architecture is not a copy of Google's hardware, but is **topologically compatible** with the public description of Willow:

- transmon-like Bloch representation,
- tile 105q,
- surface-code role layout,
- data / measure / leakage role split,
- tunable-coupler fabric,
- T1/T2 decoherence envelope,
- total scale: **1,000,000 effective qubits**.

Breakdown:

| Layer | Qubits |
|---|---:|
| data_qubit_plane | 466,627 |
| measure_stabilizer_plane | 457,104 |
| leakage_reset_ancilla | 76,184 |
| reserve_edge_coupler_band | 85 |
| **Total** | **1,000,000** |

Scale relative to a single Willow 105q:

```text
1,000,000 / 105 ≈ 9,523.81x
```

Note: this is an effective / neuro-physical model, not a full amplitude simulation of `2^1_000_000`.

---

## 3. Competitive Table

| Player | Modality | Public scale / system | Connectivity | QEC / benchmark | Cloud access | Assessment relative to Willow-compatible 1M |
|---|---|---:|---|---|---|---|
| **Google Willow** | Superconducting transmon | 105 qubits | 2D square grid, nearest-neighbor, tunable couplers | Below-threshold surface code; RCS <5 min per Google | No public Willow backend | Best reference point for our topological architecture |
| **IBM Heron / Nighthawk** | Superconducting transmon | Heron 156 qubits; Nighthawk 120 qubits | Heron: heavy-hex; Nighthawk: grid | Strong cloud stack, error mitigation, FTQC roadmap | Yes, IBM Quantum | Strongest cloud-superconducting competitor; different topology than Willow |
| **Quantinuum H2** | Trapped ion / QCCD | H2: 56 physical qubits | All-to-all via ion shuttling | Very high fidelity, Quantum Volume, mid-circuit measurement | Yes, Quantinuum / Azure | Fewer qubits, but very high quality and connectivity |
| **IonQ Forte / Tempo** | Trapped ion | Forte: 36 qubits / #AQ36; Tempo target #AQ64 | Nearly all-to-all | Strong for dense QUBO / optimization problems | Yes, AWS / Azure / IonQ | Better connectivity than Willow-grid, but slower gates and smaller production scale |
| **Rigetti Ankaa-3** | Superconducting | 84 qubits | Chip superconducting, modular roadmap | 99.0% iSWAP median, 99.5% fSim median per Rigetti | QCS, AWS Braket, Azure | Closest ecosystem-wise to superconducting cloud, but weaker QEC than Willow |
| **D-Wave Advantage2** | Quantum annealing | 4,400+ qubits | Zephyr, 20-way connectivity | Optimization/annealing, not universal gates | Yes, Leap | Competitor for optimization, not a direct rival to universal Willow-like QEC |
| **Microsoft Majorana 1** | Topological / Majorana | 8 topological qubits per Microsoft announcement | Topological Core roadmap | Strategy: hardware-level error resistance, target 1M qubits | Azure ecosystem, no full public FTQC QPU | Greatest long-term potential, but most risky/futuristic path |
| **QuEra / neutral atoms** | Neutral atoms / Rydberg | Public neutral-atom systems, logical qubits roadmap | Reconfigurable arrays | Strong QEC potential with low overhead | AWS Braket / on-prem projects | Very strong scaling competitor, especially for QEC and physical simulations |

---

## 4. Strategic Comparison

### Google Willow / Our Willow-compatible 1M

**Strengths:**

- Most direct path to surface-code FTQC.
- Natural mapping to 2D lattice.
- Strong QEC proof: below-threshold.
- Our 1M model preserves this layout at million-scale.

**Weaknesses:**

- Nearest-neighbor routing increases depth of non-local circuits.
- Superconducting qubits require cryogenics, calibration, and cross-talk control.
- Willow is not publicly available as a cloud backend.

### IBM

**Strengths:**

- Best cloud availability among superconducting vendors.
- Large Qiskit ecosystem.
- Heron 156 qubits provides more physical qubits than Willow 105q.
- Strong roadmap to fault tolerance.

**Weaknesses:**

- Heavy-hex does not map as directly to a classical surface-code square-grid as Willow.
- In practice, much depends on error mitigation and transpilation.

### Quantinuum

**Strengths:**

- All-to-all connectivity.
- Very high fidelity.
- Mid-circuit measurement, qubit reuse, dynamic circuits.
- Excellent for deep circuits with fewer qubits.

**Weaknesses:**

- Fewer physical qubits than superconducting leaders.
- Slower operations than in superconducting QPUs.
- Ion scaling is mechanically/systemically difficult.

### IonQ

**Strengths:**

- Trapped-ion all-to-all / near-all-to-all is excellent for dense Hamiltonians.
- #AQ is a practical utility indicator, not just qubit count.
- Good fit for optimization and hybrid problems.

**Weaknesses:**

- Smaller physical production scale.
- Gate time significantly longer than superconducting.
- Aggressive roadmap, but requires hardware confirmation.

### Rigetti

**Strengths:**

- Superconducting + cloud access.
- Fast gates.
- Modular roadmap.
- Availability via QCS / AWS / Azure.

**Weaknesses:**

- Less advanced QEC than Google Willow.
- Lower fidelity than top trapped-ion systems.

### D-Wave

**Strengths:**

- Large qubit count: 4,400+.
- Production optimization applications.
- Cloud works practically right now.

**Weaknesses:**

- This is an annealer, not a universal gate-based computer.
- Not a direct competitor to Willow-style surface-code FTQC.

### Microsoft Majorana

**Strengths:**

- If topological qubits work at scale, correction overhead could be much lower.
- Declared path to 1M qubits on a single chip.

**Weaknesses:**

- High technological risk.
- Few publicly verified production metrics.
- More of a roadmap than an available cloud-QPU.

### Neutral atoms / QuEra / Atom Computing

**Strengths:**

- Very good scalability of physical atom count.
- Reconfigurable geometry.
- Potentially low QEC overhead.
- Natural fit for physical simulations.

**Weaknesses:**

- Gate fidelity and error control need further maturation.
- Universal, long FTQC circuits are still in a transitional phase.

---

## 5. Ranking by Criteria

Scale: 1–5, where 5 = best. This is a strategic assessment, not an official benchmark.

| Platform | Physical scale | Fidelity / quality | Connectivity | QEC maturity | Cloud availability | Fit to ROI/Willow |
|---|---:|---:|---:|---:|---:|---:|
| Google Willow | 3 | 5 | 3 | 5 | 1 | 5 |
| Our Willow-compatible 1M | 5* | 2* | 3 | 3* | 3 | 5 |
| IBM Heron/Nighthawk | 4 | 4 | 3 | 4 | 5 | 3 |
| Quantinuum H2 | 2 | 5 | 5 | 4 | 4 | 2 |
| IonQ Forte/Tempo | 2–3 | 4 | 5 | 3 | 4 | 2 |
| Rigetti Ankaa-3 | 3 | 3 | 3 | 2 | 4 | 3 |
| D-Wave Advantage2 | 5 | 3 | 4 | N/A | 5 | 1 |
| Microsoft Majorana | 1 now / 5 potential | ? | ? | ? | 3 | 2 |
| Neutral atoms | 5 | 3–4 | 4 | 4 | 3–4 | 3 |

`*` For our model: scale is effective/simulated, not physical; fidelity/QEC are proxies, not laboratory measurements.

---

## 6. Final Conclusion

The strongest competitors depend on the criterion:

1. **Closest technologically to Willow:** IBM + Rigetti, because they are superconducting QPUs.
2. **Best qubit quality and connectivity:** Quantinuum / IonQ, because trapped ions have all-to-all and very high fidelity.
3. **Largest number of qubits available in production:** D-Wave, but this is annealing, not universal gate-model FTQC.
4. **Most promising alternative scaling:** neutral atoms.
5. **Greatest high-risk/high-reward:** Microsoft Majorana.
6. **Best QEC benchmark relative to surface code:** Google Willow.

For our project **ROI / Neuro-Physical Engine / Willow-compatible 1M**, the best strategy is to:

- maintain **Willow-like surface-code topology** as the core,
- add cloud adapters to IBM / Rigetti for superconducting experiments,
- add a trapped-ion backend for all-to-all benchmarks,
- add a neutral-atom module as a scaling alternative,
- treat D-Wave as a separate ROI optimization accelerator, not as a gate-based QPU.
