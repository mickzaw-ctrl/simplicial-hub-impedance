"""
ROI Quantum Hybrid Stack
========================

Implementation of strategic decisions:

1. The core remains Willow-like:
   - surface code,
   - 105q tile,
   - data / measure / leakage split,
   - tunable couplers.

2. Cloud superconducting gate-model:
   - IBM Quantum,
   - Rigetti via AWS Braket / Azure / QCS-ready abstraction.

3. All-to-all alternative:
   - Quantinuum,
   - IonQ.

4. D-Wave separately:
   - as an ROI/QUBO optimization accelerator, not as a gate-model QPU.

This file is an orchestrator. It does not write secrets. Tokens and keys should
be set exclusively through environment variables used by the respective SDKs.

Examples:
    python roi_quantum_hybrid_stack.py --mode summary
    python roi_quantum_hybrid_stack.py --mode gate-dry --target ibm --benchmark bell --qubits 2
    python roi_quantum_hybrid_stack.py --mode gate-dry --target quantinuum --benchmark ghz --qubits 8
    python roi_quantum_hybrid_stack.py --mode dwave-qubo --qubo-size 32
    python roi_quantum_hybrid_stack.py --mode plan

Real submission to the cloud:
    Use quantum_cloud_bridge.py with the appropriate tokens or set --execute-cloud.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


# ============================================================
# 1. STRATEGIC BACKEND CONFIGURATION
# ============================================================

@dataclass(frozen=True)
class BackendSpec:
    key: str
    family: str
    modality: str
    role: str
    access_path: str
    default_provider: str
    example_backend: str
    cloud_public: bool
    notes: str


BACKENDS: Dict[str, BackendSpec] = {
    "willow_core": BackendSpec(
        key="willow_core",
        family="core",
        modality="Willow-like superconducting transmon model",
        role="Main ROI core: surface-code, 105q tile, data/measure/leakage, tunable couplers",
        access_path="local model",
        default_provider="local",
        example_backend="willow_compatible_neuro_qubit_1m.py",
        cloud_public=False,
        notes="Google Willow is not a public backend; the core is topologically compatible, not hardware-identical.",
    ),
    "ibm": BackendSpec(
        key="ibm",
        family="superconducting_gate_model",
        modality="superconducting transmon",
        role="Superconducting cloud benchmark closest to Willow-class gate-model",
        access_path="IBM Quantum / qiskit-ibm-runtime",
        default_provider="ibm",
        example_backend="ibm_brisbane / ibm_kyiv / ibm_sherbrooke",
        cloud_public=True,
        notes="Best Qiskit ecosystem and production superconducting cloud.",
    ),
    "rigetti": BackendSpec(
        key="rigetti",
        family="superconducting_gate_model",
        modality="superconducting transmon-like / tunable coupler roadmap",
        role="Second superconducting cloud benchmark; practical test of gate-model circuits",
        access_path="AWS Braket / Azure / Rigetti QCS",
        default_provider="aws",
        example_backend="arn:aws:braket:us-west-1::device/qpu/rigetti/Ankaa-3",
        cloud_public=True,
        notes="We use it via AWS/Azure because it is easiest to automate for multi-cloud.",
    ),
    "quantinuum": BackendSpec(
        key="quantinuum",
        family="all_to_all_gate_model",
        modality="trapped ion QCCD",
        role="All-to-all benchmark for dense Hamiltonians, QEC and circuits with mid-circuit measurement",
        access_path="Azure Quantum / Quantinuum Nexus",
        default_provider="azure",
        example_backend="quantinuum.sim.h1-1sc / quantinuum.qpu.h1-1 / quantinuum.qpu.h2-1",
        cloud_public=True,
        notes="Few qubits vs superconducting, but very high quality and full connectivity.",
    ),
    "ionq": BackendSpec(
        key="ionq",
        family="all_to_all_gate_model",
        modality="trapped ion",
        role="All-to-all / near-all-to-all for QUBO, dense graphs and hybrid ROI benchmarks",
        access_path="AWS Braket / Azure Quantum / IonQ Cloud",
        default_provider="aws",
        example_backend="arn:aws:braket:us-east-1::device/qpu/ionq/Aria-1 or ionq.simulator",
        cloud_public=True,
        notes="Very good fit for dense problems without SWAP overhead.",
    ),
    "dwave": BackendSpec(
        key="dwave",
        family="annealing_optimizer",
        modality="quantum annealing",
        role="Separate ROI optimization accelerator: QUBO/Ising split-merge/resource allocation",
        access_path="D-Wave Leap / Ocean SDK",
        default_provider="dwave",
        example_backend="Advantage2_system / Leap hybrid BQM sampler",
        cloud_public=True,
        notes="Not a gate-model QPU; use for QUBO, scheduling, allocation and graph optimization.",
    ),
}


# ============================================================
# 2. WILLOW-LIKE CORE VALIDATION
# ============================================================

@dataclass
class WillowCoreValidation:
    expected_total_qubits: int = 1_000_000
    tile_qubits: int = 105
    full_tiles: int = 9_523
    reserve_edge_qubits: int = 85
    data_qubits_per_tile: int = 49
    measure_qubits_per_tile: int = 48
    leakage_qubits_per_tile: int = 8

    @property
    def data_total(self) -> int:
        return self.full_tiles * self.data_qubits_per_tile

    @property
    def measure_total(self) -> int:
        return self.full_tiles * self.measure_qubits_per_tile

    @property
    def leakage_total(self) -> int:
        return self.full_tiles * self.leakage_qubits_per_tile

    @property
    def total(self) -> int:
        return self.data_total + self.measure_total + self.leakage_total + self.reserve_edge_qubits

    def validate(self) -> Dict[str, Any]:
        ok = self.total == self.expected_total_qubits
        return {
            "ok": ok,
            "expected_total_qubits": self.expected_total_qubits,
            "actual_total_qubits": self.total,
            "tile_qubits": self.tile_qubits,
            "full_tiles": self.full_tiles,
            "reserve_edge_qubits": self.reserve_edge_qubits,
            "data_qubit_plane": self.data_total,
            "measure_stabilizer_plane": self.measure_total,
            "leakage_reset_ancilla": self.leakage_total,
            "scale_vs_single_willow_105q": self.expected_total_qubits / self.tile_qubits,
        }


# ============================================================
# 3. GATE-MODEL CLOUD ROUTING
# ============================================================

@dataclass
class GateJobPlan:
    target: str
    bridge_provider: str
    backend: Optional[str]
    benchmark: str
    qubits: int
    depth: int
    shots: int
    command: str
    dry_run_supported: bool = True
    cloud_execution_notes: str = ""


def make_gate_job_plan(
    target: str,
    benchmark: str,
    qubits: int,
    depth: int,
    shots: int,
    backend: Optional[str] = None,
    execute_cloud: bool = False,
) -> GateJobPlan:
    target = target.lower()
    if target not in BACKENDS:
        raise ValueError(f"Unknown target: {target}. Available: {', '.join(BACKENDS)}")

    spec = BACKENDS[target]
    if target == "willow_core":
        provider = "dry-run"
        chosen_backend = backend or spec.example_backend
        notes = "Willow/Google is not a public cloud backend; plan only local/dry-run."
    elif target in {"ibm", "rigetti", "quantinuum", "ionq"}:
        provider = spec.default_provider if execute_cloud else "dry-run"
        chosen_backend = backend or (None if not execute_cloud else spec.example_backend)
        notes = f"Cloud path: {spec.access_path}. Backend example: {spec.example_backend}"
    else:
        raise ValueError(f"Target {target} is not a gate-model. For D-Wave use mode=dwave-qubo.")

    cmd_parts = [
        "python quantum_cloud_bridge.py",
        f"--provider {provider}",
        f"--benchmark {benchmark}",
        f"--qubits {qubits}",
        f"--depth {depth}",
        f"--shots {shots}",
    ]
    if chosen_backend and provider != "dry-run":
        cmd_parts.append(f"--backend {chosen_backend}")

    return GateJobPlan(
        target=target,
        bridge_provider=provider,
        backend=chosen_backend,
        benchmark=benchmark,
        qubits=qubits,
        depth=depth,
        shots=shots,
        command=" ".join(cmd_parts),
        cloud_execution_notes=notes,
    )


def run_gate_dry(plan: GateJobPlan) -> Dict[str, Any]:
    """Runs a dry-run via quantum_cloud_bridge.py, without submitting to the cloud."""
    try:
        from quantum_cloud_bridge import QuantumJobConfig, run_quantum_job

        cfg = QuantumJobConfig(
            provider="dry-run",
            backend=None,
            benchmark=plan.benchmark,
            qubits=plan.qubits,
            depth=plan.depth,
            shots=plan.shots,
        )
        result = run_quantum_job(cfg)
        payload = asdict(result)
        payload["planned_target"] = plan.target
        payload["planned_command"] = plan.command
        payload["cloud_execution_notes"] = plan.cloud_execution_notes
        return payload
    except Exception as exc:
        return {
            "status": "error",
            "planned_target": plan.target,
            "planned_command": plan.command,
            "error_type": type(exc).__name__,
            "message": str(exc),
        }


# ============================================================
# 4. D-WAVE ROI QUBO ACCELERATOR
# ============================================================

@dataclass
class ROIQuboProblem:
    """
    QUBO for ROI split/merge decisions.

    x_i = 1 means: split the voxel and allocate more resolution.
    x_i = 0 means: keep/merge, save resources.

    Energy:
        E = sum_i bias_i x_i + sum_(i,j) coupling_ij x_i x_j

    bias_i < 0 encourages split.
    bias_i > 0 discourages split.
    """

    n: int
    gradients: List[float]
    densities: List[float]
    sizes: List[float]
    couplings: Dict[Tuple[int, int], float]
    qubo: Dict[Tuple[int, int], float]


def build_roi_split_qubo(
    n: int = 32,
    seed: int = 7,
    gradient_weight: float = 2.0,
    density_weight: float = 1.0,
    size_penalty: float = 0.4,
    smoothness: float = 0.15,
) -> ROIQuboProblem:
    rng = random.Random(seed)
    gradients = [rng.random() for _ in range(n)]
    densities = [rng.random() for _ in range(n)]
    sizes = [0.5 + rng.random() for _ in range(n)]

    qubo: Dict[Tuple[int, int], float] = {}
    couplings: Dict[Tuple[int, int], float] = {}

    for i in range(n):
        # High gradient and high density -> worth splitting -> negative bias.
        # Large size and resource cost -> penalty -> positive bias.
        bias = -gradient_weight * gradients[i] - density_weight * densities[i] + size_penalty * sizes[i]
        qubo[(i, i)] = bias

    # Local consistency: neighboring voxels should not chaotically split without need.
    # A positive coupling penalizes simultaneous splitting of neighbors; it can be changed to negative
    # if we want to promote clustering of splits.
    for i in range(n - 1):
        j = i + 1
        c = smoothness * (1.0 - abs(gradients[i] - gradients[j]))
        couplings[(i, j)] = c
        qubo[(i, j)] = c

    # A few long-range ROI graph connections.
    for _ in range(max(1, n // 4)):
        i = rng.randrange(n)
        j = rng.randrange(n)
        if i == j:
            continue
        a, b = min(i, j), max(i, j)
        c = smoothness * 0.5 * rng.random()
        couplings[(a, b)] = c
        qubo[(a, b)] = qubo.get((a, b), 0.0) + c

    return ROIQuboProblem(n=n, gradients=gradients, densities=densities, sizes=sizes, couplings=couplings, qubo=qubo)


def qubo_energy(sample: Dict[int, int], qubo: Dict[Tuple[int, int], float]) -> float:
    e = 0.0
    for (i, j), q in qubo.items():
        e += q * sample.get(i, 0) * sample.get(j, 0)
    return e


def solve_qubo_classical(problem: ROIQuboProblem, reads: int = 2000, seed: int = 7) -> Dict[str, Any]:
    """Lightweight simulated random search fallback with no dependencies."""
    rng = random.Random(seed)
    best_sample: Dict[int, int] = {}
    best_energy = float("inf")

    # Heuristic start: split where bias < 0.
    current = {i: 1 if problem.qubo[(i, i)] < 0 else 0 for i in range(problem.n)}
    current_e = qubo_energy(current, problem.qubo)
    best_sample = dict(current)
    best_energy = current_e

    temp0 = 1.0
    for r in range(reads):
        temp = temp0 * (1.0 - r / max(1, reads)) + 1e-3
        i = rng.randrange(problem.n)
        candidate = dict(current)
        candidate[i] = 1 - candidate[i]
        cand_e = qubo_energy(candidate, problem.qubo)
        delta = cand_e - current_e
        if delta < 0 or rng.random() < math.exp(-delta / temp):
            current = candidate
            current_e = cand_e
        if current_e < best_energy:
            best_sample = dict(current)
            best_energy = current_e

    actions = {i: ("split" if best_sample[i] else "merge_or_stay") for i in range(problem.n)}
    return {
        "solver": "classical_fallback_simulated_annealing",
        "energy": best_energy,
        "sample": best_sample,
        "actions": actions,
        "split_count": sum(best_sample.values()),
        "merge_or_stay_count": problem.n - sum(best_sample.values()),
    }


def solve_qubo_dwave(problem: ROIQuboProblem, reads: int = 1000) -> Dict[str, Any]:
    """
    Tries to use D-Wave Ocean. If the token/SDK are not available, uses the fallback.

    Environment variables for real D-Wave:
        DWAVE_API_TOKEN
        DWAVE_API_ENDPOINT optional
        DWAVE_SOLVER optional
    """
    if not os.getenv("DWAVE_API_TOKEN"):
        fallback = solve_qubo_classical(problem, reads=max(1000, reads))
        fallback["dwave_status"] = "skipped_no_DWAVE_API_TOKEN"
        return fallback

    try:
        from dwave.system import EmbeddingComposite, DWaveSampler
    except ModuleNotFoundError:
        fallback = solve_qubo_classical(problem, reads=max(1000, reads))
        fallback["dwave_status"] = "skipped_missing_dwave_ocean_sdk"
        fallback["install"] = "pip install dwave-ocean-sdk"
        return fallback

    sampler = EmbeddingComposite(DWaveSampler())
    sampleset = sampler.sample_qubo(problem.qubo, num_reads=reads)
    first = sampleset.first
    sample = {int(k): int(v) for k, v in dict(first.sample).items()}
    actions = {i: ("split" if sample.get(i, 0) else "merge_or_stay") for i in range(problem.n)}
    return {
        "solver": "dwave_qpu_or_leap_sampler",
        "energy": float(first.energy),
        "sample": sample,
        "actions": actions,
        "split_count": sum(sample.values()),
        "merge_or_stay_count": problem.n - sum(sample.values()),
        "dwave_status": "completed",
    }


# ============================================================
# 5. DEPLOYMENT PLAN
# ============================================================

def deployment_plan() -> Dict[str, Any]:
    return {
        "decision": "hybrid_roi_quantum_stack",
        "core": {
            "backend": asdict(BACKENDS["willow_core"]),
            "validation": WillowCoreValidation().validate(),
            "status": "kept_as_primary_architecture",
        },
        "superconducting_cloud": {
            "ibm": asdict(BACKENDS["ibm"]),
            "rigetti": asdict(BACKENDS["rigetti"]),
            "purpose": "gate-model benchmarks with topology closest to superconducting Willow-like",
        },
        "all_to_all_alternative": {
            "quantinuum": asdict(BACKENDS["quantinuum"]),
            "ionq": asdict(BACKENDS["ionq"]),
            "purpose": "dense Hamiltonians, QUBO, non-local circuits without SWAP overhead",
        },
        "optimization_accelerator": {
            "dwave": asdict(BACKENDS["dwave"]),
            "purpose": "QUBO/Ising for voxel split-merge, resource allocation and ROI scheduling",
        },
        "security": {
            "secrets_policy": "tokens only via environment variables; never write tokens to files",
            "required_env_examples": [
                "IBM_QUANTUM_TOKEN",
                "AWS_PROFILE / AWS_REGION / BRAKET_S3_BUCKET",
                "AZURE_QUANTUM_RESOURCE_ID / AZURE_QUANTUM_LOCATION",
                "DWAVE_API_TOKEN",
            ],
        },
    }


def stack_summary() -> Dict[str, Any]:
    return {
        "willow_core_validation": WillowCoreValidation().validate(),
        "backends": {k: asdict(v) for k, v in BACKENDS.items()},
    }


# ============================================================
# 6. CLI
# ============================================================

def write_result(payload: Dict[str, Any], output: str) -> None:
    Path(output).write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="ROI Quantum Hybrid Stack")
    parser.add_argument("--mode", choices=["summary", "plan", "gate-dry", "dwave-qubo"], default="summary")
    parser.add_argument("--target", choices=list(BACKENDS.keys()), default="ibm")
    parser.add_argument("--benchmark", choices=["bell", "ghz", "rcs", "surface-proxy", "qec"], default="bell")
    parser.add_argument("--backend", default=None)
    parser.add_argument("--qubits", type=int, default=2)
    parser.add_argument("--depth", type=int, default=8)
    parser.add_argument("--shots", type=int, default=1024)
    parser.add_argument("--execute-cloud", action="store_true", help="only plans the cloud command; does not execute it here")
    parser.add_argument("--qubo-size", type=int, default=32)
    parser.add_argument("--reads", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--output", default="roi_quantum_hybrid_stack_result.json")
    args = parser.parse_args()

    t0 = time.perf_counter()

    if args.mode == "summary":
        payload = stack_summary()
    elif args.mode == "plan":
        payload = deployment_plan()
    elif args.mode == "gate-dry":
        plan = make_gate_job_plan(
            target=args.target,
            benchmark=args.benchmark,
            qubits=args.qubits,
            depth=args.depth,
            shots=args.shots,
            backend=args.backend,
            execute_cloud=args.execute_cloud,
        )
        dry = run_gate_dry(plan)
        payload = {"gate_job_plan": asdict(plan), "dry_run_result": dry}
    elif args.mode == "dwave-qubo":
        problem = build_roi_split_qubo(n=args.qubo_size, seed=args.seed)
        solution = solve_qubo_dwave(problem, reads=args.reads)
        payload = {
            "accelerator": asdict(BACKENDS["dwave"]),
            "problem": {
                "n": problem.n,
                "qubo_terms": len(problem.qubo),
                "gradient_mean": sum(problem.gradients) / problem.n,
                "density_mean": sum(problem.densities) / problem.n,
            },
            "solution": solution,
        }
    else:
        raise ValueError(args.mode)

    payload["elapsed_seconds"] = time.perf_counter() - t0
    write_result(payload, args.output)
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    print(f"\nWrote: {args.output}")


if __name__ == "__main__":
    main()
