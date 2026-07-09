"""
Quantum Cloud Bridge for Willow-compatible Neuro-Qubit System 1M
================================================================

Bridge to quantum clouds: IBM Quantum, AWS Braket, Azure Quantum and a
local/dry-run mode. The project is secret-safe: tokens/API keys are read
ONLY from environment variables, never written to a file.

IMPORTANT ABOUT GOOGLE WILLOW:
Google Willow is not a publicly available cloud backend for arbitrary tasks.
This bridge does not pretend to access Willow. It does, however, allow:
- comparing our metrics with public Willow metrics,
- exporting small benchmark circuits in the RCS/QEC style,
- running compatible tests on available quantum clouds.

Supported modes:
    dry-run      - no cloud, only validation and task description,
    local        - local statevector simulator, if qiskit is available,
    ibm          - IBM Quantum via qiskit-ibm-runtime,
    aws          - AWS Braket via amazon-braket-sdk,
    azure        - Azure Quantum via azure-quantum.

Optional installation:
    pip install qiskit qiskit-aer qiskit-ibm-runtime
    pip install amazon-braket-sdk
    pip install azure-quantum

Examples:
    python quantum_cloud_bridge.py --provider dry-run --benchmark bell --qubits 2
    python quantum_cloud_bridge.py --provider local --benchmark bell --qubits 2 --shots 1024
    python quantum_cloud_bridge.py --provider ibm --backend ibm_brisbane --benchmark rcs --qubits 8 --depth 12 --shots 1024
    python quantum_cloud_bridge.py --provider aws --backend arn:aws:braket:us-east-1::device/qpu/ionq/Aria-1 --benchmark bell --shots 100
    python quantum_cloud_bridge.py --provider azure --backend ionq.qpu --benchmark bell --shots 100

Environment variables:
    IBM_QUANTUM_TOKEN       - IBM Quantum token
    IBM_QUANTUM_INSTANCE    - optional CRN/instance for IBM Runtime

    AWS_PROFILE             - optional AWS profile
    AWS_REGION              - e.g. us-east-1
    BRAKET_S3_BUCKET        - required for AWS Braket QPU/simulator managed job
    BRAKET_S3_PREFIX        - optional prefix, default quantum-cloud-bridge

    AZURE_QUANTUM_RESOURCE_ID - full resource id of Azure Quantum workspace
    AZURE_QUANTUM_LOCATION    - e.g. eastus
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, Tuple


# ============================================================
# 1. CONFIGURATION AND METADATA
# ============================================================

@dataclass
class QuantumJobConfig:
    provider: str = "dry-run"
    backend: Optional[str] = None
    benchmark: str = "bell"
    qubits: int = 2
    depth: int = 8
    shots: int = 1024
    seed: int = 7
    optimization_level: int = 1
    dry_run: bool = False


@dataclass
class QuantumJobResult:
    status: str
    provider: str
    backend: Optional[str]
    benchmark: str
    qubits: int
    depth: int
    shots: int
    job_id: Optional[str]
    counts: Optional[Dict[str, int]]
    metrics: Dict[str, Any]
    message: str


# ============================================================
# 2. QISKIT CIRCUIT BUILDERS
# ============================================================

def require_qiskit():
    try:
        from qiskit import QuantumCircuit
        return QuantumCircuit
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "qiskit is missing. Install: pip install qiskit qiskit-aer qiskit-ibm-runtime"
        ) from exc


def build_bell_qiskit(qubits: int = 2):
    QuantumCircuit = require_qiskit()
    if qubits < 2:
        raise ValueError("Bell benchmark requires at least 2 qubits.")
    qc = QuantumCircuit(qubits, qubits)
    qc.h(0)
    qc.cx(0, 1)
    # Remaining qubits, if any, stay in |0>; we measure all of them.
    qc.measure(range(qubits), range(qubits))
    return qc


def build_ghz_qiskit(qubits: int = 8):
    QuantumCircuit = require_qiskit()
    if qubits < 2:
        raise ValueError("GHZ benchmark requires at least 2 qubits.")
    qc = QuantumCircuit(qubits, qubits)
    qc.h(0)
    for i in range(qubits - 1):
        qc.cx(i, i + 1)
    qc.measure(range(qubits), range(qubits))
    return qc


def build_rcs_qiskit(qubits: int = 8, depth: int = 12, seed: int = 7):
    """
    Small Random Circuit Sampling benchmark inspired by RCS tests.
    This is NOT a full Google Willow benchmark. It is a portable cloud test.
    """
    QuantumCircuit = require_qiskit()
    rng = random.Random(seed)
    qc = QuantumCircuit(qubits, qubits)

    single_gates = ["h", "sx", "x", "rz"]
    for layer in range(depth):
        # Random 1Q gates.
        for q in range(qubits):
            g = rng.choice(single_gates)
            if g == "h":
                qc.h(q)
            elif g == "sx":
                qc.sx(q)
            elif g == "x":
                qc.x(q)
            elif g == "rz":
                qc.rz(rng.uniform(-math.pi, math.pi), q)

        # Nearest neighbors, even/odd layers as in a 1D lattice.
        start = layer % 2
        for q in range(start, qubits - 1, 2):
            # CZ/CX are widely supported; real Willow uses its own native gates.
            if rng.random() < 0.5:
                qc.cz(q, q + 1)
            else:
                qc.cx(q, q + 1)

    qc.measure(range(qubits), range(qubits))
    return qc


def build_surface_code_proxy_qiskit(distance: int = 3):
    """
    Minimal proxy circuit for surface code: data qubits + measure qubits.
    It is not a full QEC decoder, but it tests the role of data/measure and syndrome measurement.

    For d=3 it uses 9 data qubits and 8 measure qubits = 17 qubits.
    """
    QuantumCircuit = require_qiskit()
    if distance < 3 or distance % 2 == 0:
        raise ValueError("surface proxy uses an odd distance >= 3.")

    data = distance * distance
    measure = 2 * distance * (distance - 1) // 2 + 2 * distance * (distance - 1) // 2
    # For simplicity we limit the number of stabilizers to data-1.
    measure = min(data - 1, measure)
    n = data + measure
    qc = QuantumCircuit(n, measure)

    # Prepare a delicate superposition on the data plane.
    for q in range(data):
        qc.h(q)

    # Parity stabilizer proxy: measure qubits collect the parity of neighboring data qubits.
    for m in range(measure):
        anc = data + m
        a = m
        b = (m + 1) % data
        qc.cx(a, anc)
        qc.cx(b, anc)
        qc.measure(anc, m)

    return qc


def build_qiskit_circuit(config: QuantumJobConfig):
    b = config.benchmark.lower()
    if b == "bell":
        return build_bell_qiskit(config.qubits)
    if b == "ghz":
        return build_ghz_qiskit(config.qubits)
    if b == "rcs":
        return build_rcs_qiskit(config.qubits, config.depth, config.seed)
    if b in {"surface", "surface-proxy", "qec"}:
        return build_surface_code_proxy_qiskit(distance=config.qubits)
    raise ValueError(f"Unknown benchmark: {config.benchmark}. Use: bell, ghz, rcs, surface-proxy.")


# ============================================================
# 3. COUNTS METRICS
# ============================================================

def normalize_counts(counts: Dict[str, int]) -> Dict[str, int]:
    return {str(k).replace(" ", ""): int(v) for k, v in counts.items()}


def shannon_entropy_bits(counts: Dict[str, int]) -> float:
    total = sum(counts.values())
    if total <= 0:
        return 0.0
    ent = 0.0
    for c in counts.values():
        if c <= 0:
            continue
        p = c / total
        ent -= p * math.log2(p)
    return ent


def heavy_output_probability(counts: Dict[str, int]) -> float:
    """
    Proxy HOP: the share of results above the median of counts.
    For real RCS it requires an ideal distribution; here it is only a cloud-friendly proxy.
    """
    if not counts:
        return 0.0
    vals = sorted(counts.values())
    med = vals[len(vals) // 2]
    total = sum(vals)
    heavy = sum(v for v in vals if v > med)
    return heavy / total if total else 0.0


def bell_fidelity_proxy(counts: Dict[str, int], qubits: int) -> float:
    """For Bell: success = 00..0 or the ending of two qubits is correlated."""
    if qubits < 2 or not counts:
        return 0.0
    total = sum(counts.values())
    good = 0
    for bitstr, c in counts.items():
        s = bitstr.replace(" ", "")
        # Qiskit returns little-endian bits in the classical string; for the proxy the correlation of the extreme 2 bits is sufficient.
        if len(s) >= 2 and s[-1] == s[-2]:
            good += c
    return good / total if total else 0.0


def metrics_from_counts(counts: Optional[Dict[str, int]], benchmark: str, qubits: int) -> Dict[str, Any]:
    if counts is None:
        return {}
    counts = normalize_counts(counts)
    return {
        "unique_bitstrings": len(counts),
        "total_shots": sum(counts.values()),
        "shannon_entropy_bits": shannon_entropy_bits(counts),
        "heavy_output_probability_proxy": heavy_output_probability(counts),
        "bell_fidelity_proxy": bell_fidelity_proxy(counts, qubits) if benchmark.lower() == "bell" else None,
    }


# ============================================================
# 4. PROVIDERS
# ============================================================

def dry_run_descriptor(config: QuantumJobConfig) -> Dict[str, Any]:
    """Fallback without Qiskit: a circuit description sufficient for cloud validation."""
    b = config.benchmark.lower()
    if b == "bell":
        operations = {"h": 1, "cx": 1, "measure": config.qubits}
        depth = 3
        width = config.qubits
    elif b == "ghz":
        operations = {"h": 1, "cx": max(0, config.qubits - 1), "measure": config.qubits}
        depth = max(3, config.qubits + 1)
        width = config.qubits
    elif b == "rcs":
        twoq_per_layer = config.qubits // 2
        operations = {
            "random_1q": config.qubits * config.depth,
            "nearest_neighbor_2q": twoq_per_layer * config.depth,
            "measure": config.qubits,
        }
        depth = 2 * config.depth + 1
        width = config.qubits
    elif b in {"surface", "surface-proxy", "qec"}:
        d = config.qubits
        data = d * d
        measure = data - 1
        operations = {"h_data": data, "parity_cx": 2 * measure, "measure_syndrome": measure}
        depth = 4
        width = data + measure
    else:
        operations = {}
        depth = config.depth
        width = config.qubits

    return {
        "circuit_depth": depth,
        "circuit_width": width,
        "classical_bits": width if b != "surface-proxy" else max(0, width - config.qubits * config.qubits),
        "operations": operations,
        "qasm_preview": "Qiskit not installed; dry-run descriptor generated without QASM.",
        "requires_qiskit_for_qasm": True,
    }


def run_dry(config: QuantumJobConfig) -> QuantumJobResult:
    try:
        qc = build_qiskit_circuit(config)
        metrics = {
            "circuit_depth": qc.depth(),
            "circuit_width": qc.num_qubits,
            "classical_bits": qc.num_clbits,
            "operations": dict(qc.count_ops()),
            "qasm_preview": None,
            "requires_qiskit_for_qasm": False,
        }
        try:
            metrics["qasm_preview"] = qc.qasm()[:2000]
        except Exception:
            metrics["qasm_preview"] = "QASM export unavailable in this qiskit version."
        message = "Circuit built correctly with Qiskit. Not sent to the cloud."
    except RuntimeError:
        metrics = dry_run_descriptor(config)
        message = "Dry-run executed without Qiskit. Not sent to the cloud."

    return QuantumJobResult(
        status="dry-run",
        provider=config.provider,
        backend=config.backend,
        benchmark=config.benchmark,
        qubits=config.qubits,
        depth=config.depth,
        shots=config.shots,
        job_id=None,
        counts=None,
        metrics=metrics,
        message=message,
    )


def run_local(config: QuantumJobConfig) -> QuantumJobResult:
    qc = build_qiskit_circuit(config)
    try:
        from qiskit_aer import AerSimulator
        backend = AerSimulator()
        job = backend.run(qc, shots=config.shots)
        result = job.result()
        counts = normalize_counts(result.get_counts())
    except ModuleNotFoundError:
        # Fallback to BasicSimulator, if available in qiskit.
        try:
            from qiskit.providers.basic_provider import BasicSimulator
            backend = BasicSimulator()
            job = backend.run(qc, shots=config.shots)
            result = job.result()
            counts = normalize_counts(result.get_counts())
        except Exception as exc:
            raise RuntimeError(
                "No local simulator available. Install: pip install qiskit-aer"
            ) from exc

    return QuantumJobResult(
        status="completed",
        provider="local",
        backend="qiskit-aer/basic-simulator",
        benchmark=config.benchmark,
        qubits=config.qubits,
        depth=config.depth,
        shots=config.shots,
        job_id=None,
        counts=counts,
        metrics=metrics_from_counts(counts, config.benchmark, config.qubits),
        message="Local benchmark completed.",
    )


def run_ibm(config: QuantumJobConfig) -> QuantumJobResult:
    token = os.getenv("IBM_QUANTUM_TOKEN")
    instance = os.getenv("IBM_QUANTUM_INSTANCE")
    if not token:
        raise RuntimeError("IBM_QUANTUM_TOKEN is missing from environment variables.")
    if not config.backend:
        raise RuntimeError("Provide --backend, e.g. ibm_brisbane, ibm_kyiv, ibm_sherbrooke.")

    qc = build_qiskit_circuit(config)

    try:
        from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
        from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2 as Sampler
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "IBM Runtime is missing. Install: pip install qiskit qiskit-ibm-runtime"
        ) from exc

    service_kwargs = {"channel": "ibm_quantum", "token": token}
    if instance:
        service_kwargs["instance"] = instance
    service = QiskitRuntimeService(**service_kwargs)
    backend = service.backend(config.backend)

    pm = generate_preset_pass_manager(backend=backend, optimization_level=config.optimization_level)
    isa_circuit = pm.run(qc)

    sampler = Sampler(mode=backend)
    job = sampler.run([isa_circuit], shots=config.shots)
    result = job.result()

    counts = None
    try:
        # Qiskit Runtime V2: result[0].data.<classical_register>.get_counts()
        pub_result = result[0]
        data = pub_result.data
        # take the first classical register
        reg_name = list(data.keys())[0]
        counts = normalize_counts(data[reg_name].get_counts())
    except Exception:
        counts = None

    return QuantumJobResult(
        status="submitted/completed" if counts is not None else "submitted",
        provider="ibm",
        backend=config.backend,
        benchmark=config.benchmark,
        qubits=config.qubits,
        depth=config.depth,
        shots=config.shots,
        job_id=str(job.job_id()),
        counts=counts,
        metrics=metrics_from_counts(counts, config.benchmark, config.qubits),
        message="IBM job submitted. If counts=None, read the result by job_id in the IBM Quantum dashboard.",
    )


def run_aws(config: QuantumJobConfig) -> QuantumJobResult:
    if not config.backend:
        raise RuntimeError("Provide the --backend ARN of the AWS Braket device.")
    bucket = os.getenv("BRAKET_S3_BUCKET")
    prefix = os.getenv("BRAKET_S3_PREFIX", "quantum-cloud-bridge")
    region = os.getenv("AWS_REGION", "us-east-1")
    if not bucket:
        raise RuntimeError("BRAKET_S3_BUCKET is missing. AWS Braket requires an S3 bucket for results.")

    # Build an independent Braket circuit for bell/ghz/rcs.
    try:
        from braket.aws import AwsDevice
        from braket.circuits import Circuit
    except ModuleNotFoundError as exc:
        raise RuntimeError("AWS Braket SDK is missing. Install: pip install amazon-braket-sdk") from exc

    b = config.benchmark.lower()
    circ = Circuit()
    if b == "bell":
        if config.qubits < 2:
            raise ValueError("Bell requires >=2 qubits.")
        circ.h(0).cnot(0, 1)
    elif b == "ghz":
        circ.h(0)
        for i in range(config.qubits - 1):
            circ.cnot(i, i + 1)
    elif b == "rcs":
        rng = random.Random(config.seed)
        for layer in range(config.depth):
            for q in range(config.qubits):
                if rng.random() < 0.5:
                    circ.h(q)
                else:
                    circ.rx(q, rng.uniform(-math.pi, math.pi))
            for q in range(layer % 2, config.qubits - 1, 2):
                circ.cnot(q, q + 1)
    else:
        raise ValueError("AWS path supports benchmarks: bell, ghz, rcs.")

    device = AwsDevice(config.backend, aws_session=None)
    task = device.run(circ, s3_destination_folder=(bucket, prefix), shots=config.shots)
    result = task.result()
    counts = normalize_counts(result.measurement_counts) if hasattr(result, "measurement_counts") else None

    return QuantumJobResult(
        status="completed" if counts is not None else "submitted",
        provider="aws",
        backend=config.backend,
        benchmark=config.benchmark,
        qubits=config.qubits,
        depth=config.depth,
        shots=config.shots,
        job_id=str(task.id),
        counts=counts,
        metrics=metrics_from_counts(counts, config.benchmark, config.qubits),
        message=f"AWS Braket task completed in region {region}.",
    )


def run_azure(config: QuantumJobConfig) -> QuantumJobResult:
    if not config.backend:
        raise RuntimeError("Provide --backend, e.g. ionq.simulator, ionq.qpu, quantinuum.sim.h1-1sc.")
    resource_id = os.getenv("AZURE_QUANTUM_RESOURCE_ID")
    location = os.getenv("AZURE_QUANTUM_LOCATION")
    if not resource_id or not location:
        raise RuntimeError("AZURE_QUANTUM_RESOURCE_ID or AZURE_QUANTUM_LOCATION is missing.")

    try:
        from azure.quantum import Workspace
        from azure.quantum.qiskit import AzureQuantumProvider
    except ModuleNotFoundError as exc:
        raise RuntimeError("Azure Quantum SDK is missing. Install: pip install azure-quantum qiskit") from exc

    qc = build_qiskit_circuit(config)
    workspace = Workspace(resource_id=resource_id, location=location)
    provider = AzureQuantumProvider(workspace)
    backend = provider.get_backend(config.backend)
    job = backend.run(qc, shots=config.shots)
    result = job.result()

    counts = None
    try:
        counts = normalize_counts(result.get_counts())
    except Exception:
        counts = None

    return QuantumJobResult(
        status="completed" if counts is not None else "submitted",
        provider="azure",
        backend=config.backend,
        benchmark=config.benchmark,
        qubits=config.qubits,
        depth=config.depth,
        shots=config.shots,
        job_id=str(job.id()),
        counts=counts,
        metrics=metrics_from_counts(counts, config.benchmark, config.qubits),
        message="Azure Quantum job executed or submitted.",
    )


# ============================================================
# 5. ROUTER
# ============================================================

def run_quantum_job(config: QuantumJobConfig) -> QuantumJobResult:
    provider = config.provider.lower()
    if config.dry_run or provider in {"dry", "dry-run", "none"}:
        config.provider = "dry-run"
        return run_dry(config)
    if provider == "local":
        return run_local(config)
    if provider == "ibm":
        return run_ibm(config)
    if provider == "aws":
        return run_aws(config)
    if provider == "azure":
        return run_azure(config)
    if provider in {"google", "willow", "google-willow"}:
        return QuantumJobResult(
            status="unavailable",
            provider=provider,
            backend=config.backend or "Google Willow",
            benchmark=config.benchmark,
            qubits=config.qubits,
            depth=config.depth,
            shots=config.shots,
            job_id=None,
            counts=None,
            metrics={},
            message=(
                "Google Willow is not a public cloud backend for arbitrary tasks. "
                "Use dry-run/local/ibm/aws/azure or export the circuit to QASM."
            ),
        )
    raise ValueError(f"Unsupported provider: {config.provider}")


# ============================================================
# 6. CLI
# ============================================================

def main() -> None:
    parser = argparse.ArgumentParser(description="Quantum Cloud Bridge")
    parser.add_argument("--provider", default="dry-run", choices=["dry-run", "local", "ibm", "aws", "azure", "google", "willow"])
    parser.add_argument("--backend", default=None)
    parser.add_argument("--benchmark", default="bell", choices=["bell", "ghz", "rcs", "surface-proxy", "qec"])
    parser.add_argument("--qubits", type=int, default=2, help="for surface-proxy it means distance, e.g. 3,5,7")
    parser.add_argument("--depth", type=int, default=8)
    parser.add_argument("--shots", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--optimization-level", type=int, default=1)
    parser.add_argument("--output", default="quantum_cloud_job_result.json")
    args = parser.parse_args()

    config = QuantumJobConfig(
        provider=args.provider,
        backend=args.backend,
        benchmark=args.benchmark,
        qubits=args.qubits,
        depth=args.depth,
        shots=args.shots,
        seed=args.seed,
        optimization_level=args.optimization_level,
    )

    t0 = time.perf_counter()
    try:
        result = run_quantum_job(config)
    except Exception as exc:
        result = QuantumJobResult(
            status="error",
            provider=config.provider,
            backend=config.backend,
            benchmark=config.benchmark,
            qubits=config.qubits,
            depth=config.depth,
            shots=config.shots,
            job_id=None,
            counts=None,
            metrics={"error_type": type(exc).__name__},
            message=str(exc),
        )
    elapsed = time.perf_counter() - t0

    payload = asdict(result)
    payload["elapsed_seconds"] = elapsed
    payload["config"] = asdict(config)

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    print(json.dumps(payload, indent=2, ensure_ascii=False))
    print(f"\nWrote: {args.output}")


if __name__ == "__main__":
    main()
