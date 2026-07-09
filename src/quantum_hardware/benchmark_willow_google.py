"""
Benchmark: Willow-compatible Neuro-Qubit 1M vs Google Willow public metrics
===========================================================================

This benchmark does NOT claim that a classical ML simulator achieves the quantum
advantage of Google Willow. The goal is to compare architecture, scale, memory and
proxy-QEC with the publicly described Willow metrics.

Modes:
1. Without PyTorch: architectural and memory report.
2. With PyTorch: additionally a runtime benchmark of train_tick/tick of the 1M model.

Usage:
    python benchmark_willow_google.py

Optionally:
    python benchmark_willow_google.py --ticks 5 --sample 1024 --device cpu
    python benchmark_willow_google.py --ticks 20 --sample 4096 --device cuda

Results:
    willow_google_benchmark_results.json
    willow_google_benchmark_results.md
"""

from __future__ import annotations

import argparse
import json
import math
import platform
import statistics
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional


# ============================================================
# 1. PUBLIC GOOGLE WILLOW REFERENCE METRICS
# ============================================================

@dataclass(frozen=True)
class GoogleWillowReference:
    name: str = "Google Willow"
    physical_qubits: int = 105
    qubit_type: str = "superconducting transmon"
    topology: str = "2D square-grid / nearest-neighbor / tunable-coupler style"
    qec_code: str = "surface code"
    demonstrated_distances: str = "d=3, d=5, d=7"
    data_qubits_d7: int = 49
    t1_us_nature: float = 68.0
    t2_cpmg_us_nature: float = 89.0
    qec_threshold_result: str = "below surface-code threshold; logical error reduced by >2 when d increases by 2"
    rcs_benchmark: str = "Random Circuit Sampling in under 5 minutes; classical estimate 10 septillion years"
    reference_urls: tuple[str, ...] = (
        "https://www.nature.com/articles/s41586-024-08449-y",
        "https://blog.google/innovation-and-ai/technology/research/google-willow-quantum-chip/",
    )


# ============================================================
# 2. OUR WILLOW-COMPATIBLE 1M ARCHITECTURE METRICS
# ============================================================

@dataclass(frozen=True)
class WillowCompatible1MReference:
    name: str = "Willow-compatible Neuro-Qubit System 1M"
    total_qubits: int = 1_000_000
    willow_tile_qubits: int = 105
    full_tiles: int = 9_523
    reserve_edge_qubits: int = 85
    data_qubits_per_tile: int = 49
    measure_qubits_per_tile: int = 48
    leakage_qubits_per_tile: int = 8

    @property
    def data_qubit_plane(self) -> int:
        return self.full_tiles * self.data_qubits_per_tile

    @property
    def measure_stabilizer_plane(self) -> int:
        return self.full_tiles * self.measure_qubits_per_tile

    @property
    def leakage_reset_ancilla(self) -> int:
        return self.full_tiles * self.leakage_qubits_per_tile

    @property
    def exact_sum(self) -> int:
        return (
            self.data_qubit_plane
            + self.measure_stabilizer_plane
            + self.leakage_reset_ancilla
            + self.reserve_edge_qubits
        )

    @property
    def scale_vs_willow(self) -> float:
        return self.total_qubits / 105.0

    @property
    def bloch_state_memory_mb_float32(self) -> float:
        # 1M qubits * 3 Bloch components * 4 bytes
        return self.total_qubits * 3 * 4 / (1024**2)

    @property
    def full_state_amplitudes_log10(self) -> float:
        # log10(2^n)
        return self.total_qubits * math.log10(2)

    @property
    def full_state_memory_log10_bytes_complex64(self) -> float:
        # 2^n complex64 amplitudes * 8 bytes
        return self.total_qubits * math.log10(2) + math.log10(8)


# ============================================================
# 3. BENCHMARK HELPERS
# ============================================================

def architecture_report() -> Dict[str, Any]:
    google = GoogleWillowReference()
    ours = WillowCompatible1MReference()

    assert ours.exact_sum == ours.total_qubits, f"bad qubit count: {ours.exact_sum}"

    return {
        "google_willow_public_reference": asdict(google),
        "our_willow_compatible_1m": {
            "name": ours.name,
            "total_qubits": ours.total_qubits,
            "willow_tile_qubits": ours.willow_tile_qubits,
            "full_105q_tiles": ours.full_tiles,
            "reserve_edge_qubits": ours.reserve_edge_qubits,
            "data_qubit_plane": ours.data_qubit_plane,
            "measure_stabilizer_plane": ours.measure_stabilizer_plane,
            "leakage_reset_ancilla": ours.leakage_reset_ancilla,
            "exact_sum": ours.exact_sum,
            "scale_vs_single_willow_105q": ours.scale_vs_willow,
            "effective_bloch_state_memory_mb_float32": ours.bloch_state_memory_mb_float32,
            "full_amplitude_state_log10_amplitudes": ours.full_state_amplitudes_log10,
            "full_amplitude_state_log10_bytes_complex64": ours.full_state_memory_log10_bytes_complex64,
        },
        "interpretation": {
            "compatibility": "topology/QEC-role compatible, not hardware-identical",
            "not_claimed": "does not reproduce Willow RCS quantum advantage on classical hardware",
            "reason": "full quantum state would require 2^1,000,000 amplitudes",
        },
    }


def maybe_runtime_benchmark(device: str, ticks: int, sample: int, hidden_dim: int, chunk_size: int) -> Dict[str, Any]:
    """Run only if PyTorch and the local 1M Willow-compatible module are available."""
    try:
        import torch
    except ModuleNotFoundError:
        return {
            "runtime_status": "skipped",
            "reason": "PyTorch is not installed. Install with: pip install torch",
        }

    try:
        from willow_compatible_neuro_qubit_1m import WillowCompatibleNeuroQubitSystem1M, WillowPhysicalParams
    except BaseException as exc:
        return {
            "runtime_status": "skipped",
            "reason": f"Could not import willow_compatible_neuro_qubit_1m.py: {type(exc).__name__}: {exc}",
        }

    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"

    torch.manual_seed(7)

    system = WillowCompatibleNeuroQubitSystem1M(
        device=device,
        hidden_dim=hidden_dim,
        chunk_size=chunk_size,
        training_sample_per_role=sample,
        params=WillowPhysicalParams(),
    ).to(device)

    optimizer = torch.optim.Adam(system.parameters(), lr=1e-3)

    # Warm-up: one light training tick.
    warmup_start = time.perf_counter()
    warmup_metrics = system.train_tick(optimizer)
    warmup_seconds = time.perf_counter() - warmup_start

    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()

    train_times: List[float] = []
    tick_times: List[float] = []
    last_metrics: Dict[str, float] = {}

    for _ in range(ticks):
        t0 = time.perf_counter()
        last_metrics = system.train_tick(optimizer)
        if device == "cuda":
            torch.cuda.synchronize()
        train_times.append(time.perf_counter() - t0)

        t1 = time.perf_counter()
        post = system.tick()
        if device == "cuda":
            torch.cuda.synchronize()
        tick_times.append(time.perf_counter() - t1)
        last_metrics.update({f"extra_tick_{k}": v for k, v in post.items()})

    peak_cuda_mb = None
    if device == "cuda":
        peak_cuda_mb = torch.cuda.max_memory_allocated() / (1024**2)

    def stats(xs: List[float]) -> Dict[str, float]:
        return {
            "mean_s": statistics.mean(xs),
            "median_s": statistics.median(xs),
            "min_s": min(xs),
            "max_s": max(xs),
        }

    return {
        "runtime_status": "completed",
        "python": sys.version,
        "platform": platform.platform(),
        "torch_version": torch.__version__,
        "device": device,
        "ticks_measured": ticks,
        "training_sample_per_role": sample,
        "hidden_dim": hidden_dim,
        "chunk_size": chunk_size,
        "warmup_seconds": warmup_seconds,
        "train_tick_stats": stats(train_times),
        "state_tick_stats": stats(tick_times),
        "peak_cuda_memory_mb": peak_cuda_mb,
        "last_metrics": last_metrics,
    }


def qec_projection_report() -> Dict[str, Any]:
    """
    Simple surface-code scaling projection using Google's public qualitative Nature result:
    error is reduced by more than half for each +2 increase in distance.

    This is not a claim about our model's true logical error rate. It is a target curve
    for Willow-like QEC behavior.
    """
    base_d = 7
    base_logical_error_proxy = 1.0
    lambdas = [2.0, 2.2, 2.5, 3.0]
    distances = [7, 9, 11, 13, 15, 17, 21, 25]

    projections = {}
    for lam in lambdas:
        curve = {}
        for d in distances:
            steps = max(0, (d - base_d) // 2)
            curve[f"d={d}"] = base_logical_error_proxy / (lam ** steps)
        projections[f"lambda={lam}"] = curve

    return {
        "definition": "lambda = logical error suppression factor for each distance increase by +2",
        "google_willow_target": "Nature reports reduction by more than half for each +2 in code distance, i.e. lambda > 2",
        "normalized_projection_from_d7": projections,
    }


# ============================================================
# 4. OUTPUT
# ============================================================

def write_outputs(results: Dict[str, Any], json_path: Path, md_path: Path) -> None:
    json_path.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")

    arch = results["architecture"]
    g = arch["google_willow_public_reference"]
    o = arch["our_willow_compatible_1m"]
    rt = results["runtime"]

    lines = []
    lines.append("# Benchmark: Willow-compatible 1M vs Google Willow")
    lines.append("")
    lines.append("## Scope")
    lines.append("")
    lines.append("This benchmark compares **architecture and proxy metrics**. It does not claim that a classical ML simulator reproduces the actual quantum advantage of Google Willow.")
    lines.append("")
    lines.append("## Public reference point: Google Willow")
    lines.append("")
    lines.append(f"- Physical qubits: **{g['physical_qubits']}**")
    lines.append(f"- Type: **{g['qubit_type']}**")
    lines.append(f"- Topology: **{g['topology']}**")
    lines.append(f"- Error-correcting code: **{g['qec_code']}**, {g['demonstrated_distances']}")
    lines.append(f"- Data for d=7: **{g['data_qubits_d7']} data qubits**")
    lines.append(f"- T1: **{g['t1_us_nature']} µs**, T2,CPMG: **{g['t2_cpmg_us_nature']} µs**")
    lines.append(f"- QEC: **{g['qec_threshold_result']}**")
    lines.append(f"- RCS: **{g['rcs_benchmark']}**")
    lines.append("")
    lines.append("## Our Willow-compatible 1M model")
    lines.append("")
    lines.append(f"- Total number of qubits: **{o['total_qubits']:,}**".replace(",", " "))
    lines.append(f"- Full 105q tiles: **{o['full_105q_tiles']:,}**".replace(",", " "))
    lines.append(f"- Reserve/edge: **{o['reserve_edge_qubits']}**")
    lines.append(f"- Data plane: **{o['data_qubit_plane']:,}**".replace(",", " "))
    lines.append(f"- Measure/stabilizer plane: **{o['measure_stabilizer_plane']:,}**".replace(",", " "))
    lines.append(f"- Leakage/reset ancilla: **{o['leakage_reset_ancilla']:,}**".replace(",", " "))
    lines.append(f"- Scale relative to a single Willow 105q: **{o['scale_vs_single_willow_105q']:.2f}x**")
    lines.append(f"- Effective Bloch state memory float32: **{o['effective_bloch_state_memory_mb_float32']:.2f} MB**")
    lines.append(f"- Full amplitude state would have approximately **10^{o['full_amplitude_state_log10_amplitudes']:.0f}** amplitudes")
    lines.append("")
    lines.append("## Runtime")
    lines.append("")
    if rt.get("runtime_status") == "completed":
        lines.append(f"- Status: **completed**")
        lines.append(f"- Device: **{rt['device']}**")
        lines.append(f"- PyTorch: **{rt['torch_version']}**")
        lines.append(f"- Ticks measured: **{rt['ticks_measured']}**")
        lines.append(f"- train_tick mean: **{rt['train_tick_stats']['mean_s']:.6f} s**")
        lines.append(f"- state_tick mean: **{rt['state_tick_stats']['mean_s']:.6f} s**")
        if rt.get("peak_cuda_memory_mb") is not None:
            lines.append(f"- CUDA peak memory: **{rt['peak_cuda_memory_mb']:.2f} MB**")
        lines.append("")
        lines.append("### Last metrics")
        lines.append("")
        lines.append("```json")
        lines.append(json.dumps(rt.get("last_metrics", {}), indent=2, ensure_ascii=False))
        lines.append("```")
    else:
        lines.append(f"- Status: **{rt.get('runtime_status')}**")
        lines.append(f"- Reason: {rt.get('reason')}")
    lines.append("")
    lines.append("## QEC projection target")
    lines.append("")
    lines.append("Willow target: for a code distance increase of +2, the logical error should drop by more than 2x. In the benchmark this is recorded as `lambda > 2`.")
    lines.append("")
    lines.append("## Sources")
    lines.append("")
    for url in g["reference_urls"]:
        lines.append(f"- {url}")
    lines.append("")

    md_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ticks", type=int, default=3, help="number of measured runtime ticks")
    parser.add_argument("--sample", type=int, default=1024, help="training sample per role")
    parser.add_argument("--hidden-dim", type=int, default=32)
    parser.add_argument("--chunk-size", type=int, default=65_536)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--json", default="willow_google_benchmark_results.json")
    parser.add_argument("--md", default="willow_google_benchmark_results.md")
    args = parser.parse_args()

    results = {
        "architecture": architecture_report(),
        "qec_projection": qec_projection_report(),
        "runtime": maybe_runtime_benchmark(
            device=args.device,
            ticks=max(1, args.ticks),
            sample=max(1, args.sample),
            hidden_dim=args.hidden_dim,
            chunk_size=args.chunk_size,
        ),
    }

    write_outputs(results, Path(args.json), Path(args.md))
    print(json.dumps(results, indent=2, ensure_ascii=False))
    print(f"\nWrote: {args.json}")
    print(f"Wrote: {args.md}")


if __name__ == "__main__":
    main()
