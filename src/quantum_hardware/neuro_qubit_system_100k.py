"""
Neuro-Qubit System 100K
=======================

A scalable, neuro-physical system with a total of 100,000 logical qubits.

IMPORTANT:
We do not simulate the full state vector of 100,000 qubits, because that would
require 2^100000 amplitudes. Instead we use a physically supported representation:
- each qubit is described locally by a Bloch vector: [x, y, z],
- entanglement is modeled through a sparse correlation graph,
- neural layers update the local state and coupling gates,
- the number of logical qubits is exactly 100,000.

Usage:
    python neuro_qubit_system_100k.py

If you have PyTorch:
    pip install torch
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import math
import random

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except ModuleNotFoundError as exc:  # clear message when the environment has no PyTorch
    raise SystemExit(
        "This system requires PyTorch. Install with: pip install torch\n"
        f"Details: {exc}"
    )


# ============================================================
# 1. MAP OF 100,000 LOGICAL QUBITS
# ============================================================

@dataclass(frozen=True)
class QubitBlockSpec:
    name: str
    qubits: int
    role: str


QUBIT_LAYOUT: List[QubitBlockSpec] = [
    QubitBlockSpec(
        name="boundary_encoder",
        qubits=16_384,
        role="Right Core / holographic boundary / information compression",
    ),
    QubitBlockSpec(
        name="variational_core",
        qubits=32_768,
        role="Main variational core / free energy minimization",
    ),
    QubitBlockSpec(
        name="memory_reservoir",
        qubits=24_576,
        role="Quantum-neural memory reservoir / long-term states",
    ),
    QubitBlockSpec(
        name="bulk_decoder",
        qubits=16_384,
        role="Left Core / bulk / geometry reconstruction from boundary code",
    ),
    QubitBlockSpec(
        name="entanglement_bus",
        qubits=4_096,
        role="Entanglement bus / correlations between blocks",
    ),
    QubitBlockSpec(
        name="autonomic_scaler",
        qubits=2_048,
        role="Adaptive controller / split-merge of qubits / RL decisions",
    ),
    QubitBlockSpec(
        name="error_syndrome_ancilla",
        qubits=3_744,
        role="Ancillary qubits / error syndromes / state stabilization",
    ),
]

TOTAL_QUBITS = sum(block.qubits for block in QUBIT_LAYOUT)
assert TOTAL_QUBITS == 100_000, f"The system has {TOTAL_QUBITS}, but it should have 100000 qubits."


# ============================================================
# 2. NEURO-QUBIT BLOCK
# ============================================================

class NeuroQubitBlock(nn.Module):
    """
    A block of logical qubits represented by Bloch vectors.

    State:
        bloch[i] = [x_i, y_i, z_i]

    Normalization:
        ||bloch[i]|| <= 1

    Interpretation:
        z = +1 means a predominance of |0>,
        z = -1 means a predominance of |1>,
        x/y describe the phase coherence components.
    """

    def __init__(self, n_qubits: int, hidden_dim: int = 64, device: str = "cpu") -> None:
        super().__init__()
        self.n_qubits = int(n_qubits)
        self.device = torch.device(device)

        # The Bloch state is not a model parameter — it is the dynamic state of the system.
        init = torch.zeros(self.n_qubits, 3, device=self.device)
        init[:, 2] = 1.0  # start near |0>
        self.register_buffer("bloch", init)

        # Neural update of the local qubit dynamics.
        self.local_update = nn.Sequential(
            nn.Linear(3, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 3),
        )

        # Mixing gate: how much of the old state to keep, how much of the new to admit.
        self.gate = nn.Sequential(
            nn.Linear(3, hidden_dim // 2),
            nn.SiLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid(),
        )

    def forward(self, external_field: torch.Tensor | None = None) -> torch.Tensor:
        x = self.bloch

        delta = self.local_update(x)
        if external_field is not None:
            # external_field can have shape [3] or [n_qubits, 3]
            delta = delta + external_field.to(x.device)

        candidate = F.normalize(x + 0.05 * delta, dim=-1, eps=1e-8)
        alpha = self.gate(x)
        new_state = F.normalize((1.0 - alpha) * x + alpha * candidate, dim=-1, eps=1e-8)

        # Update of the system state without treating it like network weights.
        self.bloch = new_state.detach()
        return self.bloch

    def entropy_proxy(self) -> torch.Tensor:
        """
        Local approximate entropy indicator.
        For a pure Bloch state ||r||≈1 entropy is small.
        For a mixed state ||r||<1 entropy grows.
        """
        r = self.bloch.norm(dim=-1).clamp(0.0, 1.0)
        p0 = ((1.0 + r) / 2.0).clamp(1e-8, 1.0)
        p1 = ((1.0 - r) / 2.0).clamp(1e-8, 1.0)
        return -(p0 * torch.log(p0) + p1 * torch.log(p1)).mean()

    def excitation_density(self) -> torch.Tensor:
        """Excitation density: probability of the |1> component."""
        z = self.bloch[:, 2].clamp(-1.0, 1.0)
        p1 = (1.0 - z) / 2.0
        return p1.mean()


# ============================================================
# 3. SPARSE ENTANGLEMENT GRAPH
# ============================================================

class SparseEntanglementGraph(nn.Module):
    """
    A sparse correlation graph between blocks.

    We do not create a 100000 x 100000 matrix, only a list of edges between blocks.
    Each edge carries a mean correlation field from one block to another.
    """

    def __init__(self, block_names: List[str], device: str = "cpu") -> None:
        super().__init__()
        self.block_names = list(block_names)
        self.device = torch.device(device)

        edges: List[Tuple[int, int]] = []
        name_to_i = {name: i for i, name in enumerate(self.block_names)}

        def edge(a: str, b: str) -> None:
            edges.append((name_to_i[a], name_to_i[b]))
            edges.append((name_to_i[b], name_to_i[a]))

        # Holographic-neural topology.
        edge("boundary_encoder", "variational_core")
        edge("variational_core", "bulk_decoder")
        edge("variational_core", "memory_reservoir")
        edge("memory_reservoir", "entanglement_bus")
        edge("entanglement_bus", "bulk_decoder")
        edge("autonomic_scaler", "variational_core")
        edge("error_syndrome_ancilla", "variational_core")
        edge("error_syndrome_ancilla", "entanglement_bus")

        self.register_buffer("edges", torch.tensor(edges, dtype=torch.long, device=self.device))
        self.edge_strength = nn.Parameter(torch.full((len(edges), 1), 0.05, device=self.device))

    def fields(self, block_means: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        means = torch.stack([block_means[name] for name in self.block_names], dim=0)
        out = {name: torch.zeros(3, device=means.device) for name in self.block_names}

        for e, (src, dst) in enumerate(self.edges.tolist()):
            dst_name = self.block_names[dst]
            strength = torch.tanh(self.edge_strength[e])
            out[dst_name] = out[dst_name] + strength * means[src]

        return out

    def entanglement_energy(self, block_means: Dict[str, torch.Tensor]) -> torch.Tensor:
        means = torch.stack([block_means[name] for name in self.block_names], dim=0)
        energy = torch.zeros((), device=means.device)
        for e, (src, dst) in enumerate(self.edges.tolist()):
            corr = torch.dot(means[src], means[dst])
            strength = torch.tanh(self.edge_strength[e]).squeeze()
            energy = energy - strength * corr
        return energy / max(1, len(self.edges))


# ============================================================
# 4. 100K SYSTEM: NEURO-QUBIT REALITY ENGINE
# ============================================================

class NeuroQubitSystem100K(nn.Module):
    def __init__(self, device: str = "cpu") -> None:
        super().__init__()
        self.device = torch.device(device)

        self.blocks = nn.ModuleDict(
            {
                spec.name: NeuroQubitBlock(spec.qubits, device=device)
                for spec in QUBIT_LAYOUT
            }
        )
        self.entanglement = SparseEntanglementGraph(
            block_names=[spec.name for spec in QUBIT_LAYOUT],
            device=device,
        )

        # Temperature and couplings are left as physical/learnable parameters.
        self.log_temperature = nn.Parameter(torch.tensor(0.0, device=self.device))
        self.energy_bias = nn.Parameter(torch.tensor(0.0, device=self.device))

    @property
    def total_qubits(self) -> int:
        return TOTAL_QUBITS

    def block_means(self) -> Dict[str, torch.Tensor]:
        return {
            name: block.bloch.mean(dim=0)
            for name, block in self.blocks.items()
        }

    def physics_loss(self) -> Dict[str, torch.Tensor]:
        means = self.block_means()

        # Entanglement energy between blocks.
        E_ent = self.entanglement.entanglement_energy(means)

        # Excitation energy — the system prefers low-chaos but not dead states.
        densities = torch.stack([block.excitation_density() for block in self.blocks.values()])
        E_exc = densities.mean()

        # Local entropy.
        entropies = torch.stack([block.entropy_proxy() for block in self.blocks.values()])
        S = entropies.mean()

        T = torch.exp(self.log_temperature).clamp(1e-4, 100.0)
        F_free = E_ent + E_exc + self.energy_bias - T * S

        return {
            "F": F_free,
            "E_entanglement": E_ent,
            "E_excitation": E_exc,
            "S": S,
            "T": T,
        }

    def tick(self) -> Dict[str, float]:
        """One evolution step of the system."""
        means_before = self.block_means()
        fields = self.entanglement.fields(means_before)

        for name, block in self.blocks.items():
            block(external_field=fields[name])

        loss = self.physics_loss()
        return {k: float(v.detach().cpu()) for k, v in loss.items()}

    def summary(self) -> str:
        lines = []
        lines.append("NEURO-QUBIT SYSTEM 100K")
        lines.append("=" * 32)
        lines.append(f"Total number of logical qubits: {self.total_qubits:,}".replace(",", " "))
        lines.append("")
        for spec in QUBIT_LAYOUT:
            lines.append(f"- {spec.name:24s} | {spec.qubits:7,d} qubits | {spec.role}".replace(",", " "))
        return "\n".join(lines)


# ============================================================
# 5. DEMO
# ============================================================

def main() -> None:
    torch.manual_seed(7)
    random.seed(7)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    system = NeuroQubitSystem100K(device=device).to(device)

    print(system.summary())
    print(f"\nCompute device: {device}")
    print("Starting neuro-qubit evolution...\n")

    optimizer = torch.optim.Adam(system.parameters(), lr=1e-3)

    for step in range(20):
        optimizer.zero_grad(set_to_none=True)

        # Update the dynamic states.
        metrics = system.tick()

        # Compute the loss after the update and train the couplings/blocks.
        loss_dict = system.physics_loss()
        loss = loss_dict["F"]
        loss.backward()
        torch.nn.utils.clip_grad_norm_(system.parameters(), max_norm=1.0)
        optimizer.step()

        if step % 2 == 0:
            print(
                f"tick={step:03d} | "
                f"F={float(loss_dict['F'].detach().cpu()): .6f} | "
                f"E_ent={float(loss_dict['E_entanglement'].detach().cpu()): .6f} | "
                f"E_exc={float(loss_dict['E_excitation'].detach().cpu()): .6f} | "
                f"S={float(loss_dict['S'].detach().cpu()): .6f} | "
                f"T={float(loss_dict['T'].detach().cpu()): .6f}"
            )


if __name__ == "__main__":
    main()
