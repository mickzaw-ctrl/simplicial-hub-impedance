"""
Neuro-Qubit System 1M
=====================

Scalable neuro-qubit system having EXACTLY 1,000,000 logical qubits.

This is not a full simulation of the state vector of 2^1,000,000 amplitudes — such
a simulation is impossible on a classical computer. Instead, we use an efficient
representation:

- 1 logical qubit = 1 local Bloch vector [x, y, z],
- global entanglement = sparse correlation graph between blocks,
- local dynamics = small neural networks operating block-wise/in chunks,
- physical loss = free energy F = E - T*S,
- number of logical qubits = exactly 1,000,000.

Run:
    python neuro_qubit_system_1m.py

Requirements:
    pip install torch
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple

import random

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except ModuleNotFoundError as exc:
    raise SystemExit(
        "This system requires PyTorch. Install: pip install torch\n"
        f"Details: {exc}"
    )


# ============================================================
# 1. MAP OF 1,000,000 LOGICAL QUBITS
# ============================================================

@dataclass(frozen=True)
class QubitBlockSpec:
    name: str
    qubits: int
    role: str


QUBIT_LAYOUT: List[QubitBlockSpec] = [
    QubitBlockSpec(
        name="boundary_encoder",
        qubits=163_840,
        role="Right Core / holographic boundary / information compression",
    ),
    QubitBlockSpec(
        name="variational_core",
        qubits=327_680,
        role="Main variational core / free energy minimization",
    ),
    QubitBlockSpec(
        name="memory_reservoir",
        qubits=245_760,
        role="Quantum-neural memory reservoir / long-term states",
    ),
    QubitBlockSpec(
        name="bulk_decoder",
        qubits=163_840,
        role="Left Core / bulk / geometry reconstruction from boundary code",
    ),
    QubitBlockSpec(
        name="entanglement_bus",
        qubits=40_960,
        role="Entanglement bus / correlations between blocks",
    ),
    QubitBlockSpec(
        name="autonomic_scaler",
        qubits=20_480,
        role="Adaptive controller / split-merge of cubics / RL decisions",
    ),
    QubitBlockSpec(
        name="error_syndrome_ancilla",
        qubits=37_440,
        role="Ancillary qubits / error syndromes / state stabilization",
    ),
]

TOTAL_QUBITS = sum(block.qubits for block in QUBIT_LAYOUT)
assert TOTAL_QUBITS == 1_000_000, f"System has {TOTAL_QUBITS}, but should have 1,000,000 qubits."


# ============================================================
# 2. NEURO-QUBIT BLOCK
# ============================================================

class NeuroQubitBlock(nn.Module):
    """
    A block of logical qubits described by Bloch vectors.

    State:
        bloch[i] = [x_i, y_i, z_i]

    Physical condition:
        ||bloch[i]|| <= 1

    Interpretation:
        z = +1 -> dominance of |0>,
        z = -1 -> dominance of |1>,
        x/y -> phase coherence.
    """

    def __init__(
        self,
        n_qubits: int,
        hidden_dim: int = 32,
        chunk_size: int = 65_536,
        device: str = "cpu",
    ) -> None:
        super().__init__()
        self.n_qubits = int(n_qubits)
        self.chunk_size = int(chunk_size)
        self.device = torch.device(device)

        init = torch.zeros(self.n_qubits, 3, device=self.device)
        init[:, 2] = 1.0
        self.register_buffer("bloch", init)

        # Small neural network, shared by qubits in the block.
        # hidden_dim=32 reduces cost for the 1M system.
        self.local_update = nn.Sequential(
            nn.Linear(3, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 3),
        )
        self.gate = nn.Sequential(
            nn.Linear(3, max(8, hidden_dim // 2)),
            nn.SiLU(),
            nn.Linear(max(8, hidden_dim // 2), 1),
            nn.Sigmoid(),
        )

    def _slices(self) -> Iterable[slice]:
        for start in range(0, self.n_qubits, self.chunk_size):
            end = min(start + self.chunk_size, self.n_qubits)
            yield slice(start, end)

    def differentiable_next_state(
        self,
        x: torch.Tensor,
        external_field: torch.Tensor | None = None,
    ) -> torch.Tensor:
        delta = self.local_update(x)
        if external_field is not None:
            delta = delta + external_field.to(x.device)

        candidate = F.normalize(x + 0.05 * delta, dim=-1, eps=1e-8)
        alpha = self.gate(x)
        new_state = F.normalize((1.0 - alpha) * x + alpha * candidate, dim=-1, eps=1e-8)
        return new_state

    @torch.no_grad()
    def commit_tick(self, external_field: torch.Tensor | None = None) -> None:
        """
        Fast state update without building the autograd graph.
        Used for evolving 1M qubits with low memory consumption.
        """
        for sl in self._slices():
            x = self.bloch[sl]
            delta = self.local_update(x)
            if external_field is not None:
                delta = delta + external_field.to(x.device)

            candidate = F.normalize(x + 0.05 * delta, dim=-1, eps=1e-8)
            alpha = self.gate(x)
            self.bloch[sl] = F.normalize((1.0 - alpha) * x + alpha * candidate, dim=-1, eps=1e-8)

    def sampled_training_loss(
        self,
        external_field: torch.Tensor | None = None,
        sample_size: int = 4096,
    ) -> torch.Tensor:
        """
        Differentiable training sample. We do not train with autograd through all 1M
        qubits at once, but through a random sample from each block.
        """
        n = min(sample_size, self.n_qubits)
        idx = torch.randint(0, self.n_qubits, (n,), device=self.bloch.device)
        x = self.bloch[idx].detach()
        y = self.differentiable_next_state(x, external_field=external_field)

        # Excitation energy after the step.
        p1 = (1.0 - y[:, 2].clamp(-1.0, 1.0)) / 2.0
        excitation = p1.mean()

        # Penalty for loss of physical normalization.
        norm_penalty = (y.norm(dim=-1).clamp(min=0.0) - 1.0).pow(2).mean()

        # Small penalty for abrupt movements to keep dynamics stable.
        smooth_motion = (y - x).pow(2).mean()

        return excitation + 0.01 * norm_penalty + 0.001 * smooth_motion

    def entropy_proxy(self) -> torch.Tensor:
        r = self.bloch.norm(dim=-1).clamp(0.0, 1.0)
        p0 = ((1.0 + r) / 2.0).clamp(1e-8, 1.0)
        p1 = ((1.0 - r) / 2.0).clamp(1e-8, 1.0)
        return -(p0 * torch.log(p0) + p1 * torch.log(p1)).mean()

    def excitation_density(self) -> torch.Tensor:
        z = self.bloch[:, 2].clamp(-1.0, 1.0)
        p1 = (1.0 - z) / 2.0
        return p1.mean()

    def mean_bloch(self) -> torch.Tensor:
        return self.bloch.mean(dim=0)


# ============================================================
# 3. SPARSE BLOCK ENTANGLEMENT GRAPH
# ============================================================

class SparseEntanglementGraph(nn.Module):
    """
    Sparse correlation graph between blocks.

    We do not create a 1,000,000 x 1,000,000 matrix.
    Global entanglement is modeled as a graph between functional regions.
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
# 4. 1M SYSTEM: NEURO-QUBIT REALITY ENGINE
# ============================================================

class NeuroQubitSystem1M(nn.Module):
    def __init__(
        self,
        device: str = "cpu",
        hidden_dim: int = 32,
        chunk_size: int = 65_536,
        training_sample_per_block: int = 4096,
    ) -> None:
        super().__init__()
        self.device = torch.device(device)
        self.training_sample_per_block = int(training_sample_per_block)

        self.blocks = nn.ModuleDict(
            {
                spec.name: NeuroQubitBlock(
                    spec.qubits,
                    hidden_dim=hidden_dim,
                    chunk_size=chunk_size,
                    device=device,
                )
                for spec in QUBIT_LAYOUT
            }
        )
        self.entanglement = SparseEntanglementGraph(
            block_names=[spec.name for spec in QUBIT_LAYOUT],
            device=device,
        )

        self.log_temperature = nn.Parameter(torch.tensor(0.0, device=self.device))
        self.energy_bias = nn.Parameter(torch.tensor(0.0, device=self.device))

    @property
    def total_qubits(self) -> int:
        return TOTAL_QUBITS

    def block_means(self) -> Dict[str, torch.Tensor]:
        return {name: block.mean_bloch() for name, block in self.blocks.items()}

    def physics_loss(self) -> Dict[str, torch.Tensor]:
        means = self.block_means()
        E_ent = self.entanglement.entanglement_energy(means)

        densities = torch.stack([block.excitation_density() for block in self.blocks.values()])
        E_exc = densities.mean()

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

    def training_loss(self) -> Dict[str, torch.Tensor]:
        """
        Differentiable training loss on samples from each block.
        Allows training neural parameters without storing autograd for 1M qubits.
        """
        means = self.block_means()
        fields = self.entanglement.fields(means)

        local_losses = []
        for name, block in self.blocks.items():
            local_losses.append(
                block.sampled_training_loss(
                    external_field=fields[name],
                    sample_size=self.training_sample_per_block,
                )
            )

        physical = self.physics_loss()
        local = torch.stack(local_losses).mean()
        F_train = physical["F"] + 0.05 * local

        out = dict(physical)
        out["Local_Neural_Loss"] = local
        out["Train_Loss"] = F_train
        return out

    @torch.no_grad()
    def tick(self) -> Dict[str, float]:
        """One full evolution step of all 1,000,000 qubits."""
        means_before = self.block_means()
        fields = self.entanglement.fields(means_before)

        for name, block in self.blocks.items():
            block.commit_tick(external_field=fields[name])

        loss = self.physics_loss()
        return {k: float(v.detach().cpu()) for k, v in loss.items()}

    def train_tick(self, optimizer: torch.optim.Optimizer) -> Dict[str, float]:
        """
        One learning step + one evolution step.
        1. Trains parameters on samples.
        2. Commits evolution of all 1M qubits without autograd.
        """
        optimizer.zero_grad(set_to_none=True)
        loss_dict = self.training_loss()
        loss = loss_dict["Train_Loss"]
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
        optimizer.step()

        tick_metrics = self.tick()

        metrics = {k: float(v.detach().cpu()) for k, v in loss_dict.items()}
        metrics.update({f"post_tick_{k}": v for k, v in tick_metrics.items()})
        return metrics

    def summary(self) -> str:
        lines = []
        lines.append("NEURO-QUBIT SYSTEM 1M")
        lines.append("=" * 32)
        lines.append(f"Total number of logical qubits: {self.total_qubits:,}".replace(",", " "))
        lines.append("")
        for spec in QUBIT_LAYOUT:
            lines.append(
                f"- {spec.name:24s} | {spec.qubits:9,d} qubits | {spec.role}".replace(",", " ")
            )
        return "\n".join(lines)


# ============================================================
# 5. DEMO
# ============================================================

def main() -> None:
    torch.manual_seed(7)
    random.seed(7)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    system = NeuroQubitSystem1M(
        device=device,
        hidden_dim=32,
        chunk_size=65_536,
        training_sample_per_block=4096,
    ).to(device)

    print(system.summary())
    print(f"\nCompute device: {device}")
    print("Starting evolution of the 1,000,000 neuro-qubit system...\n")

    optimizer = torch.optim.Adam(system.parameters(), lr=1e-3)

    for step in range(10):
        metrics = system.train_tick(optimizer)
        print(
            f"tick={step:03d} | "
            f"TrainLoss={metrics['Train_Loss']: .6f} | "
            f"F={metrics['F']: .6f} | "
            f"E_ent={metrics['E_entanglement']: .6f} | "
            f"E_exc={metrics['E_excitation']: .6f} | "
            f"S={metrics['S']: .6f} | "
            f"T={metrics['T']: .6f} | "
            f"postF={metrics['post_tick_F']: .6f}"
        )


if __name__ == "__main__":
    main()
