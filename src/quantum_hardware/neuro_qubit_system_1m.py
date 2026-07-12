"""
Neuro-Qubit System 1M  [BCC-patched v2]
========================================

Scalable neuro-qubit system having EXACTLY 1,000,000 logical qubits.

This is not a full simulation of the state vector of 2^1,000,000 amplitudes — such
a simulation is impossible on a classical computer. Instead, we use an efficient
representation:

- 1 logical qubit = 1 local Bloch vector [x, y, z],
- global entanglement = sparse correlation graph between blocks,
- local dynamics = small neural networks operating block-wise/in chunks,
- physical loss = free energy F = E - T*S,
- number of logical qubits = exactly 1,000,000.

BCC-Cognitive patches (v2):
  [BCC-1] Starting temperature: log_temperature initialised to -2.0 → T₀ ≈ 0.135
          (safe quantum regime; original T=1.0 was at the coherence phase boundary)
  [BCC-2] T1/T2 physical decoherence in commit_tick:
            amplitude damping  bloch[:, 2] *= exp(-dt/T1)
            phase   damping    bloch[:, :2] *= exp(-dt/T2)
          with hardware-realistic defaults T1=68 µs, T2=89 µs (Google Willow metrics)
          and dt=1.0 ns per tick (tunable via DecoherenceParams).
  [BCC-3] AutonomicScaler feedback loop: NeuroQubitSystem1M.tick() now reads the
          autonomic_scaler block mean and writes a scaled feedback signal back to
          variational_core via SparseEntanglementGraph, closing the RL control loop.

Run:
    python neuro_qubit_system_1m_bcc.py

Requirements:
    pip install torch
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Tuple

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
# 0.  [BCC-2] DECOHERENCE PARAMETERS (T1 / T2)
# ============================================================

@dataclass
class DecoherenceParams:
    """
    Physical T1/T2 decoherence parameters applied each tick.

    T1  — amplitude damping (energy relaxation): Bloch z-component decays.
    T2  — phase damping (dephasing): Bloch x/y components decay.
    dt  — simulated physical time per tick in nanoseconds.

    Defaults mirror Google Willow public metrics (Nature 2024):
        T1 = 68 µs = 68_000 ns
        T2 = 89 µs = 89_000 ns  (T2,CPMG)
    """
    T1_ns: float = 68_000.0   # amplitude relaxation time [ns]
    T2_ns: float = 89_000.0   # phase coherence time [ns]
    dt_ns: float = 1.0        # simulated time per tick [ns]

    @property
    def decay_z(self) -> float:
        """Amplitude-damping factor for z-component per tick."""
        return math.exp(-self.dt_ns / self.T1_ns)

    @property
    def decay_xy(self) -> float:
        """Phase-damping factor for x/y components per tick."""
        return math.exp(-self.dt_ns / self.T2_ns)


# ============================================================
# 1.  MAP OF 1,000,000 LOGICAL QUBITS
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
        role="Adaptive RL controller / split-merge decisions / feedback to variational_core",
    ),
    QubitBlockSpec(
        name="error_syndrome_ancilla",
        qubits=37_440,
        role="Ancillary qubits / error syndromes / state stabilization",
    ),
]

TOTAL_QUBITS = sum(block.qubits for block in QUBIT_LAYOUT)
assert TOTAL_QUBITS == 1_000_000, (
    f"System has {TOTAL_QUBITS}, but should have 1,000,000 qubits."
)


# ============================================================
# 2.  NEURO-QUBIT BLOCK
# ============================================================

class NeuroQubitBlock(nn.Module):
    """
    A block of logical qubits described by Bloch vectors.

    State:
        bloch[i] = [x_i, y_i, z_i]

    Physical condition:
        ||bloch[i]|| <= 1

    Interpretation:
        z = +1  →  dominance of |0⟩
        z = -1  →  dominance of |1⟩
        x/y     →  phase coherence

    [BCC-2] Physical decoherence is applied in commit_tick() via a
    DecoherenceParams object passed from the parent system.
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

        # Initialise all qubits in |0⟩: Bloch vector = (0, 0, 1)
        init = torch.zeros(self.n_qubits, 3, device=self.device)
        init[:, 2] = 1.0
        self.register_buffer("bloch", init)

        # Small neural network shared across qubits in the block.
        # hidden_dim=32 keeps memory cost manageable for 1M qubits.
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

    # ----------------------------------------------------------
    def _slices(self) -> Iterable[slice]:
        for start in range(0, self.n_qubits, self.chunk_size):
            yield slice(start, min(start + self.chunk_size, self.n_qubits))

    # ----------------------------------------------------------
    def differentiable_next_state(
        self,
        x: torch.Tensor,
        external_field: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        delta = self.local_update(x)
        if external_field is not None:
            delta = delta + external_field.to(x.device)

        candidate = F.normalize(x + 0.05 * delta, dim=-1, eps=1e-8)
        alpha = self.gate(x)
        return F.normalize((1.0 - alpha) * x + alpha * candidate, dim=-1, eps=1e-8)

    # ----------------------------------------------------------
    @torch.no_grad()
    def commit_tick(
        self,
        external_field: Optional[torch.Tensor] = None,
        decoherence: Optional[DecoherenceParams] = None,   # [BCC-2]
    ) -> None:
        """
        Fast state update without building the autograd graph.
        Used for evolving 1M qubits with low memory consumption.

        [BCC-2] If a DecoherenceParams object is provided, applies physical
        T1 amplitude damping to the z-component and T2 phase damping to x/y
        after each chunk update, modelling superconducting qubit decoherence.
        """
        for sl in self._slices():
            x = self.bloch[sl]
            delta = self.local_update(x)
            if external_field is not None:
                delta = delta + external_field.to(x.device)

            candidate = F.normalize(x + 0.05 * delta, dim=-1, eps=1e-8)
            alpha = self.gate(x)
            new_state = F.normalize(
                (1.0 - alpha) * x + alpha * candidate, dim=-1, eps=1e-8
            )

            # [BCC-2] Apply T1 / T2 physical decoherence -----------------
            if decoherence is not None:
                # Phase damping: x, y components shrink at rate 1/T2
                new_state[:, 0] *= decoherence.decay_xy
                new_state[:, 1] *= decoherence.decay_xy
                # Amplitude damping: z drifts toward ground state at rate 1/T1
                new_state[:, 2] = (
                    1.0 - (1.0 - new_state[:, 2]) * decoherence.decay_z
                )
                # Re-normalise to keep ||bloch|| ≤ 1 after damping
                norms = new_state.norm(dim=-1, keepdim=True).clamp(min=1.0)
                new_state = new_state / norms
            # -------------------------------------------------------------

            self.bloch[sl] = new_state

    # ----------------------------------------------------------
    def sampled_training_loss(
        self,
        external_field: Optional[torch.Tensor] = None,
        sample_size: int = 4096,
    ) -> torch.Tensor:
        """
        Differentiable training loss on a random sample of qubits.
        Avoids materialising autograd graph for all 1M qubits at once.
        """
        n = min(sample_size, self.n_qubits)
        idx = torch.randint(0, self.n_qubits, (n,), device=self.bloch.device)
        x = self.bloch[idx].detach()
        y = self.differentiable_next_state(x, external_field=external_field)

        # Excitation energy after the step
        p1 = (1.0 - y[:, 2].clamp(-1.0, 1.0)) / 2.0
        excitation = p1.mean()

        # Penalty for loss of physical normalisation
        norm_penalty = (y.norm(dim=-1).clamp(min=0.0) - 1.0).pow(2).mean()

        # Small penalty for abrupt state changes (stability)
        smooth_motion = (y - x).pow(2).mean()

        return excitation + 0.01 * norm_penalty + 0.001 * smooth_motion

    # ----------------------------------------------------------
    def entropy_proxy(self) -> torch.Tensor:
        r = self.bloch.norm(dim=-1).clamp(0.0, 1.0)
        p0 = ((1.0 + r) / 2.0).clamp(1e-8, 1.0)
        p1 = ((1.0 - r) / 2.0).clamp(1e-8, 1.0)
        return -(p0 * torch.log(p0) + p1 * torch.log(p1)).mean()

    def excitation_density(self) -> torch.Tensor:
        z = self.bloch[:, 2].clamp(-1.0, 1.0)
        return ((1.0 - z) / 2.0).mean()

    def mean_bloch(self) -> torch.Tensor:
        return self.bloch.mean(dim=0)


# ============================================================
# 3.  SPARSE BLOCK ENTANGLEMENT GRAPH
# ============================================================

class SparseEntanglementGraph(nn.Module):
    """
    Sparse correlation graph between functional qubit blocks.

    We do not create a 1,000,000 × 1,000,000 matrix.
    Global entanglement is modelled as a learnable weighted graph between blocks.

    [BCC-3] Added a dedicated feedback edge:
        autonomic_scaler → variational_core  (RL control signal)
    This edge is present in the original edge list; the new inject_scaler_feedback()
    method applies an explicit, non-linear feedback signal at each tick so the
    AutonomicScaler block can actively modulate variational_core's external field.
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

        edge("boundary_encoder",     "variational_core")
        edge("variational_core",     "bulk_decoder")
        edge("variational_core",     "memory_reservoir")
        edge("memory_reservoir",     "entanglement_bus")
        edge("entanglement_bus",     "bulk_decoder")
        edge("autonomic_scaler",     "variational_core")   # [BCC-3] RL control loop
        edge("error_syndrome_ancilla", "variational_core")
        edge("error_syndrome_ancilla", "entanglement_bus")

        self.register_buffer(
            "edges", torch.tensor(edges, dtype=torch.long, device=self.device)
        )
        self.edge_strength = nn.Parameter(
            torch.full((len(edges), 1), 0.05, device=self.device)
        )

        # [BCC-3] Dedicated RL projection: autonomic_scaler (3-dim) → variational_core field (3-dim)
        self._scaler_idx = name_to_i["autonomic_scaler"]
        self._vcore_idx  = name_to_i["variational_core"]
        self.scaler_projection = nn.Linear(3, 3, bias=False)
        nn.init.eye_(self.scaler_projection.weight)   # identity init — neutral start

    # ----------------------------------------------------------
    def fields(self, block_means: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        means = torch.stack([block_means[name] for name in self.block_names], dim=0)
        out = {name: torch.zeros(3, device=means.device) for name in self.block_names}

        for e, (src, dst) in enumerate(self.edges.tolist()):
            dst_name = self.block_names[dst]
            strength = torch.tanh(self.edge_strength[e])
            out[dst_name] = out[dst_name] + strength * means[src]

        return out

    # ----------------------------------------------------------
    def inject_scaler_feedback(
        self,
        fields: Dict[str, torch.Tensor],
        block_means: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """
        [BCC-3] AutonomicScaler feedback loop.

        Reads the autonomic_scaler Bloch mean, projects it through a learnable
        linear layer (scaler_projection), and adds the result as an additional
        control signal to the variational_core external field.

        This closes the RL control loop:
            autonomic_scaler.bloch  →  scaler_projection  →  variational_core.field
        """
        scaler_mean = block_means["autonomic_scaler"].detach()           # [3]
        feedback = torch.tanh(self.scaler_projection(scaler_mean))       # [3], bounded
        fields["variational_core"] = fields["variational_core"] + 0.1 * feedback
        return fields

    # ----------------------------------------------------------
    def entanglement_energy(self, block_means: Dict[str, torch.Tensor]) -> torch.Tensor:
        means = torch.stack([block_means[name] for name in self.block_names], dim=0)
        energy = torch.zeros((), device=means.device)

        for e, (src, dst) in enumerate(self.edges.tolist()):
            corr = torch.dot(means[src], means[dst])
            strength = torch.tanh(self.edge_strength[e]).squeeze()
            energy = energy - strength * corr

        return energy / max(1, len(self.edges))


# ============================================================
# 4.  1M SYSTEM: NEURO-QUBIT REALITY ENGINE  (BCC-patched)
# ============================================================

class NeuroQubitSystem1M(nn.Module):
    """
    BCC-patched v2 changes vs original:

    [BCC-1] log_temperature = -2.0  →  T₀ ≈ 0.135  (safe quantum regime)
            Original T=1.0 sat exactly on the coherence/decoherence phase boundary.
    [BCC-2] commit_tick() passes DecoherenceParams to each block.
            Physical T1/T2 damping applied per tick with Willow-realistic defaults.
    [BCC-3] tick() calls entanglement.inject_scaler_feedback() to close the
            autonomic_scaler → variational_core RL loop before committing state.
    """

    def __init__(
        self,
        device: str = "cpu",
        hidden_dim: int = 32,
        chunk_size: int = 65_536,
        training_sample_per_block: int = 4096,
        decoherence: Optional[DecoherenceParams] = None,  # [BCC-2]
    ) -> None:
        super().__init__()
        self.device = torch.device(device)
        self.training_sample_per_block = int(training_sample_per_block)

        # [BCC-2] Store decoherence params (default: Willow T1=68µs, T2=89µs, dt=1ns)
        self.decoherence = decoherence if decoherence is not None else DecoherenceParams()

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

        # [BCC-1] Temperature fix: -2.0 → T₀ = exp(-2) ≈ 0.135
        #         Original was 0.0 → T₀ = 1.0 (phase boundary — unstable)
        self.log_temperature = nn.Parameter(
            torch.tensor(-2.0, device=self.device)   # [BCC-1] was: 0.0
        )
        self.energy_bias = nn.Parameter(torch.tensor(0.0, device=self.device))

    # ----------------------------------------------------------
    @property
    def total_qubits(self) -> int:
        return TOTAL_QUBITS

    def block_means(self) -> Dict[str, torch.Tensor]:
        return {name: block.mean_bloch() for name, block in self.blocks.items()}

    # ----------------------------------------------------------
    def physics_loss(self) -> Dict[str, torch.Tensor]:
        means = self.block_means()
        E_ent = self.entanglement.entanglement_energy(means)

        densities = torch.stack(
            [block.excitation_density() for block in self.blocks.values()]
        )
        E_exc = densities.mean()

        entropies = torch.stack(
            [block.entropy_proxy() for block in self.blocks.values()]
        )
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

    # ----------------------------------------------------------
    def training_loss(self) -> Dict[str, torch.Tensor]:
        """
        Differentiable training loss on samples from each block.
        Allows training neural parameters without storing autograd for 1M qubits.
        """
        means = self.block_means()
        fields = self.entanglement.fields(means)
        fields = self.entanglement.inject_scaler_feedback(fields, means)  # [BCC-3]

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

    # ----------------------------------------------------------
    @torch.no_grad()
    def tick(self) -> Dict[str, float]:
        """
        One full evolution step of all 1,000,000 qubits.

        [BCC-3] Reads autonomic_scaler mean and injects feedback into
                variational_core field before committing state updates.
        [BCC-2] Passes DecoherenceParams to each block's commit_tick().
        """
        means_before = self.block_means()
        fields = self.entanglement.fields(means_before)
        fields = self.entanglement.inject_scaler_feedback(fields, means_before)  # [BCC-3]

        for name, block in self.blocks.items():
            block.commit_tick(
                external_field=fields[name],
                decoherence=self.decoherence,   # [BCC-2]
            )

        loss = self.physics_loss()
        return {k: float(v.detach().cpu()) for k, v in loss.items()}

    # ----------------------------------------------------------
    def train_tick(self, optimizer: torch.optim.Optimizer) -> Dict[str, float]:
        """
        One learning step + one evolution step.
        1. Trains parameters on samples (with BCC-3 scaler feedback in loss).
        2. Commits evolution of all 1M qubits (with BCC-2 decoherence, BCC-3 feedback).
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

    # ----------------------------------------------------------
    def summary(self) -> str:
        T0 = math.exp(self.log_temperature.item())
        dec = self.decoherence
        lines = [
            "NEURO-QUBIT SYSTEM 1M  [BCC-patched v2]",
            "=" * 44,
            f"Total logical qubits : {self.total_qubits:>12,}".replace(",", " "),
            f"Temperature T₀       : {T0:>12.4f}  (log_T={self.log_temperature.item():.2f})",
            f"T1 decoherence       : {dec.T1_ns:>10.0f} ns",
            f"T2 decoherence       : {dec.T2_ns:>10.0f} ns",
            f"dt per tick          : {dec.dt_ns:>10.2f} ns",
            f"decay_z  (T1/tick)   : {dec.decay_z:>12.8f}",
            f"decay_xy (T2/tick)   : {dec.decay_xy:>12.8f}",
            "",
        ]
        for spec in QUBIT_LAYOUT:
            lines.append(
                f"  {spec.name:26s}  {spec.qubits:>9,d} qubits   {spec.role}".replace(",", " ")
            )
        return "\n".join(lines)


# ============================================================
# 5.  DEMO
# ============================================================

def main() -> None:
    torch.manual_seed(7)
    random.seed(7)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # [BCC-2] Custom decoherence: Willow defaults, 1 ns/tick
    dec = DecoherenceParams(T1_ns=68_000.0, T2_ns=89_000.0, dt_ns=1.0)

    system = NeuroQubitSystem1M(
        device=device,
        hidden_dim=32,
        chunk_size=65_536,
        training_sample_per_block=4096,
        decoherence=dec,          # [BCC-2]
    ).to(device)

    print(system.summary())
    print(f"\nCompute device: {device}")
    print("Starting BCC-patched evolution of 1,000,000 neuro-qubits...\n")
    print(
        f"{'tick':>4} | {'TrainLoss':>10} | {'F':>9} | {'E_ent':>8} | "
        f"{'E_exc':>8} | {'S':>8} | {'T':>7} | {'postF':>9}"
    )
    print("-" * 80)

    optimizer = torch.optim.Adam(system.parameters(), lr=1e-3)

    for step in range(10):
        m = system.train_tick(optimizer)
        print(
            f"{step:>4d} | "
            f"{m['Train_Loss']:>10.6f} | "
            f"{m['F']:>9.6f} | "
            f"{m['E_entanglement']:>8.6f} | "
            f"{m['E_excitation']:>8.6f} | "
            f"{m['S']:>8.6f} | "
            f"{m['T']:>7.4f} | "
            f"{m['post_tick_F']:>9.6f}"
        )


if __name__ == "__main__":
    main()
