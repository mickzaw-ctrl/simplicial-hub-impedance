"""
ROI — Physics-Informed Neuro-Physical Engine (runnable PyTorch skeleton)

This is a cleaned, executable version of the architecture from the prompt:
1. FreeEnergyFunctional: differentiable F = E - T*S.
2. HolographicCore: VAE-like encoder/decoder producing a psi field.
3. AutonomicScalerAgent: lightweight actor-critic scaler deciding merge/stay/split.
4. UniverseEngine: one optimization tick.

Run:
    python roi_universe_engine.py
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical


# ============================================================
# 1. PHYSICS LOSS: F = E - T*S
# ============================================================

class FreeEnergyFunctional(nn.Module):
    """
    Differentiable free-energy functional.

    Important implementation detail:
    Graph-only terms such as trace(A^3) do not train the neural core unless
    they depend on the model output. Therefore we weight graph energy by the
    learned local mass/probability field derived from psi.
    """

    def __init__(
        self,
        mu: float = 1.0,
        J: float = 0.65,
        T: float = 1.0,
        target_k: float = 8.0,
        smoothness_lambda: float = 0.05,
    ) -> None:
        super().__init__()
        # Kept as buffers: physical hyperparameters, not learned weights.
        self.register_buffer("mu", torch.tensor(float(mu)))
        self.register_buffer("J", torch.tensor(float(J)))
        self.register_buffer("T", torch.tensor(float(T)))
        self.register_buffer("target_k", torch.tensor(float(target_k)))
        self.smoothness_lambda = float(smoothness_lambda)

    @staticmethod
    def _symmetrize_adjacency(adj: torch.Tensor) -> torch.Tensor:
        # Undirected, no self-loops.
        adj = 0.5 * (adj + adj.transpose(-1, -2))
        adj = adj - torch.diag_embed(torch.diagonal(adj, dim1=-2, dim2=-1))
        return adj.clamp(0.0, 1.0)

    def forward(
        self,
        psi: torch.Tensor,
        adjacency_matrix: torch.Tensor,
        degrees: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Args:
            psi: learned probability/intensity field, shape [N, D], values in (0, 1).
            adjacency_matrix: graph adjacency, shape [N, N].
            degrees: node degrees, shape [N].

        Returns:
            F_total, E_total, S_total, components
        """
        eps = 1e-8
        n = psi.shape[0]
        A = self._symmetrize_adjacency(adjacency_matrix)

        # Local learned mass/intensity per cubic/node. Shape [N].
        mass = psi.mean(dim=-1).clamp(min=eps, max=1.0)

        # 1) Chemical-potential term: penalize deviation from target degree,
        # weighted by learned local mass so gradients flow into psi.
        E_mu = self.mu * torch.mean(mass * (degrees - self.target_k).pow(2))

        # 2) Curvature/triangle term. Weighted adjacency makes triangles depend on psi.
        # A_eff_ij = A_ij * sqrt(m_i m_j)
        sqrt_mass = torch.sqrt(mass + eps)
        A_eff = A * (sqrt_mass[:, None] * sqrt_mass[None, :])
        triangles = torch.trace(A_eff @ A_eff @ A_eff) / 6.0
        E_triangles = -self.J * triangles / max(float(n), 1.0)

        # 3) Smoothness/elasticity: connected cubics prefer compatible psi fields.
        edge_count = A.sum().clamp(min=1.0)
        diff = psi[:, None, :] - psi[None, :, :]
        E_smooth = self.smoothness_lambda * (A[..., None] * diff.pow(2)).sum() / edge_count

        E_total = E_mu + E_triangles + E_smooth

        # 4) Entropy. Normalize psi globally to a distribution.
        p = psi.clamp(min=eps)
        p = p / p.sum().clamp(min=eps)
        S_total = -(p * torch.log(p)).sum()

        F_total = E_total - self.T * S_total

        components = {
            "E_mu": E_mu.detach(),
            "E_triangles": E_triangles.detach(),
            "E_smooth": E_smooth.detach(),
            "triangles": triangles.detach(),
        }
        return F_total, E_total, S_total, components


# ============================================================
# 2. HOLOGRAPHIC VAE: Boundary encoder + Bulk decoder
# ============================================================

class HolographicCore(nn.Module):
    def __init__(self, n_cubics_dim: int = 128, n_clusters_dim: int = 16) -> None:
        super().__init__()

        # Right Core / Boundary / Encoder
        self.right_core_encoder = nn.Sequential(
            nn.Linear(n_cubics_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
        )
        self.fc_mu = nn.Linear(32, n_clusters_dim)
        self.fc_logvar = nn.Linear(32, n_clusters_dim)

        # Left Core / Bulk / Decoder
        self.left_core_decoder = nn.Sequential(
            nn.Linear(n_clusters_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, n_cubics_dim),
        )

    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.right_core_encoder(x)
        return self.fc_mu(h), self.fc_logvar(h).clamp(min=-8.0, max=8.0)

    @staticmethod
    def reparameterize(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode_logits(self, z: torch.Tensor) -> torch.Tensor:
        return self.left_core_decoder(z)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        logits = self.decode_logits(z)
        psi = torch.sigmoid(logits)
        return psi, logits, mu, logvar


# ============================================================
# 3. AUTONOMIC SCALER: lightweight actor-critic
# ============================================================

class AutonomicScalerAgent(nn.Module):
    """
    State:  [local |grad F|, cubic size, local density]
    Action: 0=merge, 1=stay, 2=split
    """

    def __init__(self, state_dim: int = 3, action_dim: int = 3) -> None:
        super().__init__()
        self.policy = nn.Sequential(
            nn.Linear(state_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, action_dim),
        )
        self.value = nn.Sequential(
            nn.Linear(state_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        logits = self.policy(state)
        value = self.value(state).squeeze(-1)
        return logits, value

    def act(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        logits, value = self.forward(state)
        dist = Categorical(logits=logits)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        entropy = dist.entropy()
        return action, log_prob, entropy, value


# ============================================================
# 4. UNIVERSE ENGINE
# ============================================================

@dataclass
class EngineConfig:
    n_cubics_dim: int = 128
    n_clusters_dim: int = 16
    lr_core: float = 1e-3
    lr_scaler: float = 3e-4
    beta_kl: float = 0.01
    recon_weight: float = 0.25
    scaler_weight: float = 0.05
    grad_split_threshold: float = 0.025
    grad_merge_threshold: float = 0.008


class UniverseEngine:
    def __init__(self, device: str = "cpu", config: EngineConfig | None = None) -> None:
        self.device = torch.device(device)
        self.config = config or EngineConfig()

        self.physics = FreeEnergyFunctional(mu=1.0, J=0.65, T=1.0).to(self.device)
        self.holo_core = HolographicCore(
            n_cubics_dim=self.config.n_cubics_dim,
            n_clusters_dim=self.config.n_clusters_dim,
        ).to(self.device)
        self.scaler = AutonomicScalerAgent().to(self.device)

        self.core_optimizer = torch.optim.Adam(self.holo_core.parameters(), lr=self.config.lr_core)
        self.scaler_optimizer = torch.optim.Adam(self.scaler.parameters(), lr=self.config.lr_scaler)

    def _scaler_reward(self, actions: torch.Tensor, states: torch.Tensor) -> torch.Tensor:
        """
        Simple online reward shaping.
        This is not full PPO; it is a compact actor-critic signal that can be
        replaced with PPO clipping once rollouts are stored.
        """
        grad_norm = states[:, 0]
        size = states[:, 1]
        density = states[:, 2]

        split_score = grad_norm - self.config.grad_split_threshold + 0.2 * density
        merge_score = self.config.grad_merge_threshold - grad_norm + 0.1 * (1.0 - density) - 0.05 * size
        stay_score = -torch.abs(grad_norm - 0.5 * (self.config.grad_split_threshold + self.config.grad_merge_threshold))

        rewards_by_action = torch.stack([merge_score, stay_score, split_score], dim=-1)
        reward = rewards_by_action.gather(1, actions[:, None]).squeeze(1)
        return reward

    def train_step(self, graph_batch: Dict[str, torch.Tensor]) -> Dict[str, object]:
        # Move tensors to device.
        cubics_state = graph_batch["cubics"].to(self.device).detach().requires_grad_(True)
        adj_matrix = graph_batch["adj"].to(self.device)
        degrees = graph_batch["degrees"].to(self.device)
        sizes = graph_batch["sizes"].to(self.device)
        density = graph_batch["density"].to(self.device)

        self.core_optimizer.zero_grad(set_to_none=True)
        self.scaler_optimizer.zero_grad(set_to_none=True)

        # Holographic flow.
        psi, logits, mu, logvar = self.holo_core(cubics_state)

        # Physics free energy.
        F_total, E_total, S_total, components = self.physics(psi, adj_matrix, degrees)

        # VAE terms.
        recon_loss = F.binary_cross_entropy_with_logits(logits, cubics_state, reduction="mean")
        kl_loss = -0.5 * torch.mean(1.0 + logvar - mu.pow(2) - logvar.exp())

        # Total core loss.
        core_loss = F_total + self.config.recon_weight * recon_loss + self.config.beta_kl * kl_loss

        # Local gradient for scaler. Retain graph because core_loss.backward follows.
        grad_F = torch.autograd.grad(
            core_loss,
            cubics_state,
            retain_graph=True,
            create_graph=False,
            allow_unused=False,
        )[0]
        local_grad_norm = grad_F.norm(dim=1).detach()

        # Normalize gradient feature for stable scaler states.
        grad_feature = local_grad_norm / (local_grad_norm.mean().clamp(min=1e-8))
        states = torch.column_stack([grad_feature, sizes, density]).detach()

        actions, log_probs, policy_entropy, values = self.scaler.act(states)
        rewards = self._scaler_reward(actions, states).detach()
        advantages = rewards - values.detach()

        actor_loss = -(log_probs * advantages).mean()
        critic_loss = F.mse_loss(values, rewards)
        entropy_bonus = policy_entropy.mean()
        scaler_loss = actor_loss + 0.5 * critic_loss - 0.01 * entropy_bonus

        # Optimize both parts. Keep scaler separate; its loss does not need to
        # change the holographic core directly.
        core_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.holo_core.parameters(), max_norm=1.0)
        self.core_optimizer.step()

        scaler_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.scaler.parameters(), max_norm=1.0)
        self.scaler_optimizer.step()

        with torch.no_grad():
            action_counts = torch.bincount(actions.cpu(), minlength=3).tolist()

        return {
            "Free_Energy": float(F_total.detach().cpu()),
            "Energy": float(E_total.detach().cpu()),
            "Entropy": float(S_total.detach().cpu()),
            "Core_Loss": float(core_loss.detach().cpu()),
            "Recon_Loss": float(recon_loss.detach().cpu()),
            "KL_Loss": float(kl_loss.detach().cpu()),
            "Scaler_Loss": float(scaler_loss.detach().cpu()),
            "Actions": actions.detach().cpu(),
            "Action_Counts": {"merge": action_counts[0], "stay": action_counts[1], "split": action_counts[2]},
            "Components": {k: float(v.cpu()) for k, v in components.items()},
        }


def make_dummy_graph(n: int = 64, d: int = 128, p_edge: float = 0.30, device: str = "cpu") -> Dict[str, torch.Tensor]:
    """Generate a symmetric random graph batch."""
    cubics = torch.rand(n, d, device=device)
    upper = torch.triu((torch.rand(n, n, device=device) < p_edge).float(), diagonal=1)
    adj = upper + upper.t()
    degrees = adj.sum(dim=1)
    sizes = torch.ones(n, device=device)
    density = torch.rand(n, device=device)
    return {
        "cubics": cubics,
        "adj": adj,
        "degrees": degrees,
        "sizes": sizes,
        "density": density,
    }


if __name__ == "__main__":
    torch.manual_seed(7)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    engine = UniverseEngine(device=device)

    print(f"Initializing the Universe on device={device}...")
    for epoch in range(101):
        dummy_graph = make_dummy_graph(n=64, d=128, p_edge=0.30, device=device)
        metrics = engine.train_step(dummy_graph)

        if epoch % 10 == 0:
            actions = metrics["Action_Counts"]
            print(
                f"Epoch {epoch:4d} | "
                f"F={metrics['Free_Energy']:.4f} | "
                f"E={metrics['Energy']:.4f} | "
                f"S={metrics['Entropy']:.4f} | "
                f"CoreLoss={metrics['Core_Loss']:.4f} | "
                f"A(M/S/P)={actions['merge']}/{actions['stay']}/{actions['split']}"
            )
