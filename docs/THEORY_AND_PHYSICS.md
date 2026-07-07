# Theoretical Framework & Physical Mechanics of Hub Impedance

**Author:** Michał Ślusarczyk  
**Date:** July 2026  
**Keywords:** Simplicial Quantum Gravity, Causal Dynamical Triangulations (CDT), Regge Calculus, Laplace-Beltrami Operator, Spectral Dimension, Time Dilation.

---

## 1. The Challenge of Topological Anomalies in Discrete Spacetime

In non-perturbative, background-independent approaches to quantum gravity—most notably **Causal Dynamical Triangulations (CDT)**, **Regge Calculus**, and **Relational Observables in Simplicial Gravity (ROI)**—continuous Riemannian spacetime manifolds are approximated by triangulations constructed from discrete building blocks (e.g., 4D pentatopes or 3D spatial tetrahedra). The quantum Feynman path integral is formulated as a statistical sum over all valid geometric and topological triangulations $\mathcal{T}$:

$$Z = \sum_{\mathcal{T}} \frac{1}{C(\mathcal{T})} \exp\left( -S_{\text{Regge}}[\mathcal{T}] \right)$$

Within this non-perturbative phase space, discrete models frequently exhibit phase transitions into pathological regimes such as the **branched polymer phase** or the **soft hub phase**. In these configurations, the geometry minimizes action by concentrating a massive proportion of adjacent simplices around a small subset of vertices ($d_u \gg \langle d \rangle$).

### The Unphysical Shortcut Problem
Historically, probing the geometry of these triangulations relied on unweighted random walks or heat diffusion on the 1-skeleton dual graph ($q = 0.00$). In an unweighted walk, transition probabilities depend solely on the degree of the originating node ($P(u \to v) = 1 / d_u$). Because a topological hub $h$ possesses thousands of incident edges, diffusion probes across the entire universe are continuously drawn into the hub. Once inside, a random or quantum walker can transition to virtually any distant patch of the triangulation in a single step.

Consequently, topological hubs act as **unphysical metric shortcuts (wormholes)** that artificially collapse the network diameter. When computing the scale-dependent **spectral dimension** $D_s(\tau)$ via the return probability $P_r(\tau) \sim \tau^{-D_s / 2}$, these shortcuts cause an artificial dimensionality collapse ($D_s \to 0$ or fractal polymer values ~1–2), obscuring the genuine 4D macroscopic spacetime geometry.

---

## 2. The Concrete Solution: Physical Edge Impedance ($q = 0.25$)

Rather than removing hubs from the Monte Carlo algorithm—an ad-hoc procedure that violates statistical ergodicity, introduces artificial boundary potentials, and breaks diffeomorphism invariance—this repository implements a non-invasive physical solution: **decoupling topological coordination from the metric transport operator**.

We endow each edge $(u,v) \in E$ in the 1-skeleton with a physical conductance weight (inverse impedance) scaled by the degrees of the connected vertices:

$$w_{uv} = \left( \max(1, d_u) \cdot \max(1, d_v) \right)^{-q}$$

In continuous-time quantum walks (CTQW) executed on quantum hardware, the unitary evolution is governed by the Hermitian Hamiltonian:

$$H_{uv} = -w_{uv} = -\left( d_u \cdot d_v \right)^{-q}$$

### Criticality of the $q = 0.25$ Exponent
* **$q = 0.00$ (Unweighted Walk):** Edges have uniform conductance ($w_{uv} = 1.0$). Hubs dominate information transport and induce constructive resonance shortcuts.
* **$q = 0.50$ (Symmetric Normalization):** Edges scale as $(d_u d_v)^{-0.50}$. While this fully eliminates degree bias, it introduces excessive damping across smooth bulk geometry patches.
* **$q = 0.25$ (Optimal Fractional Impedance):** Edges scale as $w_{uv} = (d_u d_v)^{-0.25}$. For a jump between a regular bulk node ($d_{\text{bulk}} \approx \langle d \rangle$) and a massive hub ($d_{\text{hub}} \gg \langle d \rangle$), the coupling coefficient is attenuated by $\gamma \approx (d_{\text{hub}} / \langle d \rangle)^{-0.25}$.

This fractional impedance creates an effective metric barrier around coordination anomalies while preserving regular diffusion across smooth bulk patches ($d_u \approx d_v$).

---

## 3. Physical Interpretation: Regge Calculus & General Relativity

### 3.1. Discrete Laplace-Beltrami Approximation
In continuous General Relativity, scalar field propagation and heat diffusion are governed by the invariant Laplace-Beltrami operator:

$$\Delta_{\mathcal{M}} \phi = \frac{1}{\sqrt{|g|}} \partial_\mu \left( \sqrt{|g|} g^{\mu\nu} \partial_\nu \phi \right)$$

The metric determinant $\sqrt{|g|}$ defines the local volume density. In unweighted graph diffusion ($q=0$), every edge is treated as having identical unit length and capacity, implying that a vertex where 10,000 simplices converge has zero volume of its own—a severe discretization error.

The fractional weight $w_{uv} = (d_u d_v)^{-0.25}$ acts as a **discrete approximation of the inverse metric determinant $\frac{1}{\sqrt{|g|}}$**, ensuring that effective metric distances scale proportionally to the actual volume of packed simplicial matter.

### 3.2. Gravitational Time Dilation & Metric Resistance
In simplicial Regge calculus, scalar curvature $R(u)$ is concentrated on hinges and is directly proportional to the angular deficit $\delta_u = 2\pi - \sum_i \theta_i$. A topological hub represents an extreme concentration of curvature and local simplicial density—analogous to a microscopic gravitational well or spacetime singularity.

In General Relativity, time slows down in deep gravitational wells (gravitational time dilation). A probe particle cannot traverse an area of extreme curvature instantaneously. The fractional impedance $d_{\text{hub}}^{-0.25}$ models exactly this physical effect: it slows down wavefunction propagation through topological anomalies, restoring **metric locality and relativistic causality** without censoring quantum topological fluctuations.
