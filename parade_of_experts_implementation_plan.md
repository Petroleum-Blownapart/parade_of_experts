# Parade of Experts (PoE): Architecture & Implementation Plan

## 1. Concept Overview

The **Parade of Experts (PoE)** is a dynamic, recurrent neural architecture designed for adaptive computation. Unlike standard feed-forward networks or static Mixture-of-Experts (MoE) layers that allocate a fixed computational budget to every input, PoE dynamically scales test-time compute based on input complexity.

### Core Components
1. **The "Potato" ($z_t$):** A shared, recurrent latent gestalt vector that acts as a working-memory buffer. It encodes the current representation of the input and is iteratively refined across passes.
2. **The Central Controller ($C$):** The coordinator network that inspects the potato at each recurrent pass $t$. It determines whether the current representation is sufficient to make a confident prediction or if it requires further processing by a specialist module.
3. **The Experts ($E_1, \dots, E_N$):** A collection of specialized neural submodules. When called upon by the controller, an expert receives the potato, modifies it to resolve specific ambiguities (e.g., denoising, spatial alignment, contextual integration), and passes it back to the controller.
4. **The Halting Gate:** A mechanism that terminates the loop as soon as the controller's confidence exceeds a target threshold $\theta$, or when a maximum pass limit $T_{\max}$ is reached.

---

## 2. Mathematical Formulation

### 2.1 Initial Encoding
Given a raw input feature vector $x \in \mathbb{R}^{d_{\text{in}}}$, an initial encoder projects the input into the latent working space of dimension $d$:
$$z_0 = \text{Encoder}(x) \in \mathbb{R}^d$$

### 2.2 Controller Evaluation
At each recurrent step $t \in \{0, 1, \dots, T_{\max} - 1\}$, the controller $C$ evaluates $z_t$ to produce three outputs:

1. **Task Inference Logits ($\hat{y}_t$):**
   $$\hat{y}_t = W_{\text{out}} z_t + b_{\text{out}}$$

2. **Confidence / Halting Score ($h_t \in [0, 1]$):**
   This can be parameterized via a dedicated learned halting head:
   $$h_t = \sigma(W_h z_t + b_h)$$
   Alternatively, it can be derived directly from the normalized predictive entropy of the output logits (for $K$-class classification):
   $$h_t = 1 - \frac{\mathcal{H}(\text{Softmax}(\hat{y}_t))}{\log(K)}$$

3. **Expert Routing Distribution ($p_t \in \Delta^N$):**
   A probability distribution over the $N$ available experts, scaled by a temperature parameter $\tau$:
   $$p_t = \text{Softmax}\left(\frac{W_r z_t + b_r}{\tau}\right)$$

### 2.3 Expert Modification (Gated Residual Update)
If the halting condition is not met ($h_t < \theta$), the potato is routed to a selected expert $E_i$. To prevent catastrophic forgetting of the initial input features over multiple recurrent passes, experts apply a **gated residual update** (analogous to a GRU cell) rather than overwriting $z_t$ outright:

$$\tilde{z}_t = E_i(z_t)$$
$$g_t = \sigma\left(W_g [z_t; \tilde{z}_t] + b_g\right)$$
$$z_{t+1} = \text{LayerNorm}\left(z_t + g_t \odot \tilde{z}_t\right)$$

where $[z_t; \tilde{z}_t]$ denotes vector concatenation and $\odot$ denotes element-wise multiplication.

---

## 3. Training Strategies & Architectural Challenges

Training a model with discrete routing decisions and variable-depth loops introduces non-differentiable branching. Below are the two primary strategies to train PoE end-to-end:

| Design Axis | Approach A: Differentiable Relaxation (Recommended for v1) | Approach B: Reinforcement Learning / POMDP |
| :--- | :--- | :--- |
| **Expert Selection** | **Straight-Through Gumbel-Softmax:** Sample a discrete one-hot expert on the forward pass, but backpropagate gradients through the continuous softmax probabilities. | **Policy Gradient (PPO / REINFORCE):** Treat the controller as an agent selecting discrete actions from $\{E_1, \dots, E_N, \text{Halt}\}$. |
| **Halting Mechanism** | **Adaptive Computation Time (ACT):** Unroll to $T_{\max}$ during training and compute the final prediction as a halting-weighted sum of intermediate logits. Use hard early-exit at inference. | **Step Penalty Reward:** Reward the agent $+1$ for accurate inference and penalize $-\lambda_{\text{step}}$ for every expert call made. |
| **Preventing Expert Collapse** | **Switch Load-Balancing Loss:** Add an auxiliary loss that penalizes variance in total routing probability allocated per expert across a batch. | **Entropy Regularization:** Add an entropy bonus $\beta \mathcal{H}(p_t)$ to the controller objective to encourage exploration across experts. |
| **Preventing Infinite Loops** | **Step & History Encoding:** Concatenate a learnable step embedding $e_t$ or a binary mask of previously visited experts to $z_t$ so the controller is aware of routing history. | **State Action Masking:** Mask out or penalize selecting the exact same expert on consecutive passes unless explicitly desired. |

---

## 4. Loss Function Engineering

For the differentiable relaxation approach, the total objective combines three distinct terms:

$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{task}} + \lambda_{\text{ponder}} \mathcal{L}_{\text{ponder}} + \lambda_{\text{balance}} \mathcal{L}_{\text{balance}}$$

1. **Task Loss ($\mathcal{L}_{\text{task}}$):**
   Standard task objective (e.g., Cross-Entropy or MSE) computed on the halting-weighted output:
   $$\hat{y}_{\text{final}} = \sum_{t=0}^{T_{\max}-1} w_t \hat{y}_t$$
   where $w_t = h_t \prod_{j=0}^{t-1} (1 - h_j)$ represents the probability of halting at step $t$.

2. **Ponder Cost ($\mathcal{L}_{\text{ponder}}$):**
   Penalizes the expected number of expert passes per sample, encouraging the controller to halt early on simple inputs:
   $$\mathcal{L}_{\text{ponder}} = \frac{1}{B} \sum_{b=1}^B \sum_{t=0}^{T_{\max}-1} \mathbb{I}(\text{active}_{b,t})$$

3. **Expert Load-Balancing Loss ($\mathcal{L}_{\text{balance}}$):**
   Prevents the controller from routing all potatoes to a single dominant expert. Let $\bar{p}_i$ be the average routing probability assigned to expert $i$ across the batch and steps:
   $$\mathcal{L}_{\text{balance}} = N \sum_{i=1}^N \bar{p}_i^2$$
   This term is minimized when $\bar{p}_i = \frac{1}{N}$ for all $i$.

---

## 5. Reference PyTorch Implementation

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class ExpertModule(nn.Module):
    """Specialized module that modifies the potato via a gated residual update."""
    def __init__(self, d_model: int, d_hidden: int):
        super().__init__()
        self.processor = nn.Sequential(
            nn.Linear(d_model, d_hidden),
            nn.GELU(),
            nn.Linear(d_hidden, d_model)
        )
        self.gate = nn.Linear(d_model * 2, d_model)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        delta = self.processor(z)
        g = torch.sigmoid(self.gate(torch.cat([z, delta], dim=-1)))
        return self.norm(z + g * delta)


class ParadeController(nn.Module):
    """Central controller that evaluates confidence and routes the potato."""
    def __init__(self, d_model: int, num_classes: int, num_experts: int, max_steps: int):
        super().__init__()
        self.step_embed = nn.Embedding(max_steps, d_model)
        self.classifier = nn.Linear(d_model, num_classes)
        self.router = nn.Linear(d_model, num_experts)
        self.halter = nn.Linear(d_model, 1)

    def forward(self, z: torch.Tensor, step: int, tau: float = 1.0, hard: bool = True):
        # Inject step awareness so the controller knows how long it has been pondering
        step_idx = torch.tensor(step, device=z.device)
        z_step = z + self.step_embed(step_idx)

        logits = self.classifier(z_step)
        halt_prob = torch.sigmoid(self.halter(z_step)).squeeze(-1)

        routing_logits = self.router(z_step)
        expert_weights = F.gumbel_softmax(routing_logits, tau=tau, hard=hard, dim=-1)
        routing_probs = F.softmax(routing_logits, dim=-1)

        return logits, halt_prob, expert_weights, routing_probs


class ParadeOfExperts(nn.Module):
    def __init__(
        self,
        input_dim: int,
        d_model: int,
        num_classes: int,
        num_experts: int = 4,
        max_steps: int = 5
    ):
        super().__init__()
        self.max_steps = max_steps
        self.num_experts = num_experts

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, d_model),
            nn.LayerNorm(d_model),
            nn.GELU()
        )
        self.controller = ParadeController(d_model, num_classes, num_experts, max_steps)
        self.experts = nn.ModuleList([
            ExpertModule(d_model, d_model * 2) for _ in range(num_experts)
        ])

    def forward(self, x: torch.Tensor, threshold: float = 0.85, tau: float = 1.0):
        B = x.size(0)
        z = self.encoder(x)  # Initial potato: (B, d_model)

        remainders = torch.ones(B, device=x.device)
        accumulated_logits = torch.zeros(B, self.controller.classifier.out_features, device=x.device)
        ponder_steps = torch.zeros(B, device=x.device)
        routing_history = []
        active_mask = torch.ones(B, dtype=torch.bool, device=x.device)

        for step in range(self.max_steps):
            logits, halt_prob, expert_weights, routing_probs = self.controller(
                z, step=step, tau=tau, hard=True
            )
            routing_history.append(routing_probs)

            # Determine halting weight for the current pass
            if step == self.max_steps - 1:
                step_weight = remainders
            else:
                step_weight = halt_prob * remainders

            accumulated_logits = accumulated_logits + step_weight.unsqueeze(-1) * logits
            remainders = remainders - step_weight
            ponder_steps = ponder_steps + active_mask.float()

            # Update active mask for samples below confidence threshold
            active_mask = active_mask & (halt_prob < threshold)
            if not self.training and not active_mask.any():
                break

            # Execute experts and select via Gumbel one-hot weights
            expert_outputs = torch.stack([exp(z) for exp in self.experts], dim=1)
            z_next = torch.sum(expert_weights.unsqueeze(-1) * expert_outputs, dim=1)

            # Freeze potato state for already-halted samples
            z = torch.where(active_mask.unsqueeze(-1), z_next, z)

        routing_tensor = torch.stack(routing_history, dim=1)  # (B, steps, num_experts)
        return accumulated_logits, ponder_steps.mean(), routing_tensor
```

---

## 6. Phased Implementation Plan

### Phase 1: Synthetic Benchmark Creation
Standard datasets (like clean MNIST) have uniform difficulty, causing adaptive models to collapse to a static 1-step policy.
* **Action:** Construct a **Multi-Condition Dataset** (e.g., Modified MNIST or CIFAR-10) partitioned into four difficulty tiers:
  1. *Tier 0 (Clean):* Unmodified inputs (Target: 0 expert passes, immediate inference).
  2. *Tier 1 (Noisy):* Additive Gaussian noise (Target: 1 pass via a Denoising Expert).
  3. *Tier 2 (Transformed):* Random rotations or spatial shifts (Target: 1 pass via a Spatial Expert).
  4. *Tier 3 (Compound):* Both noisy and rotated/occluded (Target: 2+ passes chaining multiple experts).

### Phase 2: Warm-Up & Expert Initialization
If early halting is active from Epoch 1, the controller will prematurely halt or route randomly before experts learn useful transformations.
* **Action:** Train for the first $N_{\text{warmup}}$ epochs with:
  * Halting threshold fixed to $\theta = 1.0$ (forces full $T_{\max}$ unrolling).
  * Ponder penalty $\lambda_{\text{ponder}} = 0.0$.
  * High Gumbel temperature $\tau = 2.0$ and strong load-balancing weight $\lambda_{\text{balance}} = 0.05$ so all experts receive dense gradients.

### Phase 3: Annealing & Dynamic Halting
Transition the network from a soft, fixed-depth ensemble into a discrete, variable-depth state machine.
* **Action:** Over the next $M$ epochs:
  * Linearly anneal Gumbel temperature $\tau$ from $2.0 \rightarrow 0.5$.
  * Ramp up $\lambda_{\text{ponder}}$ from $0.0 \rightarrow 0.02$.
  * Enable inference thresholding at $\theta = 0.85$.
* **Validation:** Verify that Tier 0 samples exit at $t = 0$ while Tier 3 samples take $t \ge 2$ passes.

### Phase 4: Architectural Heterogeneity
Once routing succeeds with homogeneous MLP experts, introduce distinct inductive biases into the expert pool while preserving the $\mathbb{R}^d \rightarrow \mathbb{R}^d$ interface:
* **Expert 1 (Local/Spatial):** 1D/2D Depthwise Separable Convolution block.
* **Expert 2 (Global Context):** Multi-head Self-Attention block (if the potato is a sequence of tokens $Z_t \in \mathbb{R}^{L \times d}$).
* **Expert 3 (Denoising/Compression):** Low-rank bottleneck autoencoder block ($d \rightarrow d/4 \rightarrow d$).
* **Expert 4 (Non-linear Disentanglement):** Wide sparse MLP with ReLU/SwiGLU activations ($d \rightarrow 4d \rightarrow d$).

---

## 7. Diagnostic & Evaluation Metrics

To confirm the model is genuinely leveraging the "parade" mechanism rather than gaming the loss function, track these metrics during training:

1. **Expert Transition Matrix ($M \in \mathbb{R}^{(N+1) \times (N+1)}$):**
   Track empirical probabilities $P(\text{State}_{t+1} = j \mid \text{State}_t = i)$, including the `Halt` state. Non-uniform off-diagonal entries prove the controller is discovering multi-step compositional pipelines (e.g., *Expert 2 $\rightarrow$ Expert 1 $\rightarrow$ Halt*).
2. **Confidence Gain per Expert ($\Delta h_i$):**
   Compute the average change in controller confidence $h_{t+1} - h_t$ immediately following an invocation of Expert $i$. Positive $\Delta h_i$ confirms the expert is resolving ambiguity in the potato.
3. **Latent Trajectory Cosine Similarity:**
   Monitor $\cos(z_t, z_{t+1})$ across passes.
   * If $\cos(z_t, z_{t+1}) \approx 1.0$, the residual gate $g_t$ is saturating near $0$ and experts are performing no-ops.
   * If $\cos(z_t, z_{t+1}) \approx 0.0$, the expert is completely overwriting the potato and destroying prior context.
   * Healthy refinement typically falls in the $[0.75, 0.95]$ range.
4. **Compute-vs-Accuracy Pareto Curve:**
   Sweep the inference threshold $\theta \in [0.50, 0.99]$ and plot Task Accuracy against Average Expert Passes per Sample, comparing against a static-depth baseline of equivalent parameter count.