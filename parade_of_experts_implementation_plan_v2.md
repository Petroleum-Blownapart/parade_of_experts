# Parade of Experts (PoE): CIFAR-10 Architecture & Winner-Take-All Implementation Plan

## 1. Concept Overview

The **Parade of Experts (PoE)** is a dynamic, recurrent neural architecture designed for adaptive computation on **CIFAR-10**. Unlike standard feed-forward networks or static Mixture-of-Experts (MoE) layers that allocate a fixed computational budget to every image, PoE dynamically scales test-time compute based on input complexity.

### Core Components
1. **The "Potato" ($z_t$):** A shared, recurrent latent gestalt vector ($z_t \in \mathbb{R}^d$) that acts as a working-memory buffer. An initial convolutional stem encodes the $3 \times 32 \times 32$ CIFAR-10 image into $z_0$, which is iteratively refined across passes.
2. **The Central Controller ($C$):** The coordinator network that inspects the potato at each recurrent pass $t$. It outputs class predictions, a halting confidence score, and expert routing logits.
3. **The Experts ($E_1, \dots, E_N$):** A collection of specialized neural submodules. When called upon by the controller, an expert receives the potato, modifies it via a gated residual update to resolve specific ambiguities (e.g., denoising, spatial alignment, feature disentanglement), and passes it back to the controller.
4. **Oracle-Guided Winner-Take-All (WTA) Training:** During training, the potato is passed to *all* experts in parallel at each step, and each resulting candidate potato is evaluated by the classifier against the ground-truth label. The expert that achieves the lowest cross-entropy loss for a given sample is designated the "winner." Its index serves as the ground-truth target to train the router, and **gradients are backpropagated exclusively through the winning expert's branch**.

---

## 2. Mathematical Formulation

### 2.1 Initial CIFAR-10 Encoding (Convolutional Stem)
Given an input image $x \in \mathbb{R}^{3 \times 32 \times 32}$, a lightweight convolutional encoder projects the spatial features into the initial latent potato vector $z_0$ of dimension $d$:
$$z_0 = \text{ConvEncoder}(x) \in \mathbb{R}^d$$

### 2.2 Controller Evaluation
At each recurrent step $t \in \{0, 1, \dots, T_{\max}\}$, the controller adds a learnable step embedding $e_t \in \mathbb{R}^d$ to $z_t$ ($\tilde{z}_t = z_t + e_t$) and computes:

1. **Task Inference Logits ($\hat{y}_t$):**
   $$\hat{y}_t = W_{\text{out}} \tilde{z}_t + b_{\text{out}} \in \mathbb{R}^{10}$$

2. **Halting Confidence Score ($h_t \in [0, 1]$):**
   $$h_t = \sigma(W_h \tilde{z}_t + b_h)$$

3. **Expert Router Logits ($r_t \in \mathbb{R}^N$):**
   $$r_t = W_r \tilde{z}_t + b_r$$

### 2.3 Expert Modification (Gated Residual Update)
Each expert $E_i$ ($i \in \{1, \dots, N\}$) transforms the potato using a gated residual update so prior context is preserved across recurrent passes:
$$\Delta z_t^{(i)} = E_i(z_t)$$
$$g_t^{(i)} = \sigma\left(W_g^{(i)} [z_t; \Delta z_t^{(i)}] + b_g^{(i)}\right)$$
$$z_{t+1}^{(i)} = \text{LayerNorm}\left(z_t + g_t^{(i)} \odot \Delta z_t^{(i)}\right)$$

---

## 3. Oracle-Guided Winner-Take-All (WTA) Training Mechanism

Instead of relying on noisy Gumbel-Softmax relaxations or reinforcement learning, training uses an exact **lookahead oracle** at every recurrent step $t$:

### Step 1: Parallel Expert Evaluation
For each sample $b$ in a mini-batch of size $B$, pass the current potato $z_{t,b}$ through all $N$ experts in parallel to obtain $N$ candidate next-step potatoes:
$$Z_{t+1, b} = \left\{ z_{t+1, b}^{(1)}, z_{t+1, b}^{(2)}, \dots, z_{t+1, b}^{(N)} \right\}$$

Pass each candidate potato through the controller's classifier (using step embedding $e_{t+1}$) to produce candidate class logits:
$$\hat{y}_{t+1, b}^{(i)} = W_{\text{out}} \left(z_{t+1, b}^{(i)} + e_{t+1}\right) + b_{\text{out}} \quad \forall i \in \{1, \dots, N\}$$

### Step 2: Finding the Winning Expert ($i^*_b$)
Evaluate the sample-wise Cross-Entropy (CE) loss for each candidate expert against the true CIFAR-10 label $y_b$:
$$\ell_{t+1, b}^{(i)} = \mathcal{L}_{\text{CE}}\left(\hat{y}_{t+1, b}^{(i)}, y_b\right)$$

The winning expert $i^*_b$ for instance $b$ at step $t$ is the one that minimizes this loss:
$$i^*_b = \arg\min_{i \in \{1, \dots, N\}} \ell_{t+1, b}^{(i)}$$

*(Note: To prevent initial "rich-get-richer" expert collapse in Epoch 1 where one randomly lucky expert wins every sample and starves the others, we include a small utilization penalty $\lambda_{\text{util}} \cdot \text{freq}(i)$ inside the $\arg\min$ during warm-up).*

### Step 3: Exclusive Backpropagation Through the Winning Route
In PyTorch, indexing the stacked expert outputs with the winning expert indices:
$$z_{t+1, b} = z_{t+1, b}^{(i^*_b)}$$
naturally constructs a computational graph where **gradients flow only through the winning expert $E_{i^*_b}$** for sample $b$. All other $N - 1$ experts receive zero gradient for sample $b$ on step $t$.

### Step 4: Supervised Router & Halting Training
* **Router Loss ($\mathcal{L}_{\text{router}}$):** Because $i^*_b$ identifies the empirically optimal expert for $z_{t,b}$, we train the controller's router head directly via standard Cross-Entropy loss using $i^*_b$ as the target label:
  $$\mathcal{L}_{\text{router}}^{(t)} = \frac{1}{B} \sum_{b=1}^B \mathcal{L}_{\text{CE}}\left(r_{t, b}, i^*_b\right)$$
* **Halting Loss ($\mathcal{L}_{\text{halt}}$):** We define a binary oracle halting target $h^*_{t,b} \in \{0, 1\}$. The controller should halt ($h^*_{t,b} = 1$) if its current prediction is already correct ($\arg\max \hat{y}_{t,b} = y_b$) AND its confidence is high (or if the best expert offers negligible loss improvement $\ell_{t,b} - \ell_{t+1,b}^{(i^*_b)} < \delta$). We train the halting head via Binary Cross-Entropy:
  $$\mathcal{L}_{\text{halt}}^{(t)} = \frac{1}{B} \sum_{b=1}^B \mathcal{L}_{\text{BCE}}\left(h_{t, b}, h^*_{t, b}\right)$$

### Step 5: Inference Execution (Single-Expert Routing)
At test time (`model.eval()`), ground-truth labels are unavailable, so the lookahead oracle is bypassed:
1. The controller evaluates $z_t$ to get $\hat{y}_t$, $h_t$, and $r_t$.
2. If $h_t \ge \theta$, inference halts immediately and outputs $\hat{y}_t$.
3. Otherwise, the controller picks the single highest-scoring expert $\hat{i} = \arg\max_i r_{t,i}$, passes $z_t$ **only** to $E_{\hat{i}}$, and repeats until $h_t \ge \theta$ or $t = T_{\max}$.

---

## 4. Total Training Objective

At each step $t$, the loss accumulated over active samples is:
$$\mathcal{L}_{\text{total}} = \sum_{t=0}^{T_{\max}} \left( \mathcal{L}_{\text{task}}^{(t)} + \lambda_{\text{router}} \mathcal{L}_{\text{router}}^{(t)} + \lambda_{\text{halt}} \mathcal{L}_{\text{halt}}^{(t)} \right)$$

where:
* $\mathcal{L}_{\text{task}}^{(0)} = \mathcal{L}_{\text{CE}}(\hat{y}_0, y)$ trains the initial encoder and classifier.
* $\mathcal{L}_{\text{task}}^{(t+1)} = \frac{1}{B} \sum_{b=1}^B \ell_{t+1, b}^{(i^*_b)}$ trains **only the winning expert $E_{i^*_b}$** at step $t$, plus the shared classifier and upstream trajectory.

---

## 5. Phased CIFAR-10 Experimental Roadmap

### Phase 1: Multi-Condition CIFAR-10 Pipeline
To give the experts clear incentives to specialize in different transformations, each CIFAR-10 training image is dynamically assigned one of four conditions:
1. **Clean ($25\%$):** Standard normalized CIFAR-10 image (should halt at $t=0$ or $t=1$).
2. **Gaussian Noise ($25\%$):** Additive noise $\epsilon \sim \mathcal{N}(0, \sigma^2)$ requiring feature denoising.
3. **Cutout / Occlusion ($25\%$):** Random rectangular patch zeroed out, requiring contextual completion.
4. **Compound Corruption ($25\%$):** Both noise and occlusion/blur, requiring multiple expert passes.

### Phase 2: Anti-Collapse Warm-Up (Epochs 1–5)
In pure Winner-Take-All training, if Expert 1 starts with slightly better random weights, it can win $100\%$ of samples in batch 1, receive $100\%$ of gradients, and permanently starve Experts $2\dots N$.
* **Solution:** Maintain an exponential moving average (EMA) of each expert's win frequency $\bar{f}_i \in [0, 1]$. During selection, add a small load-balancing margin inside `torch.no_grad()`:
  $$i^*_b = \arg\min_i \left( \ell_{t+1, b}^{(i)} + \lambda_{\text{margin}} \left(\bar{f}_i - \frac{1}{N}\right) \right)$$
  This acts as a gentle handicap on over-worked experts during early epochs, ensuring all $N$ experts win niches and specialize.

### Phase 3: Full Winner-Take-All Specialization (Epochs 6–50)
* Decay $\lambda_{\text{margin}} \rightarrow 0$ so selection is governed purely by raw Cross-Entropy loss $\arg\min_i \ell_{t+1,b}^{(i)}$.
* Track router Top-1 accuracy against the oracle choice $i^*_b$. As experts specialize into distinct domains, the router's accuracy in predicting the winning expert from $z_t$ will climb steadily.

### Phase 4: Inference Threshold Sweep
* Evaluate the trained model on the CIFAR-10 test set across halting thresholds $\theta \in \{0.5, 0.7, 0.85, 0.95\}$.
* Log the **Accuracy vs. Average Expert Passes** curve and inspect the **Expert Transition Matrix** across the 4 corruption tiers.