Ross Berger  
ECE-110  
UCSC Fall Quarter 2026

# Parade of Experts Project Proposal

## Outline

My intention is to develop an experimental technique for deep learning that combines concepts from RNN's 
and "Mixture of Experts" models.

The **Parade of Experts (PoE)** combines ideas from **Recurrent Neural Networks (RNNs)** and **Mixture-of-Experts (MoE)** models to achieve adaptive, multi-pass computation:

1. **The "Potato" ($z_t \in \mathbb{R}^{d}$):** Instead of passing activations through a fixed stack of layers, an input image is encoded into a 1D latent vector $z_0$ called the *potato* ($d = 128$). This vector serves as a recurrent working-memory state that gets passed from expert to expert and iteratively refined.
2. **The Central Controller:** At each pass $t \in \{0, 1, \dots, T_{\max}\}$, the controller inspects the current potato $z_t$ (injected with a step embedding $e_t$) and makes three predictions:
   - **Class Logits ($\hat{y}_t$):** What class does the potato currently look like?
   - **Halting Logit ($h_t$):** Is the representation already confident and refined enough to stop, or does it need another expert pass?
   - **Routing Logits ($r_t$):** If we don't stop, which of the 8 specialized experts should refine the potato next?
3. **The Expert Pool ($E_1, \dots, E_N$):** A set of 8 independent neural modules. When an expert receives $z_t$, it computes a proposed modification $\tilde{z}_t$ and applies a **GRU-style gated residual update** so it refines the potato without erasing prior context.

## Motivation

## Background

## Implementation plan

![Overview Diagram](PoEDiagram.png)

## Evaluation plan

## Deliverable

## Referrences