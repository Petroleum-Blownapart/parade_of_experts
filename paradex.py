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