import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as T


# =====================================================================
# 1. Standard CIFAR-10 DataLoaders
# =====================================================================
def get_cifar10_dataloaders(root: str = "./data", batch_size: int = 128, num_workers: int = 2):
    """
    Creates standard CIFAR-10 training and test DataLoaders with
    canonical normalization and basic training augmentations.
    """
    normalize = T.Normalize(
        mean=(0.4914, 0.4822, 0.4465),
        std=(0.2470, 0.2435, 0.2616)
    )

    train_transform = T.Compose([
        T.RandomCrop(32, padding=4),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        normalize,
    ])

    test_transform = T.Compose([
        T.ToTensor(),
        normalize,
    ])

    train_dataset = torchvision.datasets.CIFAR10(
        root=root, train=True, download=True, transform=train_transform
    )
    test_dataset = torchvision.datasets.CIFAR10(
        root=root, train=False, download=True, transform=test_transform
    )

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True
    )

    return train_loader, test_loader


# =====================================================================
# 2. Parade of Experts Architecture (CIFAR-10)
# =====================================================================
class ConvPotatoEncoder(nn.Module):
    """Encodes a 3x32x32 CIFAR-10 image into the initial potato vector z_0."""
    def __init__(self, d_model: int = 128):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.GELU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, bias=False),  # 16x16
            nn.BatchNorm2d(128),
            nn.GELU(),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1, bias=False), # 8x8
            nn.BatchNorm2d(256),
            nn.GELU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(256, d_model),
            nn.LayerNorm(d_model)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.stem(x)


class Expert(nn.Module):
    """
    A single specialized expert that processes the potato and applies
    a gated residual update to preserve core state information.
    """
    def __init__(self, d_model: int = 128, d_hidden: int = 512, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_hidden),
            nn.LayerNorm(d_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_hidden, d_model)
        )
        self.gate = nn.Linear(d_model * 2, d_model)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        delta = self.net(z)
        g = torch.sigmoid(self.gate(torch.cat([z, delta], dim=-1)))
        return self.norm(z + g * delta)


class ParadeController(nn.Module):
    """
    Central controller that inspects the potato at step t and produces:
      1. Class inference logits
      2. Halting confidence probability
      3. Expert routing logits
    """
    def __init__(self, d_model: int = 128, num_classes: int = 10, num_experts: int = 8, max_steps: int = 4):
        super().__init__()
        self.step_embed = nn.Embedding(max_steps + 1, d_model)
        self.classifier = nn.Linear(d_model, num_classes)
        self.router = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, num_experts)
        )
        self.halter = nn.Sequential(
            nn.Linear(d_model, d_model // 4),
            nn.GELU(),
            nn.Linear(d_model // 4, 1)
        )

    def classify_only(self, z: torch.Tensor, step: int) -> torch.Tensor:
        """Evaluates only the classifier head on candidate potatoes."""
        step_idx = torch.tensor(step, device=z.device)
        z_step = z + self.step_embed(step_idx)
        return self.classifier(z_step)

    def forward(self, z: torch.Tensor, step: int):
        step_idx = torch.tensor(step, device=z.device)
        z_step = z + self.step_embed(step_idx)

        logits = self.classifier(z_step)
        halt_logit = self.halter(z_step).squeeze(-1)
        router_logits = self.router(z_step)
        return logits, halt_logit, router_logits


class ParadeOfExpertsCIFAR(nn.Module):
    def __init__(
        self,
        d_model: int = 128,
        num_classes: int = 10,
        num_experts: int = 8,
        max_steps: int = 3
    ):
        super().__init__()
        self.d_model = d_model
        self.num_classes = num_classes
        self.num_experts = num_experts
        self.max_steps = max_steps

        self.encoder = ConvPotatoEncoder(d_model)
        self.controller = ParadeController(d_model, num_classes, num_experts, max_steps)
        self.experts = nn.ModuleList([
            Expert(d_model, d_model * 2) for _ in range(num_experts)
        ])

        # Track EMA of expert win frequencies to prevent Epoch-1 collapse
        self.register_buffer("expert_freq_ema", torch.full((num_experts,), 1.0 / num_experts))

    def forward_train_wta(
        self,
        x: torch.Tensor,
        labels: torch.Tensor,
        margin_weight: float = 0.1,
        improvement_delta: float = 0.05
    ):
        """
        Oracle-Guided Winner-Take-All (WTA) Training Pass:
          - Evaluates ALL experts at each step.
          - Finds which expert yields the lowest Cross-Entropy loss per sample.
          - Trains the router to predict that winning expert.
          - Backpropagates task gradients ONLY through the winning expert's route.
        """
        B = x.size(0)
        batch_indices = torch.arange(B, device=x.device)

        # 1. Encode initial potato z_0
        z = self.encoder(x)

        # Initial controller evaluation at step 0
        logits_0, halt_logit_0, router_logits_0 = self.controller(z, step=0)
        loss_t = F.cross_entropy(logits_0, labels, reduction="none")  # (B,)

        total_task_loss = loss_t.mean()
        total_router_loss = torch.tensor(0.0, device=x.device)
        total_halt_loss = torch.tensor(0.0, device=x.device)

        router_correct = 0
        total_routed = 0
        expert_win_counts = torch.zeros(self.num_experts, device=x.device)

        curr_logits = logits_0
        curr_halt_logit = halt_logit_0
        curr_router_logits = router_logits_0
        curr_loss = loss_t

        for step in range(self.max_steps):
            # 2. Pass current potato z to ALL experts in parallel: (B, N, d_model)
            candidate_potatoes = torch.stack([exp(z) for exp in self.experts], dim=1)

            # 3. Classify all candidate potatoes at step + 1: (B, N, num_classes)
            candidate_logits = self.controller.classify_only(candidate_potatoes, step=step + 1)

            # 4. Evaluate Cross-Entropy loss for every expert per sample (without grad for selection)
            with torch.no_grad():
                flat_logits = candidate_logits.reshape(B * self.num_experts, self.num_classes)
                flat_labels = labels.unsqueeze(1).expand(B, self.num_experts).reshape(-1)
                candidate_losses = F.cross_entropy(flat_logits, flat_labels, reduction="none")
                candidate_losses = candidate_losses.view(B, self.num_experts)  # (B, N)

                # Add slight load-balancing handicap during early warm-up to prevent dead experts
                freq_bias = margin_weight * (self.expert_freq_ema - (1.0 / self.num_experts))
                selection_scores = candidate_losses + freq_bias.unsqueeze(0)

                # Best expert index per instance: (B,)
                best_expert_idx = torch.argmin(selection_scores, dim=1)
                min_next_loss = candidate_losses[batch_indices, best_expert_idx]

                # Update EMA of winning expert frequencies
                batch_counts = torch.bincount(best_expert_idx, minlength=self.num_experts).float()
                expert_win_counts += batch_counts
                self.expert_freq_ema.mul_(0.95).add_(0.05 * (batch_counts / B))

                # Oracle halting target: halt (1.0) if already correct AND expert improves loss by < delta
                already_correct = (curr_logits.argmax(dim=-1) == labels)
                minimal_gain = (curr_loss - min_next_loss) < improvement_delta
                oracle_halt_target = (already_correct & minimal_gain).float()

            # 5. Train the Router using the winning expert as ground-truth target
            step_router_loss = F.cross_entropy(curr_router_logits, best_expert_idx)
            total_router_loss = total_router_loss + step_router_loss

            # Track Router accuracy against the Oracle
            router_preds = curr_router_logits.argmax(dim=-1)
            router_correct += (router_preds == best_expert_idx).sum().item()
            total_routed += B

            # 6. Train the Halting Gate against the Oracle Halting Target
            step_halt_loss = F.binary_cross_entropy_with_logits(curr_halt_logit, oracle_halt_target)
            total_halt_loss = total_halt_loss + step_halt_loss

            # 7. Select ONLY the winning potato and winning logits for the computation graph!
            # Indexing with [batch_indices, best_expert_idx] ensures backprop ONLY updates
            # the winning expert for each instance; non-winning experts get zero gradient.
            z = candidate_potatoes[batch_indices, best_expert_idx]  # (B, d_model)
            winning_logits = candidate_logits[batch_indices, best_expert_idx]  # (B, num_classes)

            # Compute differentiable task loss along the winning route
            winning_loss = F.cross_entropy(winning_logits, labels, reduction="none")
            total_task_loss = total_task_loss + winning_loss.mean()

            # Prepare controller outputs for the next step
            if step < self.max_steps - 1:
                curr_logits, curr_halt_logit, curr_router_logits = self.controller(z, step=step + 1)
                curr_loss = winning_loss
            else:
                curr_logits = winning_logits

        # Average losses across steps
        total_task_loss = total_task_loss / (self.max_steps + 1)
        total_router_loss = total_router_loss / self.max_steps
        total_halt_loss = total_halt_loss / self.max_steps

        metrics = {
            "task_loss": total_task_loss.item(),
            "router_loss": total_router_loss.item(),
            "halt_loss": total_halt_loss.item(),
            "router_oracle_acc": router_correct / max(1, total_routed),
            "final_acc": (curr_logits.argmax(dim=-1) == labels).float().mean().item(),
            "expert_distribution": (expert_win_counts / max(1.0, expert_win_counts.sum())).cpu().tolist()
        }

        return total_task_loss, total_router_loss, total_halt_loss, metrics

    @torch.no_grad()
    def forward_inference(self, x: torch.Tensor, threshold: float = 0.90):
        """
        Test-Time Inference:
          - No ground-truth labels or all-expert lookahead.
          - Controller halts early when confidence >= threshold.
          - Otherwise routes each sample ONLY to the top-1 predicted expert.
        """
        B = x.size(0)
        z = self.encoder(x)

        final_logits = torch.zeros(B, self.num_classes, device=x.device)
        active_mask = torch.ones(B, dtype=torch.bool, device=x.device)
        steps_taken = torch.zeros(B, device=x.device)

        for step in range(self.max_steps + 1):
            logits, halt_logit, router_logits = self.controller(z, step=step)
            halt_prob = torch.sigmoid(halt_logit)

            # Entropy-assisted confidence can also be blended with learned halt_prob
            probs = F.softmax(logits, dim=-1)
            entropy = -torch.sum(probs * torch.log(probs + 1e-8), dim=-1)
            norm_conf = 1.0 - (entropy / math.log(self.num_classes))
            combined_conf = 0.5 * halt_prob + 0.5 * norm_conf

            if step == self.max_steps:
                # Force halt on final step
                just_halted = active_mask
            else:
                just_halted = active_mask & (combined_conf >= threshold)

            # Record predictions for samples that halted on this step
            final_logits[just_halted] = logits[just_halted]
            active_mask = active_mask & (~just_halted)

            if not active_mask.any() or step == self.max_steps:
                break

            # Increment step count for samples that need another expert pass
            steps_taken[active_mask] += 1.0

            # Route active potatoes ONLY to their predicted top-1 expert
            chosen_experts = router_logits.argmax(dim=-1)  # (B,)
            z_next = z.clone()

            for exp_idx in range(self.num_experts):
                sample_mask = active_mask & (chosen_experts == exp_idx)
                if sample_mask.any():
                    z_next[sample_mask] = self.experts[exp_idx](z[sample_mask])

            z = z_next

        return final_logits, steps_taken


# =====================================================================
# 3. Training & Evaluation Loop
# =====================================================================
def evaluate_cifar10(model: ParadeOfExpertsCIFAR, loader: DataLoader, device: torch.device, threshold: float = 0.90):
    """
    Evaluates accuracy, average expert passes per image, and the breakdown
    of how many images halted at each step on standard CIFAR-10.
    """
    model.eval()
    correct = 0
    total = 0
    total_steps = 0.0
    step_counts = torch.zeros(model.max_steps + 1, device=device)

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            logits, steps_taken = model.forward_inference(images, threshold=threshold)

            preds = logits.argmax(dim=-1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            total_steps += steps_taken.sum().item()

            # Count how many samples halted at each step (0..max_steps)
            step_Counts_batch = torch.bincount(steps_taken.long(), minlength=model.max_steps + 1)
            step_counts += step_Counts_batch

    avg_steps = total_steps / max(1, total)
    halt_distribution = (step_counts / max(1, total)).cpu().tolist()
    return correct / max(1, total), avg_steps, halt_distribution


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running Parade of Experts (Winner-Take-All) on standard CIFAR-10: {device}")

    # Hyperparameters
    batch_size = 128
    epochs = 25
    lr = 1e-3
    lambda_router = 0.5
    lambda_halt = 0.25

    train_loader, test_loader = get_cifar10_dataloaders(
        root="./data", batch_size=batch_size, num_workers=2
    )

    model = ParadeOfExpertsCIFAR(
        d_model=256,
        num_classes=10,
        num_experts=8,
        max_steps=3
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    for epoch in range(1, epochs + 1):
        model.train()
        # Decay the anti-collapse margin over the first 5 epochs
        margin_weight = max(0.0, 0.15 * (1.0 - (epoch - 1) / 5.0))

        running_acc = 0.0
        running_router_acc = 0.0
        last_dist = []

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            task_loss, router_loss, halt_loss, metrics = model.forward_train_wta(
                images, labels, margin_weight=margin_weight
            )

            loss = task_loss + lambda_router * router_loss + lambda_halt * halt_loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            running_acc += metrics["final_acc"]
            running_router_acc += metrics["router_oracle_acc"]
            last_dist = metrics["expert_distribution"]

        scheduler.step()
        num_batches = len(train_loader)
        test_acc, avg_steps, halt_dist = evaluate_cifar10(model, test_loader, device, threshold=0.90)

        dist_str = ", ".join([f"E{i}:{p:.2f}" for i, p in enumerate(last_dist)])
        halt_str = ", ".join([f"S{s}:{p:.2f}" for s, p in enumerate(halt_dist)])

        print(
            f"Epoch [{epoch:02d}/{epochs:02d}] "
            f"Train Acc: {running_acc / num_batches:.3f} | "
            f"Router->Oracle Acc: {running_router_acc / num_batches:.3f} | "
            f"Test Acc: {test_acc:.3f} | "
            f"Win Dist: [{dist_str}] | "
            f"Avg Passes: {avg_steps:.2f} ({halt_str})"
        )


if __name__ == "__main__":
    main()