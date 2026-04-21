"""Mirai-aligned cumulative probability output head."""

from __future__ import annotations

import torch
from torch import nn

from vmra_mar.data import MAX_HORIZON


class AdditiveHazardLayer(nn.Module):
    def __init__(self, input_dim: int, max_followup: int = MAX_HORIZON, make_probs_indep: bool = False) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.max_followup = max_followup
        self.make_probs_indep = make_probs_indep
        self.hazard_fc = nn.Linear(input_dim, max_followup)
        self.base_hazard_fc = nn.Linear(input_dim, 1)
        self.relu = nn.ReLU(inplace=True)
        mask = torch.ones(max_followup, max_followup)
        mask = torch.tril(mask, diagonal=0).t()
        self.register_buffer("upper_triangular_mask", mask, persistent=False)

    def hazards(self, features: torch.Tensor) -> torch.Tensor:
        return self.relu(self.hazard_fc(features))

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        hazards = self.hazards(features)
        if self.make_probs_indep:
            return hazards

        batch_size, time_steps = hazards.shape
        expanded_hazards = hazards.unsqueeze(-1).expand(batch_size, time_steps, time_steps)
        masked_hazards = expanded_hazards * self.upper_triangular_mask
        cumulative_probability = masked_hazards.sum(dim=1) + self.base_hazard_fc(features)
        return cumulative_probability
