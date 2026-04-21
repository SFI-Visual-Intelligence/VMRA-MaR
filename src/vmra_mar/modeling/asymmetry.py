"""AsymMirai-aligned asymmetry scoring and longitudinal fusion."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


def hybrid_asymmetry(
    left: torch.Tensor,
    right: torch.Tensor,
    latent_h: int = 5,
    latent_w: int = 5,
    flexible: bool = False,
    topk: int | None = None,
    bias_params: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    diff = torch.abs(left - right if bias_params is None else left - right + bias_params)
    kernel_h = max(diff.shape[-2] // latent_h, 1)
    kernel_w = max(diff.shape[-1] // latent_w, 1)
    stride = (1, 1) if flexible else (kernel_h, kernel_w)
    diff = F.max_pool2d(diff, (kernel_h, kernel_w), stride=stride)
    diff = torch.norm(diff, dim=-3)

    if topk is None:
        max_by_row, x_indices = torch.max(diff, dim=-1)
        max_scores, y_indices = torch.max(max_by_row, dim=-1)
        x_indices = x_indices.gather(1, y_indices.unsqueeze(-1)).squeeze(-1)
        return max_scores, {
            "y_argmax": y_indices.detach(),
            "x_argmax": x_indices.detach(),
            "heatmap": diff.detach(),
        }

    topk_scores, _ = torch.topk(diff.view(diff.shape[0], -1), topk, dim=-1)
    return topk_scores, {
        "y_argmax": torch.full((diff.shape[0],), -1, device=diff.device, dtype=torch.long),
        "x_argmax": torch.full((diff.shape[0],), -1, device=diff.device, dtype=torch.long),
        "heatmap": diff.detach(),
    }


class SpatialAsymmetryDetector(nn.Module):
    def __init__(
        self,
        latent_h: int = 5,
        latent_w: int = 5,
        flexible: bool = False,
        embedding_channel: int = 512,
        initial_asym_mean: float = 8_000_000.0,
        initial_asym_std: float = 1_520_381.0,
    ) -> None:
        super().__init__()
        self.latent_h = latent_h
        self.latent_w = latent_w
        self.flexible = flexible
        self.cc_stretch_params = nn.Parameter(torch.ones(embedding_channel))
        self.mlo_stretch_params = nn.Parameter(torch.ones(embedding_channel))
        self.learned_asym_mean = nn.Parameter(torch.tensor(float(initial_asym_mean), dtype=torch.float32))
        self.learned_asym_std = nn.Parameter(torch.tensor(float(initial_asym_std), dtype=torch.float32))
        self.pair_definitions = (
            (0, 1, "CC"),
            (2, 3, "MLO"),
        )

    def _stretch(self, tensor: torch.Tensor, view_name: str) -> torch.Tensor:
        params = self.cc_stretch_params if view_name == "CC" else self.mlo_stretch_params
        return tensor * params.view(1, -1, 1, 1)

    def forward(self, feature_maps: torch.Tensor, view_mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size, time_steps, _, _, _, _ = feature_maps.shape
        raw_pair_scores = []
        normalized_pair_scores = []
        pair_coords = []
        pair_valid = []

        for left_index, right_index, view_name in self.pair_definitions:
            left = feature_maps[:, :, left_index]
            right = torch.flip(feature_maps[:, :, right_index], dims=[-1])
            valid = view_mask[:, :, left_index] & view_mask[:, :, right_index]

            flat_left = left.reshape(batch_size * time_steps, *left.shape[2:])
            flat_right = right.reshape(batch_size * time_steps, *right.shape[2:])
            flat_valid = valid.reshape(batch_size * time_steps)

            flat_raw_scores = flat_left.new_zeros(batch_size * time_steps)
            flat_normalized_scores = flat_left.new_zeros(batch_size * time_steps)
            flat_coords = flat_left.new_zeros(batch_size * time_steps, 2)
            if flat_valid.any():
                valid_left = self._stretch(flat_left[flat_valid], view_name)
                valid_right = self._stretch(flat_right[flat_valid], view_name)
                valid_scores, details = hybrid_asymmetry(
                    valid_left,
                    valid_right,
                    latent_h=self.latent_h,
                    latent_w=self.latent_w,
                    flexible=self.flexible,
                )
                normalized = (valid_scores - self.learned_asym_mean) / self.learned_asym_std.abs().clamp(min=1e-6)
                flat_raw_scores[flat_valid] = valid_scores.to(flat_raw_scores.dtype)
                flat_normalized_scores[flat_valid] = torch.sigmoid(normalized).to(flat_normalized_scores.dtype)
                flat_coords[flat_valid, 0] = details["y_argmax"].to(flat_coords.dtype)
                flat_coords[flat_valid, 1] = details["x_argmax"].to(flat_coords.dtype)

            raw_pair_scores.append(flat_raw_scores.reshape(batch_size, time_steps))
            normalized_pair_scores.append(flat_normalized_scores.reshape(batch_size, time_steps))
            pair_coords.append(flat_coords.reshape(batch_size, time_steps, 2))
            pair_valid.append(valid)

        raw_scores = torch.stack(raw_pair_scores, dim=-1)
        normalized_scores = torch.stack(normalized_pair_scores, dim=-1)
        coords = torch.stack(pair_coords, dim=-2)
        valid = torch.stack(pair_valid, dim=-1)

        valid_float = valid.float()
        exam_scores = (normalized_scores * valid_float).sum(dim=-1) / valid_float.sum(dim=-1).clamp(min=1.0)
        dominant_pair = raw_scores.masked_fill(~valid, float("-inf")).argmax(dim=-1)
        dominant_coords = coords.gather(
            dim=2,
            index=dominant_pair.unsqueeze(-1).unsqueeze(-1).expand(batch_size, time_steps, 1, 2),
        ).squeeze(2)
        coord_valid = valid.any(dim=-1)
        dominant_coords = dominant_coords * coord_valid.unsqueeze(-1).float()
        return exam_scores, dominant_coords, coord_valid


class LongitudinalAsymmetryTracker(nn.Module):
    def __init__(self, threshold_ratio: float = 0.4, persistent_weight: float = 1.0) -> None:
        super().__init__()
        self.threshold_ratio = threshold_ratio
        self.persistent_weight = persistent_weight

    def forward(
        self,
        scores: torch.Tensor,
        coords: torch.Tensor,
        coord_valid: torch.Tensor,
        exam_mask: torch.Tensor,
        window_size: int,
    ) -> torch.Tensor:
        valid = exam_mask & coord_valid
        if not valid.any():
            return scores.new_zeros(scores.size(0))

        threshold = self.threshold_ratio * float(window_size)
        weights = valid.float()
        for step in range(1, scores.size(1)):
            persistent = valid[:, step] & valid[:, step - 1]
            displacement = torch.norm(coords[:, step] - coords[:, step - 1], dim=-1)
            persistent = persistent & (displacement <= threshold)
            persistent_weight = persistent.float() * self.persistent_weight
            weights[:, step] += persistent_weight
            weights[:, step - 1] += persistent_weight

        weighted_scores = scores * weights
        return weighted_scores.sum(dim=1) / weights.sum(dim=1).clamp(min=1.0)
