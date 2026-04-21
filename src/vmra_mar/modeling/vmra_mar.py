"""VMRA-MaR assembly with a frozen Mirai backbone and trainable PyTorch heads."""

from __future__ import annotations

from pathlib import Path

import torch
from torch import nn

from vmra_mar.modeling.asymmetry import LongitudinalAsymmetryTracker, SpatialAsymmetryDetector
from vmra_mar.modeling.hazard import AdditiveHazardLayer
from vmra_mar.modeling.mirai_base import FrozenMiraiImageEncoder, MiraiExamEncoder
from vmra_mar.modeling.vmrnn import VMRNNEncoder


class VMRAMaRModel(nn.Module):
    def __init__(
        self,
        snapshot_path: str | Path | None = None,
        transformer_snapshot_path: str | Path | None = None,
        mirai_package_root: str | Path | None = None,
        exam_hidden_dim: int = 512,
        vmrnn_hidden_dim: int = 128,
        exam_dropout: float = 0.1,
        vmrnn_vss_backend: str = "vmamba",
        vmamba_d_state: int = 16,
        vmamba_drop_path: float = 0.0,
        vmrnn_released_weights_path: str | Path | None = None,
    ) -> None:
        super().__init__()
        self.image_encoder = FrozenMiraiImageEncoder(snapshot_path=snapshot_path, mirai_package_root=mirai_package_root)
        self.exam_encoder = MiraiExamEncoder(
            self.image_encoder,
            hidden_dim=exam_hidden_dim,
            dropout=exam_dropout,
            transformer_snapshot_path=transformer_snapshot_path,
        )
        self.temporal_projection = nn.Linear(self.exam_encoder.out_dim, vmrnn_hidden_dim)
        self.spatial_detector = SpatialAsymmetryDetector()
        self.longitudinal_tracker = LongitudinalAsymmetryTracker()
        self.vmrnn = VMRNNEncoder(
            input_dim=vmrnn_hidden_dim,
            hidden_dim=vmrnn_hidden_dim,
            dropout=exam_dropout,
            vss_backend=vmrnn_vss_backend,
            vmamba_d_state=vmamba_d_state,
            vmamba_drop_path=vmamba_drop_path,
            released_weight_path=vmrnn_released_weights_path,
        )
        self.hazard = AdditiveHazardLayer(input_dim=vmrnn_hidden_dim + 1)
        self.risk_feature_dim = 1

    def frozen_encoder_parameters(self) -> list[nn.Parameter]:
        return list(self.image_encoder.image_model.parameters())

    def forward(self, images: torch.Tensor, view_mask: torch.Tensor, exam_mask: torch.Tensor) -> dict[str, torch.Tensor]:
        exam_embeddings, feature_maps = self.exam_encoder(images, view_mask)
        exam_embeddings = self.temporal_projection(exam_embeddings)

        asymmetry_scores, coords, coord_valid = self.spatial_detector(feature_maps, view_mask)
        r_aa = self.longitudinal_tracker(
            asymmetry_scores,
            coords,
            coord_valid,
            exam_mask,
            window_size=max(self.spatial_detector.latent_h, self.spatial_detector.latent_w),
        )

        history_embedding, states, reconstructions = self.vmrnn(exam_embeddings, exam_mask)
        final_features = torch.cat([history_embedding, r_aa.unsqueeze(-1)], dim=-1)
        logits = self.hazard(final_features)
        probs = torch.sigmoid(logits)
        return {
            "logits": logits,
            "probs": probs,
            "exam_asymmetry": asymmetry_scores,
            "r_aa": r_aa,
            "history_embedding": history_embedding,
            "states": states,
            "vmrnn_reconstruction": reconstructions,
        }
