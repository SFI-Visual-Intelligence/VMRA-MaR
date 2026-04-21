"""Formal Mirai image and multi-view fusion modules."""

from __future__ import annotations

import copy
import sys
from pathlib import Path

import torch
from torch import nn

from vmra_mar.paths import (
    DEFAULT_MIRAI_PACKAGE_ROOT,
    DEFAULT_MIRAI_SNAPSHOT,
    DEFAULT_MIRAI_TRANSFORMER_SNAPSHOT,
)

MAX_FOLLOWUP = 5
FORMAL_VIEW_SEQUENCE = (
    ("LCC", 0, 1),
    ("RCC", 0, 0),
    ("LMLO", 1, 1),
    ("RMLO", 1, 0),
)


def _ensure_import_path(mirai_package_root: Path) -> None:
    package_root = str(mirai_package_root)
    if package_root not in sys.path:
        sys.path.insert(0, package_root)


def _unwrap_loaded_model(model: object) -> nn.Module:
    if isinstance(model, dict):
        model = model["model"]
    if isinstance(model, nn.DataParallel):
        model = model.module.cpu()
    if not isinstance(model, nn.Module):
        raise TypeError(f"Expected a torch.nn.Module snapshot, found {type(model)!r}.")
    return model


def _zero_risk_factors_for_args(
    args: object,
    batch_size: int,
    device: torch.device,
    dtype: torch.dtype,
) -> list[torch.Tensor] | None:
    if not bool(getattr(args, "use_risk_factors", False)):
        return None

    key_to_dim = getattr(args, "risk_factor_key_to_num_class", None)
    risk_factor_keys = list(getattr(args, "risk_factor_keys", []) or [])
    if (not key_to_dim) and risk_factor_keys:
        from onconet.utils.risk_factors import RiskFactorVectorizer

        RiskFactorVectorizer(args)
        key_to_dim = args.risk_factor_key_to_num_class

    if key_to_dim and risk_factor_keys:
        return [
            torch.zeros(batch_size, int(key_to_dim[key]), device=device, dtype=dtype)
            for key in risk_factor_keys
        ]

    rf_dim = int(getattr(args, "rf_dim", 0) or 0)
    if rf_dim > 0:
        return [torch.zeros(batch_size, rf_dim, device=device, dtype=dtype)]
    return None


class FrozenMiraiImageEncoder(nn.Module):
    def __init__(
        self,
        snapshot_path: str | Path | None = None,
        mirai_package_root: str | Path | None = None,
        cutoff_layer: str = "layer4_1",
    ) -> None:
        super().__init__()
        self.snapshot_path = Path(snapshot_path) if snapshot_path is not None else DEFAULT_MIRAI_SNAPSHOT
        self.mirai_package_root = Path(mirai_package_root) if mirai_package_root is not None else DEFAULT_MIRAI_PACKAGE_ROOT
        self.cutoff_layer = cutoff_layer
        _ensure_import_path(self.mirai_package_root)

        self.image_model = self._load_image_model().eval()
        self.backbone = self._extract_backbone().eval()
        for parameter in self.image_model.parameters():
            parameter.requires_grad = False
        self.out_channels = int(getattr(self.image_model._model.args, "hidden_dim", 512))
        self.hidden_dim = int(getattr(self.image_model._model.args, "img_only_dim", self.out_channels))
        self._feature_shape_cache: dict[tuple[int, int], tuple[int, int, int]] = {}

    def _zero_risk_factors(self, batch_size: int, device: torch.device, dtype: torch.dtype) -> list[torch.Tensor] | None:
        return _zero_risk_factors_for_args(self.image_model._model.args, batch_size, device, dtype)

    def _load_image_model(self) -> nn.Module:
        model = _unwrap_loaded_model(torch.load(self.snapshot_path, map_location="cpu", weights_only=False))
        if not hasattr(model, "_model"):
            raise ValueError("Mirai image encoder snapshot did not expose the expected `_model` backbone.")
        return model

    def _extract_backbone(self) -> nn.Module:
        modules = []
        for name, module in self.image_model._model.named_children():
            modules.append(module)
            if name == self.cutoff_layer:
                break
        return nn.Sequential(*modules)

    def train(self, mode: bool = True) -> "FrozenMiraiImageEncoder":
        del mode
        super().train(False)
        self.image_model.eval()
        self.backbone.eval()
        return self

    def output_feature_shape(self, image_size: tuple[int, int], device: torch.device) -> tuple[int, int, int]:
        cached = self._feature_shape_cache.get(image_size)
        if cached is None:
            height, width = image_size
            dummy = torch.zeros(1, 3, height, width, device=device, dtype=torch.float32)
            with torch.no_grad():
                feature_map = self.backbone(dummy)
            cached = (feature_map.size(1), feature_map.size(2), feature_map.size(3))
            self._feature_shape_cache[image_size] = cached
        return cached

    def forward(self, images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        risk_factors = self._zero_risk_factors(images.size(0), images.device, images.dtype)
        with torch.no_grad():
            _, hidden, activations = self.image_model(images, risk_factors=risk_factors, batch=None)
        hidden = hidden[:, : self.hidden_dim]
        feature_maps = activations["activ"]
        return hidden, feature_maps


class MiraiExamEncoder(nn.Module):
    def __init__(
        self,
        image_encoder: FrozenMiraiImageEncoder,
        hidden_dim: int = 512,
        num_heads: int = 8,
        num_layers: int = 1,
        dropout: float = 0.1,
        pool_name: str = "Simple_AttentionPool",
        transformer_snapshot_path: str | Path | None = None,
    ) -> None:
        super().__init__()
        self.image_encoder = image_encoder
        self.transformer_snapshot_path = (
            Path(transformer_snapshot_path) if transformer_snapshot_path is not None else DEFAULT_MIRAI_TRANSFORMER_SNAPSHOT
        )
        self.transformer_model = self._load_or_build_transformer(
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            dropout=dropout,
            pool_name=pool_name,
        )
        self.out_dim = int(self.transformer_model.args.transfomer_hidden_dim)
        self.uses_official_transformer_snapshot = self.transformer_snapshot_path.exists()

    def _load_or_build_transformer(
        self,
        *,
        hidden_dim: int,
        num_heads: int,
        num_layers: int,
        dropout: float,
        pool_name: str,
    ) -> nn.Module:
        _ensure_import_path(self.image_encoder.mirai_package_root)
        import onconet.models.pools.attention_pool  # noqa: F401
        import onconet.models.pools.average_pool  # noqa: F401
        import onconet.models.pools.max_pool  # noqa: F401
        from onconet.models.hiddens_transfomer import AllImageTransformer

        if self.transformer_snapshot_path.exists():
            return _unwrap_loaded_model(torch.load(self.transformer_snapshot_path, map_location="cpu", weights_only=False))

        args = copy.deepcopy(self.image_encoder.image_model._model.args)
        args.wrap_model = False
        args.hidden_dim = hidden_dim
        args.transfomer_hidden_dim = hidden_dim
        args.use_precomputed_hiddens = True
        args.model_name = "transformer"
        args.precomputed_hidden_dim = self.image_encoder.hidden_dim
        args.num_images = len(FORMAL_VIEW_SEQUENCE)
        args.num_layers = num_layers
        args.num_heads = num_heads
        args.dropout = dropout
        args.use_risk_factors = False
        args.deep_risk_factor_pool = False
        args.pool_name = pool_name
        args.num_classes = MAX_FOLLOWUP
        args.survival_analysis_setup = True
        args.pred_both_sides = False
        args.max_followup = MAX_FOLLOWUP
        args.pred_missing_mammos = False
        args.also_pred_given_mammos = False
        args.predict_birads = False
        args.pred_risk_factors = False
        args.use_pred_risk_factors_at_test = False
        args.use_pred_risk_factors_if_unk = False
        args.mask_prob = 0
        args.make_probs_indep = bool(getattr(args, "make_probs_indep", False))
        return AllImageTransformer(args)

    def _transformer_batch(self, batch_size: int, device: torch.device) -> dict[str, torch.Tensor]:
        view_seq = torch.tensor([view for _, view, _ in FORMAL_VIEW_SEQUENCE], device=device, dtype=torch.long)
        side_seq = torch.tensor([side for _, _, side in FORMAL_VIEW_SEQUENCE], device=device, dtype=torch.long)
        return {
            "time_seq": torch.zeros(batch_size, len(FORMAL_VIEW_SEQUENCE), device=device, dtype=torch.long),
            "view_seq": view_seq.unsqueeze(0).expand(batch_size, -1),
            "side_seq": side_seq.unsqueeze(0).expand(batch_size, -1),
        }

    def forward(self, images: torch.Tensor, view_mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size, time_steps, num_views, channels, height, width = images.shape
        if num_views != len(FORMAL_VIEW_SEQUENCE):
            raise ValueError(f"Formal Mirai fusion expects {len(FORMAL_VIEW_SEQUENCE)} views, got {num_views}.")

        flat_exam_mask = view_mask.reshape(batch_size * time_steps, num_views)
        partial_exam_mask = flat_exam_mask.any(dim=1) & ~flat_exam_mask.all(dim=1)
        if partial_exam_mask.any():
            raise RuntimeError("Formal Mirai fusion requires complete four-view exams; partial exams are not supported.")

        total_views = batch_size * time_steps * num_views
        flat_images = images.reshape(total_views, channels, height, width)
        flat_view_mask = view_mask.reshape(total_views)

        if flat_view_mask.any():
            valid_hidden, valid_feature_maps = self.image_encoder(flat_images[flat_view_mask])
            hidden = valid_hidden.new_zeros((total_views, valid_hidden.size(-1)))
            hidden[flat_view_mask] = valid_hidden
            feature_maps = valid_feature_maps.new_zeros((total_views, *valid_feature_maps.shape[1:]))
            feature_maps[flat_view_mask] = valid_feature_maps
        else:
            feature_channels, feature_height, feature_width = self.image_encoder.output_feature_shape(
                (height, width),
                device=flat_images.device,
            )
            hidden = flat_images.new_zeros((total_views, self.image_encoder.hidden_dim))
            feature_maps = flat_images.new_zeros((total_views, feature_channels, feature_height, feature_width))

        hidden = hidden.reshape(batch_size * time_steps, num_views, -1)
        exam_embeddings = hidden.new_zeros(batch_size * time_steps, self.out_dim)
        complete_exam_mask = flat_exam_mask.all(dim=1)
        if complete_exam_mask.any():
            transformer_batch = self._transformer_batch(int(complete_exam_mask.sum().item()), hidden.device)
            exam_tokens = hidden[complete_exam_mask]
            projected = self.transformer_model.projection_layer(exam_tokens)
            encoded = self.transformer_model.transformer(
                projected,
                transformer_batch["time_seq"],
                transformer_batch["view_seq"],
                transformer_batch["side_seq"],
            )
            _, pooled_hidden = self.transformer_model.aggregate_and_classify(encoded.transpose(1, 2).unsqueeze(-1))
            exam_embeddings[complete_exam_mask] = pooled_hidden

        exam_embeddings = exam_embeddings.reshape(batch_size, time_steps, self.out_dim)
        feature_maps = feature_maps.reshape(
            batch_size,
            time_steps,
            num_views,
            feature_maps.size(1),
            feature_maps.size(2),
            feature_maps.size(3),
        )
        return exam_embeddings, feature_maps
