"""Project-wide default paths for assets, data, and outputs."""

from __future__ import annotations

from pathlib import Path


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


DEFAULT_DATA_ROOT = repo_root() / "data" / "csaw_cc"
DEFAULT_METADATA_PATH = DEFAULT_DATA_ROOT / "metadata" / "csaw_cc.csv"
DEFAULT_IMAGE_ROOT = DEFAULT_DATA_ROOT / "dicom"
DEFAULT_PRETRAINED_ROOT = DEFAULT_DATA_ROOT / "models"
DEFAULT_ASSET_SNAPSHOT_ROOT = repo_root() / "assets" / "snapshots"
DEFAULT_MIRAI_SNAPSHOT = DEFAULT_PRETRAINED_ROOT / "mgh_mammo_MIRAI_Base_May20_2019.p"
DEFAULT_MIRAI_TRANSFORMER_SNAPSHOT = DEFAULT_ASSET_SNAPSHOT_ROOT / "mgh_mammo_cancer_MIRAI_Transformer_Jan13_2020.p"
# Released VMRNN checkpoints are optional and are not distributed in this
# paper-submission repository. Pass one explicitly with
# --vmrnn-released-weights-path when reproducing a run that used it.
DEFAULT_VMRNN_RELEASED_WEIGHTS: Path | None = None
DEFAULT_MIRAI_PACKAGE_ROOT = repo_root() / "vendor" / "mirai"
DEFAULT_OUTPUT_ROOT = repo_root() / "artifacts" / "train"
