"""PyTorch training entry point for VMRA-MaR."""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
import random
from contextlib import nullcontext
from datetime import timedelta
from pathlib import Path

import torch
import torch.distributed as dist
from torch import nn
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from vmra_mar.data import MAX_HORIZON, MammogramSequenceDataset, load_dataset_bundle
from vmra_mar.metrics import build_mirai_censoring_distribution, evaluate_predictions, masked_bce_loss
from vmra_mar.modeling.vmra_mar import VMRAMaRModel
from vmra_mar.paths import (
    DEFAULT_IMAGE_ROOT,
    DEFAULT_METADATA_PATH,
    DEFAULT_MIRAI_PACKAGE_ROOT,
    DEFAULT_MIRAI_SNAPSHOT,
    DEFAULT_MIRAI_TRANSFORMER_SNAPSHOT,
    DEFAULT_OUTPUT_ROOT,
    DEFAULT_VMRNN_RELEASED_WEIGHTS,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the VMRA-MaR pipeline.")
    parser.add_argument("--data", type=Path, default=DEFAULT_METADATA_PATH, help="Path to the CSAW-CC CSV file.")
    parser.add_argument("--image-root", type=Path, default=DEFAULT_IMAGE_ROOT, help="Directory containing DICOM files.")
    parser.add_argument("--snapshot-path", type=Path, default=DEFAULT_MIRAI_SNAPSHOT, help="Path to the frozen Mirai snapshot.")
    parser.add_argument("--transformer-snapshot-path", type=Path, default=DEFAULT_MIRAI_TRANSFORMER_SNAPSHOT, help="Path to the Mirai transformer snapshot. When absent, the formal Mirai transformer architecture is initialized from scratch.")
    parser.add_argument("--mirai-package-root", type=Path, default=DEFAULT_MIRAI_PACKAGE_ROOT, help="Directory that contains the vendored onconet package.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_ROOT, help="Directory used for checkpoints and metrics.")
    parser.add_argument("--batch-size", type=int, default=4, help="Per-process batch size.")
    parser.add_argument("--eval-batch-size", type=int, default=4, help="Evaluation batch size.")
    parser.add_argument("--epochs", type=int, default=30, help="Number of training epochs.")
    parser.add_argument("--learning-rate", type=float, default=1e-3, help="Peak optimizer learning rate.")
    parser.add_argument("--min-learning-rate", type=float, default=1e-5, help="Minimum learning rate reached by the scheduler.")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="AdamW weight decay.")
    parser.add_argument("--warmup-epochs", type=float, default=1.0, help="Linear warmup duration in epochs.")
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1, help="Number of forward passes per optimizer step.")
    parser.add_argument("--clip-grad-norm", type=float, default=1.0, help="Gradient clipping threshold.")
    parser.add_argument("--num-workers", type=int, default=8, help="Number of data loading workers per process.")
    parser.add_argument("--prefetch-factor", type=int, default=2, help="Batches prefetched by each worker.")
    parser.add_argument("--persistent-workers", action="store_true", help="Keep worker processes alive between epochs.")
    parser.add_argument("--drop-last", action="store_true", help="Drop the last incomplete training batch.")
    parser.add_argument("--precision", choices=("auto", "fp32", "amp_fp16", "amp_bf16"), default="auto", help="Mixed precision mode.")
    parser.add_argument("--backend", choices=("auto", "nccl", "gloo"), default="auto", help="Distributed backend.")
    parser.add_argument("--compile-model", action="store_true", help="Compile the trainable model with torch.compile when available.")
    parser.add_argument("--resume", type=Path, default=None, help="Path to a checkpoint produced by a previous run.")
    parser.add_argument("--checkpoint-every", type=int, default=1, help="Epoch interval for checkpoint snapshots.")
    parser.add_argument("--image-height", type=int, default=512, help="Resized image height.")
    parser.add_argument("--image-width", type=int, default=640, help="Resized image width.")
    parser.add_argument("--exam-hidden-dim", type=int, default=512, help="Hidden size used by the formal Mirai multi-view transformer.")
    parser.add_argument("--vmrnn-hidden-dim", type=int, default=128, help="Hidden size used by the temporal VMRNN encoder.")
    parser.add_argument("--exam-dropout", type=float, default=0.1, help="Dropout used inside the multi-view exam encoder.")
    parser.add_argument(
        "--vmrnn-vss-backend",
        choices=("auto", "vmamba"),
        default="vmamba",
        help="Formal VMRNN backend selection. 'auto' is strict and still requires VMamba to be available.",
    )
    parser.add_argument("--vmamba-d-state", type=int, default=16, help="State size used by the true VMamba SS2D kernel.")
    parser.add_argument("--vmamba-drop-path", type=float, default=0.0, help="Drop-path rate used by the true VMamba VSS blocks.")
    parser.add_argument("--vmrnn-released-weights-path", type=Path, default=DEFAULT_VMRNN_RELEASED_WEIGHTS, help="Released VMRNN checkpoint used to initialize compatible VMamba VMRNN weights.")
    parser.add_argument("--pos-weight-mode", choices=("auto", "none"), default="auto", help="Whether to apply automatic positive-class reweighting.")
    parser.add_argument("--pos-weight-max", type=float, default=8.0, help="Upper bound for automatically computed positive-class weights.")
    parser.add_argument("--early-stopping-patience", type=int, default=0, help="Stop after this many non-improving validation epochs. Zero disables early stopping.")
    parser.add_argument("--val-ratio", type=float, default=0.15, help="Fraction of each label bucket assigned to validation.")
    parser.add_argument("--test-ratio", type=float, default=0.15, help="Fraction of each label bucket assigned to test.")
    parser.add_argument("--max-patients", type=int, default=None, help="Optional cap for quick experiments.")
    parser.add_argument("--seed", type=int, default=7, help="Random seed.")
    return parser.parse_args()


def is_distributed() -> bool:
    return int(os.environ.get("WORLD_SIZE", "1")) > 1


def is_main_process() -> bool:
    return int(os.environ.get("RANK", "0")) == 0


def local_rank() -> int:
    return int(os.environ.get("LOCAL_RANK", "0"))


def world_size() -> int:
    return int(os.environ.get("WORLD_SIZE", "1"))


def unwrap_model(model: nn.Module) -> nn.Module:
    return model.module if isinstance(model, DistributedDataParallel) else model


def choose_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda", local_rank())
    return torch.device("cpu")


def choose_backend(requested: str) -> str:
    if requested != "auto":
        return requested
    return "nccl" if torch.cuda.is_available() else "gloo"


def setup_runtime(args: argparse.Namespace) -> torch.device:
    device = choose_device()
    if torch.cuda.is_available():
        torch.cuda.set_device(device)
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
    random.seed(args.seed + int(os.environ.get("RANK", "0")))
    torch.manual_seed(args.seed + int(os.environ.get("RANK", "0")))
    if is_distributed() and not dist.is_initialized():
        dist.init_process_group(
            backend=choose_backend(args.backend),
            timeout=timedelta(minutes=30),
        )
    return device


def cleanup_runtime() -> None:
    if dist.is_initialized():
        dist.destroy_process_group()


def _loader_kwargs(device: torch.device, args: argparse.Namespace) -> dict[str, object]:
    kwargs: dict[str, object] = {
        "num_workers": args.num_workers,
        "pin_memory": device.type == "cuda",
    }
    if args.num_workers > 0:
        kwargs["persistent_workers"] = args.persistent_workers
        kwargs["prefetch_factor"] = args.prefetch_factor
    return kwargs


def build_loaders(args: argparse.Namespace, device: torch.device):
    bundle = load_dataset_bundle(
        args.data,
        image_root=args.image_root,
        max_patients=args.max_patients,
        require_local=True,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
    )
    image_size = (args.image_height, args.image_width)
    train_dataset = MammogramSequenceDataset(bundle.train, image_size=image_size)
    val_dataset = MammogramSequenceDataset(bundle.val, image_size=image_size)
    test_dataset = MammogramSequenceDataset(bundle.test, image_size=image_size)

    sampler = None
    if is_distributed():
        sampler = DistributedSampler(train_dataset, num_replicas=world_size(), rank=dist.get_rank(), shuffle=True, drop_last=args.drop_last)

    loader_kwargs = _loader_kwargs(device, args)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=sampler is None,
        sampler=sampler,
        drop_last=args.drop_last,
        **loader_kwargs,
    )
    eval_loader_kwargs = _loader_kwargs(device, args)
    val_loader = DataLoader(val_dataset, batch_size=args.eval_batch_size, shuffle=False, **eval_loader_kwargs)
    test_loader = DataLoader(test_dataset, batch_size=args.eval_batch_size, shuffle=False, **eval_loader_kwargs)
    return bundle, train_loader, val_loader, test_loader, sampler


def resolve_precision(args: argparse.Namespace, device: torch.device) -> tuple[torch.dtype | None, bool]:
    if device.type != "cuda":
        return None, False
    if args.precision == "auto":
        return torch.float16, True
    if args.precision == "amp_fp16":
        return torch.float16, True
    if args.precision == "amp_bf16":
        return torch.bfloat16, False
    return None, False


def autocast_context(device: torch.device, amp_dtype: torch.dtype | None):
    if amp_dtype is None:
        return nullcontext()
    return torch.autocast(device_type=device.type, dtype=amp_dtype)


def build_model(args: argparse.Namespace, device: torch.device) -> nn.Module:
    model = VMRAMaRModel(
        snapshot_path=args.snapshot_path,
        transformer_snapshot_path=args.transformer_snapshot_path,
        mirai_package_root=args.mirai_package_root,
        exam_hidden_dim=args.exam_hidden_dim,
        vmrnn_hidden_dim=args.vmrnn_hidden_dim,
        exam_dropout=args.exam_dropout,
        vmrnn_vss_backend=args.vmrnn_vss_backend,
        vmamba_d_state=args.vmamba_d_state,
        vmamba_drop_path=args.vmamba_drop_path,
        vmrnn_released_weights_path=args.vmrnn_released_weights_path,
    ).to(device)
    if args.compile_model and hasattr(torch, "compile"):
        model = torch.compile(model)
    if is_distributed():
        ddp_kwargs: dict[str, object] = {"find_unused_parameters": False}
        if device.type == "cuda":
            ddp_kwargs["device_ids"] = [device.index]
        model = DistributedDataParallel(model, **ddp_kwargs)
    return model


def build_optimizer(model: nn.Module, args: argparse.Namespace) -> torch.optim.Optimizer:
    frozen_ids = {id(parameter) for parameter in unwrap_model(model).frozen_encoder_parameters()}
    trainable_parameters = [
        parameter
        for parameter in model.parameters()
        if parameter.requires_grad and id(parameter) not in frozen_ids
    ]
    return torch.optim.AdamW(trainable_parameters, lr=args.learning_rate, weight_decay=args.weight_decay)


def build_scheduler(
    optimizer: torch.optim.Optimizer,
    args: argparse.Namespace,
    train_loader: DataLoader,
) -> torch.optim.lr_scheduler.LambdaLR | None:
    steps_per_epoch = math.ceil(max(len(train_loader), 1) / max(args.gradient_accumulation_steps, 1))
    total_steps = max(steps_per_epoch * args.epochs, 1)
    warmup_steps = min(int(args.warmup_epochs * steps_per_epoch), max(total_steps - 1, 0))
    min_ratio = min(args.min_learning_rate / args.learning_rate, 1.0) if args.learning_rate > 0 else 1.0

    def lr_lambda(step: int) -> float:
        if warmup_steps > 0 and step < warmup_steps:
            return float(step + 1) / float(warmup_steps)
        if total_steps <= warmup_steps:
            return min_ratio
        progress = float(step - warmup_steps) / float(max(total_steps - warmup_steps, 1))
        progress = min(max(progress, 0.0), 1.0)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_ratio + (1.0 - min_ratio) * cosine

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)


def build_scaler(device: torch.device, amp_dtype: torch.dtype | None, use_grad_scaler: bool) -> torch.cuda.amp.GradScaler | None:
    if device.type != "cuda" or amp_dtype is None or not use_grad_scaler:
        return None
    return torch.cuda.amp.GradScaler()


def build_pos_weight(bundle, args: argparse.Namespace) -> torch.Tensor | None:
    if args.pos_weight_mode == "none":
        return None

    positives = torch.zeros(MAX_HORIZON, dtype=torch.float32)
    valids = torch.zeros(MAX_HORIZON, dtype=torch.float32)
    for sample in bundle.train:
        target = torch.tensor(sample.target, dtype=torch.float32)
        mask = torch.tensor(sample.mask, dtype=torch.float32)
        positives += target * mask
        valids += mask

    negatives = valids - positives
    pos_weight = torch.ones(MAX_HORIZON, dtype=torch.float32)
    valid_horizons = (positives > 0) & (negatives > 0)
    if valid_horizons.any():
        pos_weight[valid_horizons] = (negatives[valid_horizons] / positives[valid_horizons]).clamp(
            min=1.0,
            max=max(args.pos_weight_max, 1.0),
        )
    return pos_weight


def move_batch(batch: dict[str, torch.Tensor | str], device: torch.device) -> dict[str, torch.Tensor | str]:
    moved: dict[str, torch.Tensor | str] = {}
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            moved[key] = value.to(device, non_blocking=device.type == "cuda")
        else:
            moved[key] = value
    return moved


def reduce_mean(value: torch.Tensor) -> torch.Tensor:
    if dist.is_initialized():
        dist.all_reduce(value, op=dist.ReduceOp.SUM)
        value = value / world_size()
    return value


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer | None,
    scheduler: torch.optim.lr_scheduler.LambdaLR | None,
    scaler: torch.cuda.amp.GradScaler | None,
    device: torch.device,
    amp_dtype: torch.dtype | None,
    pos_weight: torch.Tensor | None,
    args: argparse.Namespace,
) -> float:
    training = optimizer is not None
    model.train(training)
    unwrap_model(model).image_encoder.eval()

    total_loss = torch.zeros(1, device=device)
    total_batches = torch.zeros(1, device=device)
    accumulation = max(args.gradient_accumulation_steps, 1)

    if training:
        optimizer.zero_grad(set_to_none=True)

    for step, raw_batch in enumerate(loader, start=1):
        batch = move_batch(raw_batch, device)
        with autocast_context(device, amp_dtype):
            output = model(batch["images"], batch["view_mask"], batch["exam_mask"])
            loss = masked_bce_loss(output["logits"], batch["target"], batch["target_mask"], pos_weight=pos_weight)

        if training:
            scaled_loss = loss / accumulation
            if scaler is not None:
                scaler.scale(scaled_loss).backward()
            else:
                scaled_loss.backward()

            should_step = step % accumulation == 0 or step == len(loader)
            if should_step:
                if args.clip_grad_norm > 0:
                    if scaler is not None:
                        scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad_norm)
                if scaler is not None:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                if scheduler is not None:
                    scheduler.step()

        total_loss += loss.detach()
        total_batches += 1

    total_loss = reduce_mean(total_loss)
    total_batches = reduce_mean(total_batches)
    return float((total_loss / total_batches.clamp(min=1.0)).item())


def predict(model: nn.Module, loader: DataLoader, device: torch.device, amp_dtype: torch.dtype | None) -> torch.Tensor:
    module = unwrap_model(model)
    module.eval()
    all_probs = []
    with torch.no_grad():
        for raw_batch in loader:
            batch = move_batch(raw_batch, device)
            with autocast_context(device, amp_dtype):
                output = module(batch["images"], batch["view_mask"], batch["exam_mask"])
            all_probs.append(output["probs"].cpu())
    if not all_probs:
        return torch.zeros(0, MAX_HORIZON)
    return torch.cat(all_probs, dim=0)


def checkpoint_payload(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LambdaLR | None,
    scaler: torch.cuda.amp.GradScaler | None,
    history: list[float],
    val_history: list[float],
    epoch: int,
    best_epoch: int,
    best_val_loss: float | None,
    pos_weight: torch.Tensor | None,
    args: argparse.Namespace,
) -> dict[str, object]:
    module = unwrap_model(model)
    return {
        "epoch": epoch,
        "history": history,
        "val_history": val_history,
        "best_epoch": best_epoch,
        "best_val_loss": best_val_loss,
        "pos_weight": pos_weight.tolist() if pos_weight is not None else None,
        "model_state_dict": module.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
        "scaler_state_dict": scaler.state_dict() if scaler is not None else None,
        "snapshot_path": str(module.image_encoder.snapshot_path),
        "mirai_package_root": str(module.image_encoder.mirai_package_root),
        "args": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
    }


def maybe_resume(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LambdaLR | None,
    scaler: torch.cuda.amp.GradScaler | None,
    args: argparse.Namespace,
) -> tuple[int, list[float], list[float], int, float | None]:
    if args.resume is None:
        return 0, [], [], 0, None
    checkpoint = torch.load(args.resume, map_location="cpu", weights_only=False)
    unwrap_model(model).load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    if scheduler is not None and checkpoint.get("scheduler_state_dict") is not None:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    if scaler is not None and checkpoint.get("scaler_state_dict") is not None:
        scaler.load_state_dict(checkpoint["scaler_state_dict"])
    return (
        int(checkpoint.get("epoch", 0)),
        list(checkpoint.get("history", [])),
        list(checkpoint.get("val_history", [])),
        int(checkpoint.get("best_epoch", 0)),
        checkpoint.get("best_val_loss"),
    )


def save_json(output_path: Path, payload: dict[str, object]) -> None:
    output_path.write_text(json.dumps(payload, indent=2))


def train_model(args: argparse.Namespace) -> dict[str, object]:
    device = setup_runtime(args)
    bundle, train_loader, val_loader, test_loader, train_sampler = build_loaders(args, device)
    amp_dtype, use_grad_scaler = resolve_precision(args, device)
    model = build_model(args, device)
    if is_main_process():
        resolved_model = unwrap_model(model)
        print(
            "vmrnn_backend="
            f"{resolved_model.vmrnn.vss_backend} "
            f"vmamba_kernel={resolved_model.vmrnn.vmamba_kernel_backend or 'unavailable'}"
        )
        print(
            "mirai_transformer_snapshot="
            f"{'loaded' if resolved_model.exam_encoder.uses_official_transformer_snapshot else 'missing_formal_architecture_init'} "
            f"vmrnn_released_weights={'loaded' if resolved_model.vmrnn.released_weight_report is not None else 'not_loaded'}"
        )
    optimizer = build_optimizer(model, args)
    scheduler = build_scheduler(optimizer, args, train_loader)
    scaler = build_scaler(device, amp_dtype, use_grad_scaler)
    pos_weight = build_pos_weight(bundle, args)
    censoring_distribution = build_mirai_censoring_distribution(bundle.train)
    start_epoch, history, val_history, best_epoch, best_val_loss = maybe_resume(model, optimizer, scheduler, scaler, args)
    best_state_dict = None
    if args.resume is not None and best_epoch:
        best_state_dict = copy.deepcopy(unwrap_model(model).state_dict())

    metrics: dict[str, object] = {}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    last_epoch = start_epoch
    patience_counter = 0

    try:
        for epoch in range(start_epoch, args.epochs):
            if train_sampler is not None:
                train_sampler.set_epoch(epoch)
            train_loss = run_epoch(model, train_loader, optimizer, scheduler, scaler, device, amp_dtype, pos_weight, args)
            val_loss = run_epoch(model, val_loader, None, None, None, device, amp_dtype, pos_weight, args)
            history.append(train_loss)
            val_history.append(val_loss)
            last_epoch = epoch + 1

            improved = best_val_loss is None or val_loss < best_val_loss - 1e-6
            if improved:
                best_val_loss = val_loss
                best_epoch = epoch + 1
                best_state_dict = copy.deepcopy(unwrap_model(model).state_dict())
                patience_counter = 0
            else:
                patience_counter += 1

            if is_main_process():
                learning_rate = optimizer.param_groups[0]["lr"]
                print(
                    f"epoch={epoch + 1} train_loss={train_loss:.4f} "
                    f"val_loss={val_loss:.4f} lr={learning_rate:.2e}"
                )
                if args.checkpoint_every > 0 and (epoch + 1) % args.checkpoint_every == 0:
                    torch.save(
                        checkpoint_payload(
                            model,
                            optimizer,
                            scheduler,
                            scaler,
                            history,
                            val_history,
                            epoch + 1,
                            best_epoch,
                            best_val_loss,
                            pos_weight,
                            args,
                        ),
                        args.output_dir / f"checkpoint_epoch_{epoch + 1}.pt",
                    )
                if improved:
                    torch.save(
                        checkpoint_payload(
                            model,
                            optimizer,
                            scheduler,
                            scaler,
                            history,
                            val_history,
                            epoch + 1,
                            best_epoch,
                            best_val_loss,
                            pos_weight,
                            args,
                        ),
                        args.output_dir / "best_model.pt",
                    )
            if dist.is_initialized():
                dist.barrier()

            if args.early_stopping_patience > 0 and patience_counter >= args.early_stopping_patience:
                if is_main_process():
                    print(f"early_stopping epoch={epoch + 1} best_epoch={best_epoch} best_val_loss={best_val_loss:.4f}")
                break

        if is_main_process():
            final_state_dict = copy.deepcopy(unwrap_model(model).state_dict())
            if best_state_dict is not None:
                unwrap_model(model).load_state_dict(best_state_dict)
            val_probs = predict(model, val_loader, device, amp_dtype)
            test_probs = predict(model, test_loader, device, amp_dtype)
            metrics = {
                "training_loss": history,
                "validation_loss": val_history,
                "val": evaluate_predictions(
                    bundle.val,
                    val_probs,
                    train_samples=bundle.train,
                    mirai_package_root=args.mirai_package_root,
                    censoring_distribution=censoring_distribution,
                ),
                "test": evaluate_predictions(
                    bundle.test,
                    test_probs,
                    train_samples=bundle.train,
                    mirai_package_root=args.mirai_package_root,
                    censoring_distribution=censoring_distribution,
                ),
                "best_epoch": best_epoch,
                "best_val_loss": best_val_loss,
                "evaluation_checkpoint": "best_model.pt" if best_epoch else "model.pt",
                "pos_weight": pos_weight.tolist() if pos_weight is not None else None,
                "training_alignment": {
                    "loss": "weighted_masked_bce",
                    "pos_weight_mode": args.pos_weight_mode,
                    "mirai_censoring_distribution_available": censoring_distribution is not None,
                    "mirai_transformer_snapshot_loaded": unwrap_model(model).exam_encoder.uses_official_transformer_snapshot,
                    "vmrnn_vss_backend": unwrap_model(model).vmrnn.vss_backend,
                    "vmamba_kernel_backend": unwrap_model(model).vmrnn.vmamba_kernel_backend,
                    "vmrnn_released_weight_report": unwrap_model(model).vmrnn.released_weight_report,
                },
                "counts": {
                    "train": len(bundle.train),
                    "val": len(bundle.val),
                    "test": len(bundle.test),
                },
                "world_size": world_size(),
                "batch_size_per_process": args.batch_size,
                "effective_batch_size": args.batch_size * world_size() * max(args.gradient_accumulation_steps, 1),
            }
            unwrap_model(model).load_state_dict(final_state_dict)
            torch.save(
                checkpoint_payload(
                    model,
                    optimizer,
                    scheduler,
                    scaler,
                    history,
                    val_history,
                    last_epoch,
                    best_epoch,
                    best_val_loss,
                    pos_weight,
                    args,
                ),
                args.output_dir / "model.pt",
            )
            save_json(args.output_dir / "metrics.json", metrics)
        if dist.is_initialized():
            dist.barrier()
    finally:
        cleanup_runtime()

    return metrics


def main() -> None:
    args = parse_args()
    train_model(args)


if __name__ == "__main__":
    main()
