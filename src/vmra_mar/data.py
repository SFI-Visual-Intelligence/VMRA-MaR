"""Dataset loading and torch-ready sequence batches for CSAW-CC."""

from __future__ import annotations

import csv
import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import pydicom
import torch
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF


MAX_HORIZON = 5
VIEW_ORDER = ("LCC", "RCC", "LMLO", "RMLO")
MIRAI_IMAGE_MEAN = 7047.99
MIRAI_IMAGE_STD = 12005.5


@dataclass
class ViewRecord:
    key: str
    filename: str | None
    path: Path | None
    present: bool


@dataclass
class ExamRecord:
    year: int
    views: dict[str, ViewRecord]


@dataclass
class PatientSample:
    patient_id: str
    history: list[ExamRecord]
    case_label: int
    target: list[int]
    mask: list[int]
    years_to_event: int | None
    followup_years: int


@dataclass
class DatasetBundle:
    train: list[PatientSample]
    val: list[PatientSample]
    test: list[PatientSample]
    source_path: Path
    image_root: Path | None


def _stable_hash(text: str) -> int:
    return int(hashlib.sha256(text.encode("utf-8")).hexdigest()[:8], 16)


def _view_key(row: dict[str, str]) -> str:
    laterality = "L" if row["imagelaterality"].strip().lower().startswith("l") else "R"
    view = row["viewposition"].strip().upper()
    return f"{laterality}{view}"


def _empty_exam(year: int) -> ExamRecord:
    return ExamRecord(
        year=year,
        views={key: ViewRecord(key=key, filename=None, path=None, present=False) for key in VIEW_ORDER},
    )


def _ordered_group(samples: Sequence[PatientSample]) -> list[PatientSample]:
    return sorted(samples, key=lambda sample: (_stable_hash(sample.patient_id), sample.patient_id))


def _split_group(
    samples: Sequence[PatientSample],
    val_ratio: float,
    test_ratio: float,
) -> tuple[list[PatientSample], list[PatientSample], list[PatientSample]]:
    ordered = _ordered_group(samples)
    total = len(ordered)
    if total == 0:
        return [], [], []

    val_count = int(round(total * val_ratio))
    test_count = int(round(total * test_ratio))
    if val_ratio > 0 and total >= 3:
        val_count = max(val_count, 1)
    if test_ratio > 0 and total >= 3:
        test_count = max(test_count, 1)

    while val_count + test_count > total:
        if test_count >= val_count and test_count > 0:
            test_count -= 1
        elif val_count > 0:
            val_count -= 1
        else:
            break

    train_count = total - val_count - test_count
    train = ordered[:train_count]
    val = ordered[train_count : train_count + val_count]
    test = ordered[train_count + val_count :]
    return train, val, test


def _split_samples(
    samples: list[PatientSample],
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
) -> tuple[list[PatientSample], list[PatientSample], list[PatientSample]]:
    if val_ratio < 0 or test_ratio < 0 or val_ratio + test_ratio >= 1:
        raise ValueError("val_ratio and test_ratio must be non-negative and sum to less than 1.")

    cases = [sample for sample in samples if sample.case_label]
    controls = [sample for sample in samples if not sample.case_label]

    train: list[PatientSample] = []
    val: list[PatientSample] = []
    test: list[PatientSample] = []
    for group in (cases, controls):
        group_train, group_val, group_test = _split_group(group, val_ratio=val_ratio, test_ratio=test_ratio)
        train.extend(group_train)
        val.extend(group_val)
        test.extend(group_test)

    train = _ordered_group(train)
    val = _ordered_group(val)
    test = _ordered_group(test)
    return _ensure_nonempty_splits(train, val, test, require_val=val_ratio > 0, require_test=test_ratio > 0)


def _ensure_nonempty_splits(
    train: list[PatientSample],
    val: list[PatientSample],
    test: list[PatientSample],
    require_val: bool = True,
    require_test: bool = True,
) -> tuple[list[PatientSample], list[PatientSample], list[PatientSample]]:
    combined = train + val + test
    if not combined:
        raise ValueError("No patient samples could be constructed from the provided CSV.")
    if not train:
        train = combined[:1]
    if require_val and not val:
        val = combined[:1]
    if require_test and not test:
        test = combined[-1:]
    return train, val, test


def _has_any_local_image(history: Sequence[ExamRecord]) -> bool:
    for exam in history:
        for view in exam.views.values():
            if view.path is not None and view.path.exists():
                return True
    return False


def load_dataset_bundle(
    csv_path: str | Path,
    image_root: str | Path | None = None,
    patient_ids: Sequence[str] | None = None,
    max_patients: int | None = None,
    sequence_length: int = MAX_HORIZON,
    require_local: bool = False,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
) -> DatasetBundle:
    csv_path = Path(csv_path)
    image_root = Path(image_root) if image_root is not None else None
    patient_filter = {str(item) for item in patient_ids} if patient_ids is not None else None

    per_patient: dict[str, dict[int, ExamRecord]] = {}
    patient_case_label: dict[str, int] = {}

    with csv_path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            patient_id = row["anon_patientid"]
            if patient_filter is not None and patient_id not in patient_filter:
                continue
            year = int(row["exam_year"])
            patient_case_label[patient_id] = int(row["x_case"])
            patient_bucket = per_patient.setdefault(patient_id, {})
            exam = patient_bucket.setdefault(year, _empty_exam(year))
            filename = row["anon_filename"]
            path = (image_root / filename) if image_root is not None else None
            present = path.exists() if path is not None else False
            exam.views[_view_key(row)] = ViewRecord(key=_view_key(row), filename=filename, path=path, present=present)

    patient_list = sorted(per_patient)
    if max_patients is not None:
        patient_list = patient_list[:max_patients]

    samples: list[PatientSample] = []
    for patient_id in patient_list:
        exams = [per_patient[patient_id][year] for year in sorted(per_patient[patient_id])]
        case_label = patient_case_label[patient_id]

        if len(exams) > 1:
            history = exams[:-1]
            anchor_year = history[-1].year
            terminal_year = exams[-1].year
        else:
            history = exams[:]
            anchor_year = exams[-1].year
            terminal_year = anchor_year

        history = history[-sequence_length:]
        gap_years = max(terminal_year - anchor_year, 0)

        if case_label:
            years_to_event = max(1, gap_years) if len(exams) > 1 else 1
            if years_to_event > MAX_HORIZON:
                continue
            followup_years = MAX_HORIZON
            target = [1 if years_to_event <= horizon else 0 for horizon in range(1, MAX_HORIZON + 1)]
            mask = [1] * MAX_HORIZON
        else:
            years_to_event = None
            followup_years = min(gap_years, MAX_HORIZON)
            if followup_years <= 0:
                continue
            target = [0] * MAX_HORIZON
            mask = [1 if horizon <= followup_years else 0 for horizon in range(1, MAX_HORIZON + 1)]

        if require_local and not _has_any_local_image(history):
            continue

        samples.append(
            PatientSample(
                patient_id=patient_id,
                history=history,
                case_label=case_label,
                target=target,
                mask=mask,
                years_to_event=years_to_event,
                followup_years=followup_years,
            )
        )

    train, val, test = _split_samples(samples, val_ratio=val_ratio, test_ratio=test_ratio)
    return DatasetBundle(train=train, val=val, test=test, source_path=csv_path, image_root=image_root)


class MammogramSequenceDataset(Dataset):
    def __init__(
        self,
        samples: Sequence[PatientSample],
        sequence_length: int = MAX_HORIZON,
        image_size: tuple[int, int] = (512, 640),
        mean: float = MIRAI_IMAGE_MEAN,
        std: float = MIRAI_IMAGE_STD,
    ) -> None:
        self.samples = list(samples)
        self.sequence_length = sequence_length
        self.image_size = image_size
        self.mean = mean
        self.std = std

    def __len__(self) -> int:
        return len(self.samples)

    def _load_dicom(self, path: Path) -> torch.Tensor:
        dicom = pydicom.dcmread(path)
        image = torch.tensor(dicom.pixel_array, dtype=torch.float32).unsqueeze(0)
        image = TF.resize(image, list(self.image_size), antialias=True)
        image = image.repeat(3, 1, 1)
        image = (image - self.mean) / self.std
        return image

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str]:
        sample = self.samples[index]
        height, width = self.image_size
        images = torch.zeros(self.sequence_length, len(VIEW_ORDER), 3, height, width, dtype=torch.float32)
        view_mask = torch.zeros(self.sequence_length, len(VIEW_ORDER), dtype=torch.bool)
        exam_mask = torch.zeros(self.sequence_length, dtype=torch.bool)

        padded_history = list(sample.history[-self.sequence_length :])
        pad_count = self.sequence_length - len(padded_history)
        exam_slots: list[ExamRecord | None] = [None] * pad_count + padded_history

        for exam_index, exam in enumerate(exam_slots):
            if exam is None:
                continue
            for view_index, view_key in enumerate(VIEW_ORDER):
                view = exam.views[view_key]
                if view.path is None or not view.path.exists():
                    continue
                images[exam_index, view_index] = self._load_dicom(view.path)
                view_mask[exam_index, view_index] = True
            if view_mask[exam_index].all():
                exam_mask[exam_index] = True
            else:
                images[exam_index].zero_()
                view_mask[exam_index].zero_()

        return {
            "patient_id": sample.patient_id,
            "images": images,
            "view_mask": view_mask,
            "exam_mask": exam_mask,
            "target": torch.tensor(sample.target, dtype=torch.float32),
            "target_mask": torch.tensor(sample.mask, dtype=torch.float32),
        }
