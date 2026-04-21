# VMRA-MaR 
This repository contains the Python implementation for a VMRA-MaR style
longitudinal mammography risk modeling pipeline. The code is organized as a
small PyTorch package that builds patient-level mammogram histories from
CSAW-CC metadata, encodes each screening exam with Mirai-compatible components,
tracks longitudinal asymmetry, models temporal evolution with a VMamba-backed
VMRNN encoder, and predicts cumulative breast cancer risk over a five-year
follow-up horizon.

![Alt text](scripts/structure.png)

## Abstract
Breast cancer remains a leading cause of mortality worldwide and is typically detected via screening programs where healthy people are invited in regular intervals. Automated risk prediction approaches have the potential to improve this process by facilitating dynamically screening of high-risk groups. While most models focus solely on the most recent screening, there is growing interest in exploiting temporal information to capture evolving trends in breast tissue, as inspired by clinical practice. Early methods typically relied on two time steps, and although recent efforts have extended this to multiple time steps using Transformer architectures, challenges remain in fully harnessing the rich temporal dynamics inherent in longitudinal imaging data. In this work, we propose to instead leverage Vision Mamba RNN (VMRNN) with a state-space model (SSM) and LSTM-like memory mechanisms to effectively capture nuanced trends in breast tissue evolution. To further enhance our approach, we incorporate an asymmetry module that utilizes a Spatial Asymmetry Detector (SAD) and Longitudinal Asymmetry Tracker (LAT) to identify clinically relevant bilateral differences. This integrated framework demonstrates notable improvements in predicting cancer onset, especially for the more challenging high-density breast cases and achieves superior performance at extended time points (years four and five), highlighting its potential to advance early breast cancer recognition and enable more personalized screening strategies.

## Project Purpose
The implementation is designed around the thesis experiment of combining three
families of methods for mammography risk prediction:
- Image and multi-view exam encoding.
- Left-right breast asymmetry scoring.
- VMRNN-style recurrent temporal modeling with VMamba selective-scan blocks.
The resulting model consumes up to five prior screening exams per patient and
returns five risk probabilities, one for each follow-up year from year 1 through
year 5.
## Repository Layout
```text
.
|-- pyproject.toml
|-- README.md
|-- scripts/
|   `-- train_8xv100.sh
|-- src/
|   `-- vmra_mar/
|       |-- __init__.py
|       |-- __main__.py
|       |-- data.py
|       |-- math_utils.py
|       |-- metrics.py
|       |-- paths.py
|       |-- train.py
|       `-- modeling/
|           |-- asymmetry.py
|           |-- hazard.py
|           |-- mirai_base.py
|           |-- vmamba_runtime.py
|           |-- vmra_mar.py
|           `-- vmrnn.py
`-- vendor/
    `-- mirai/onconet/
```
Important paths:
- `src/vmra_mar/data.py` builds patient sequences from CSAW-CC metadata and
  returns PyTorch-ready tensors.
- `src/vmra_mar/modeling/vmra_mar.py` assembles the end-to-end model.
- `src/vmra_mar/modeling/mirai_base.py` loads the frozen Mirai image encoder and
  constructs or loads the Mirai multi-view transformer.
- `src/vmra_mar/modeling/asymmetry.py` implements spatial and longitudinal
  asymmetry features.
- `src/vmra_mar/modeling/vmrnn.py` implements the temporal VMRNN encoder.
- `src/vmra_mar/modeling/vmamba_runtime.py` provides the VMamba selective-scan
  runtime wrapper.
- `src/vmra_mar/modeling/hazard.py` maps learned patient features to cumulative
  risk logits.
- `src/vmra_mar/metrics.py` computes training loss and Mirai-compatible
  evaluation metrics.
- `src/vmra_mar/train.py` is the main training and evaluation entry point.
- `vendor/mirai/onconet/` contains vendored Mirai runtime modules needed to load
  and evaluate against public Mirai model interfaces.
## Model Architecture
The core model is `VMRAMaRModel`. Its forward pass has six stages.
### 1. Patient Sequence Input
The data loader groups rows by `anon_patientid` and `exam_year`. For each exam it
expects four mammography views in this order:
```text
LCC, RCC, LMLO, RMLO
```
Each patient sample contains a history of up to five exams. Histories shorter
than five are left-padded with empty exam slots. The model uses an `exam_mask` to
distinguish real exams from padding.
The default image tensor shape is:
```text
batch, time, views, channels, height, width
```
With default training arguments this becomes:
```text
batch, 5, 4, 3, 512, 640
```
### 2. Frozen Mirai Image Encoder
`FrozenMiraiImageEncoder` loads the Mirai snapshot from:
```text
data/csaw_cc/models/mgh_mammo_MIRAI_Base_May20_2019.p
```
The encoder is frozen during VMRA-MaR training. It returns:
- a per-image hidden vector used for exam-level fusion,
- activation feature maps used for asymmetry scoring.
DICOM images are resized, expanded to three channels, and normalized with the
Mirai statistics configured in `data.py`.
### 3. Mirai Multi-View Exam Encoder
`MiraiExamEncoder` fuses the four image-level Mirai hidden vectors into one
embedding per complete exam. The code can load an official transformer snapshot
from:
```text
assets/snapshots/mgh_mammo_cancer_MIRAI_Transformer_Jan13_2020.p
```
If that file is absent, the code constructs the formal Mirai transformer
architecture from the vendored `onconet` package and initializes it from
scratch.
The current implementation requires complete four-view exams. Partial exams are
detected and rejected before transformer fusion.
### 4. Spatial And Longitudinal Asymmetry
`SpatialAsymmetryDetector` compares left and right feature maps for CC and MLO
views. Right-side feature maps are horizontally flipped before comparison. The
module produces:
- normalized exam-level asymmetry scores,
- dominant asymmetry coordinates,
- validity masks for exams where a left-right pair is available.
`LongitudinalAsymmetryTracker` then increases the contribution of asymmetry
signals that persist across adjacent exams within a configurable coordinate
window. The final scalar feature is named `r_aa` in the model output.
### 5. VMamba-Backed VMRNN Temporal Encoder
`VMRNNEncoder` maps each exam embedding into a token grid and runs a multi-scale
recurrent encoder-decoder over time. The implementation uses:
- patch merging on the down path,
- patch expanding on the up path,
- recurrent stage cells backed by VMamba VSS blocks,
- optional reconstruction projections for diagnostic output.
The VMRNN path requires an available selective-scan backend. The runtime checks
for either:
- `mamba_ssm.ops.selective_scan_interface.selective_scan_fn`, or
- `selective_scan.selective_scan_fn`.
If neither backend is importable, model construction fails because the formal
VMRNN path is intentionally strict.
### 6. Additive Hazard Output Head
The final patient representation concatenates:
- the temporal history embedding from VMRNN,
- the scalar longitudinal asymmetry feature `r_aa`.
`AdditiveHazardLayer` maps this combined vector to five cumulative risk logits.
The training loop applies sigmoid activation for reported probabilities and uses
masked binary cross entropy on valid follow-up horizons.
## Data Requirements
The default data layout is defined in `src/vmra_mar/paths.py`:
```text
data/
`-- csaw_cc/
    |-- metadata/
    |   `-- csaw_cc.csv
    |-- dicom/
    |   `-- *.dcm
    `-- models/
        `-- mgh_mammo_MIRAI_Base_May20_2019.p
```
The CSV reader expects at least these columns:
- `anon_patientid`
- `exam_year`
- `x_case`
- `imagelaterality`
- `viewposition`
- `anon_filename`
The DICOM filenames referenced by `anon_filename` are resolved relative to
`--image-root`.
### Target Construction
The data pipeline creates five-year targets with censoring masks:
- Case patients use the terminal exam as the event endpoint. If the event falls
  outside the five-year horizon, that patient is skipped.
- Control patients receive all-zero targets and are masked only through observed
  follow-up years.
- Patients with no usable follow-up are skipped.
- When `require_local=True`, patients without any locally available DICOM image
  in their history are skipped.
The train, validation, and test splits are deterministic. Cases and controls are
split separately with stable hashing, then recombined. Default ratios are:
```text
train: 70%
validation: 15%
test: 15%
```
## Installation
Create an environment with Python 3.10 or newer:
```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
```
For the VMamba-backed VMRNN path, install the optional selective-scan
dependencies in an environment compatible with your CUDA and PyTorch versions:
```bash
python -m pip install -e ".[vmamba]"
```
The optional dependency group declares `causal-conv1d` and `mamba-ssm`.
Depending on the cluster or workstation, these packages may need prebuilt wheels,
a matching CUDA toolkit, or a container environment.
## Running Training
The main entry point is:
```bash
python -m vmra_mar.train
```
An explicit single-process run looks like:
```bash
python -m vmra_mar.train \
  --data data/csaw_cc/metadata/csaw_cc.csv \
  --image-root data/csaw_cc/dicom \
  --snapshot-path data/csaw_cc/models/mgh_mammo_MIRAI_Base_May20_2019.p \
  --output-dir artifacts/train \
  --epochs 30 \
  --batch-size 4 \
  --eval-batch-size 4 \
  --learning-rate 1e-3 \
  --min-learning-rate 1e-5
```
For distributed training, launch through `torchrun`:
```bash
PYTHONPATH=src torchrun --nproc_per_node=8 -m vmra_mar.train \
  --output-dir artifacts/train_8xv100 \
  --epochs 20 \
  --batch-size 2 \
  --eval-batch-size 2 \
  --gradient-accumulation-steps 2 \
  --num-workers 8 \
  --persistent-workers \
  --precision amp_fp16 \
  --learning-rate 3e-4 \
  --min-learning-rate 3e-5 \
  --warmup-epochs 1 \
  --clip-grad-norm 1.0
```
The same configuration is captured in:
```bash
bash scripts/train_8xv100.sh
```
## Useful Training Arguments
Data and assets:
- `--data`: path to CSAW-CC metadata CSV.
- `--image-root`: directory containing DICOM files.
- `--snapshot-path`: frozen Mirai image encoder snapshot.
- `--transformer-snapshot-path`: optional Mirai transformer snapshot.
- `--mirai-package-root`: directory containing the vendored `onconet` package.
- `--output-dir`: destination for checkpoints and metrics.
- `--max-patients`: optional cap for fast debugging.
Optimization:
- `--epochs`: total training epochs.
- `--batch-size`: per-process training batch size.
- `--eval-batch-size`: validation and test batch size.
- `--learning-rate`: peak AdamW learning rate.
- `--min-learning-rate`: minimum cosine scheduler learning rate.
- `--weight-decay`: AdamW weight decay.
- `--warmup-epochs`: linear warmup duration.
- `--gradient-accumulation-steps`: number of batches per optimizer step.
- `--clip-grad-norm`: gradient clipping threshold.
- `--early-stopping-patience`: optional validation-loss patience.
Runtime:
- `--precision`: one of `auto`, `fp32`, `amp_fp16`, or `amp_bf16`.
- `--backend`: distributed backend, either `auto`, `nccl`, or `gloo`.
- `--compile-model`: enable `torch.compile` when available.
- `--resume`: load a previous checkpoint.
- `--checkpoint-every`: checkpoint interval in epochs.
Model:
- `--image-height` and `--image-width`: resized image shape.
- `--exam-hidden-dim`: Mirai exam transformer hidden size.
- `--vmrnn-hidden-dim`: VMRNN hidden size.
- `--exam-dropout`: dropout in the exam and temporal encoders.
- `--vmrnn-vss-backend`: VMamba backend selection.
- `--vmamba-d-state`: VMamba state size.
- `--vmamba-drop-path`: VMamba drop-path rate.
- `--vmrnn-released-weights-path`: optional released VMRNN checkpoint.
## Training Outputs
The training loop writes outputs under `--output-dir`:
- `checkpoint_epoch_<N>.pt`: periodic training checkpoint.
- `best_model.pt`: checkpoint with the best validation loss.
- `model.pt`: final checkpoint from the last completed epoch.
- `metrics.json`: validation and test metrics from the selected evaluation
  checkpoint.
`metrics.json` includes:
- training and validation loss history,
- year-specific AUC for years 1 through 5,
- sample sizes and case counts per year,
- concordance index,
- decile recall,
- train, validation, and test patient counts,
- effective batch size,
- VMRNN and VMamba backend information,
- whether Mirai-compatible metric code was loaded from the vendored runtime or
  computed by the local fallback.
## Evaluation Details
The model trains with weighted masked binary cross entropy. When
`--pos-weight-mode auto` is active, positive-class weights are computed from the
training split per follow-up year and clipped by `--pos-weight-max`.
Evaluation tries to use Mirai-compatible metric utilities from
`vendor/mirai/onconet/learn/utils.py`. If that import or execution fails, the
code falls back to local implementations for:
- year-specific ROC AUC,
- concordance index,
- decile recall.
Censoring distributions are estimated from the training split with
`lifelines.KaplanMeierFitter` when `lifelines` is available.
## Programmatic Use
The package exposes the primary dataset and model interfaces:
```python
from vmra_mar import DatasetBundle, MammogramSequenceDataset, VMRAMaRModel, load_dataset_bundle
bundle = load_dataset_bundle(
    "data/csaw_cc/metadata/csaw_cc.csv",
    image_root="data/csaw_cc/dicom",
    require_local=True,
)
model = VMRAMaRModel(
    snapshot_path="data/csaw_cc/models/mgh_mammo_MIRAI_Base_May20_2019.p",
)
```
For full training behavior, prefer `vmra_mar.train.train_model` or the command
line entry point because it also handles distributed setup, mixed precision,
checkpointing, evaluation, and metric export.
