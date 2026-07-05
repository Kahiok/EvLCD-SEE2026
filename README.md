# EvLCD: Event-Guided WDR Image Restoration

Official implementation of **EvLCD**, submitted to the [SEE-600K Challenge 2026](https://github.com/yunfanLu/SEE) (ECCV Workshop).

## Method

EvLCD restores wide dynamic range (WDR) images from event-camera sequences under challenging illumination. Key components:

- **LCDE module** (adapted from [LCDPNet](https://hywang99.github.io/lcdpnet/)): LCD pyramid → guided exposure mask → DualIllumNet
- **Multi-scale event encoder**: encodes 64-ch voxel-grid events at 3 spatial scales
- **Brightness-prompt conditioning**: scalar mean luminance → 256-dim FiLM vector
- **MDTA decoder** (Restormer-style transposed attention): 21K-param refinement block

Total parameters: **15.85M**

## Requirements

Create a conda environment and install dependencies:

```bash
conda create -n evlcd python=3.10 -y
conda activate evlcd

# Install PyTorch with CUDA (adjust cuda version as needed)
conda install pytorch torchvision pytorch-cuda=12.4 -c pytorch -c nvidia -y

pip install -r requirements.txt
```

Tested with PyTorch 2.6.0, CUDA 12.4.

## Checkpoints

Download from Google Drive: https://drive.google.com/drive/folders/1HxCKfv87fEP9Q9bHBmSfntYaYRcdF6Fo?usp=sharing

Place under `checkpoints/`:

| File | Description | Used for |
|------|-------------|----------|
| `EvLCD_base_ep071.pth.tar` | Trained on SEE-600K, epoch 71 | Fine-tuning starting point |
| `EvLCD_finetune_ep019.pth.tar` | Task-weighted fine-tune, epoch 19 | **Inference (submitted result)** |

## Inference

Edit `configs/EvLCD_SEE_eval_tta.yaml` to set `DATASET.root` to your eval dataset path.

```bash
# Step 1: TTA inference (4-flip ensemble); adjust GPU IDs as needed
# sample_step: 10 in the YAML selects every 10th frame (matches the Codabench eval protocol)
# mean_prompt_manifest.json must reside inside DATASET.root; read automatically when eval_phase: true
CUDA_VISIBLE_DEVICES=0,1,2,3 PYTHONPATH=. python see/main.py \
  --yaml_file=configs/EvLCD_SEE_eval_tta.yaml \
  --log_dir=logs/eval \
  --RESUME_PATH=checkpoints/EvLCD_finetune_ep019.pth.tar \
  --TEST_ONLY=True --VISUALIZE=True --VAL_BATCH_SIZE=1

# Step 2: Brightness alignment (requires mean_prompt_manifest.json from eval dataset)
python brightness_align.py \
  --vis_dir logs/eval/vis \
  --out_dir logs/eval/vis_ba \
  --manifest /path/to/DVS346-eval-mean-prompt2/mean_prompt_manifest.json \
  --method shift

# Step 3: Collect and zip submission
python codabench/collect_eval_phase_pred.py \
  logs/eval/vis_ba submission_dir
cd submission_dir && zip -r ../submission.zip . -x "*.DS_Store"
```

## Training

### 1. Base training (from scratch)

Edit `configs/EvLCD_SEE_train.yaml` to set `DATASET.root` to your SEE-600K path.

```bash
# 4 GPUs recommended; adjust CUDA_VISIBLE_DEVICES as needed
CUDA_VISIBLE_DEVICES=0,1,2,3 PYTHONPATH=. python see/main.py \
  --yaml_file=configs/EvLCD_SEE_train.yaml \
  --log_dir=logs/EvLCD_train
```

Hardware: 4 × RTX 3090, ~23 hours (80 epochs).

### 2. Task-weighted fine-tuning

Edit `configs/EvLCD_SEE_finetune_taskw.yaml`:
- Set `DATASET.root` to your SEE-600K path
- Set `RESUME.PATH` to the base checkpoint (default: `checkpoints/EvLCD_base_ep071.pth.tar`)

```bash
# 2 GPUs recommended; adjust CUDA_VISIBLE_DEVICES as needed
CUDA_VISIBLE_DEVICES=0,1 PYTHONPATH=. python see/main.py \
  --yaml_file=configs/EvLCD_SEE_finetune_taskw.yaml \
  --log_dir=logs/EvLCD_finetune
```

Hardware: 2 × RTX 3090, ~12.5 hours (20 epochs).  
Loss weights: low-normal ×2.0, high-normal ×2.0, normal-normal ×0.5.

## Config Overview

| Config | Purpose | Epochs | LR |
|--------|---------|--------|----|
| `EvLCD_SEE_train.yaml` | Train from scratch | 80 | 1e-4 |
| `EvLCD_SEE_finetune_taskw.yaml` | Task-weighted fine-tune | 20 | 3e-5 |
| `EvLCD_SEE_eval_tta.yaml` | Inference with TTA | — | — |

## Acknowledgements

- [LCDPNet](https://hywang99.github.io/lcdpnet/) — Local Color Distribution Embedded module
- [Restormer](https://github.com/swz30/Restormer) — MDTA block design
- [SEE-600K](https://github.com/yunfanLu/SEE) — dataset and evaluation framework

## Citation

If you find this work useful, please cite:

```bibtex
@inproceedings{lin2026evlcd,
  title     = {EvLCD: Event-Guided WDR Image Restoration with Brightness-Prompt Conditioning},
  author    = {Lin, Chia-Yu and Tsai, Yun-Tze and Liu, Shao-Kai and Jian, Rong-Lin and Lee, Chia-Ming and Hsu, Chih-Chung},
  booktitle = {Proceedings of the ECCV Workshop on SEE-600K Challenge},
  year      = {2026},
  note      = {to appear}
}
```
