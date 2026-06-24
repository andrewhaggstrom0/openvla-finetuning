# QLoRA Fine-tuning of OpenVLA on Bridge V2

Fine-tuning the [OpenVLA](https://openvla.github.io/) 7B vision-language-action model on the [Bridge V2](https://rail-berkeley.github.io/bridgedata/) robot manipulation dataset using QLoRA (Quantized Low-Rank Adaptation). Trained on a single NVIDIA A100-SXM4-80GB GPU in ~25 minutes.

## Results

| Model | Token Accuracy | Mean L1 Error |
|---|---|---|
| Base OpenVLA (zero-shot) | 1.94% | 71.50 |
| Fine-tuned (step 1000, best) | **36.29%** | **0.142** |
| Fine-tuned (step 3000, final) | 33.54% | 0.183 |

**17x improvement in token accuracy, 391x reduction in L1 error** over the zero-shot baseline.

### Ablation Study

| Training Steps | Token Accuracy | Mean L1 Error |
|---|---|---|
| 0 (base) | 1.94% | 71.50 |
| 500 | 33.54% | 0.127 |
| 1,000 | **36.29%** | 0.142 |
| 1,500 | 31.92% | 0.164 |
| 2,000 | 32.48% | 0.149 |
| 2,500 | 34.01% | 0.193 |
| 3,000 | 33.54% | 0.183 |

Key finding: most learning occurs in the first 500 steps. Performance peaks at step 1,000 — early stopping is recommended for datasets of this size.

## Setup

### Requirements

- Python 3.9+
- CUDA-capable GPU (tested on A100-SXM4-80GB, works on RTX 4000 Ada with 20GB VRAM)

### Install dependencies

```bash
pip install --user torch torchvision --index-url https://download.pytorch.org/whl/cu128
pip install --user transformers==4.40.0 peft==0.11.0 bitsandbytes accelerate==0.30.0 "timm>=0.9.10,<1.0.0" pillow tensorflow-datasets
```

## Data Preparation

Download Bridge V2 shards from HuggingFace and preprocess into a cached dataset:

```python
from huggingface_hub import hf_hub_download
import os

save_dir = "/tmp/oxe_raw/bridge/1.0.0"
os.makedirs(save_dir, exist_ok=True)

for i in range(20):
    shard = f"{i:05d}"
    hf_hub_download(
        repo_id="ericonaldo/Bridge-V2",
        repo_type="dataset",
        filename=f"1.0.0/bridge_dataset-train.tfrecord-{shard}-of-01024",
        local_dir="/tmp/oxe_raw/bridge",
    )
```

Then preprocess (see the preprocessing cells in the notebook).

## Training

### Interactive (Jupyter)

Run the training cells in the notebook for a quick sanity check (~50 steps).

### Batch job (SLURM)

Edit `train_job.sh` with your partition and account, then:

```bash
sbatch train_job.sh
tail -f logs/train.log
```

Training runs for 3,000 steps with checkpoints saved every 250 steps. To resume from a checkpoint, simply resubmit — the script automatically detects and resumes from the latest checkpoint.

### Key hyperparameters

```python
LR           = 5e-4
GRAD_ACCUM   = 8          # effective batch size = 8
MAX_STEPS    = 3000
WARMUP_STEPS = 200
SAVE_EVERY   = 250
LORA_RANK    = 32
LORA_ALPHA   = 64
TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj"]
```

## Evaluation

```bash
python evaluate.py ~/openvla_project/checkpoints/full_run/step_1000 ~/openvla_project/logs/eval_results.txt
```

To run the full ablation study across all checkpoints simultaneously:

```bash
for step in 500 1000 1500 2000 2500 3000; do
    sbatch --job-name=eval_${step} \
           --partition=<your-partition> \
           --gres=gpu:1 \
           --wrap="python evaluate.py checkpoints/full_run/step_${step} logs/eval_results_step${step}.txt"
done
```

## Project Structure

```
openvla_project/
├── train.py              # QLoRA training script with checkpoint resume
├── evaluate.py           # Offline evaluation script
├── train_job.sh          # SLURM batch job script
├── eval_job.sh           # SLURM evaluation job script
├── data/
│   └── bridge_500ep.pkl  # Preprocessed dataset (not tracked by git)
├── checkpoints/          # LoRA adapter weights (not tracked by git)
│   └── full_run/
│       ├── step_250/
│       ├── step_500/
│       └── ...
└── logs/                 # Training and evaluation logs (not tracked by git)
```

## Model Architecture

OpenVLA uses a fused visual encoder (SigLIP + DINOv2) combined with a Llama 2 7B language model backbone. Actions are discretized into 256 bins per joint dimension, repurposing the 256 least-used tokens in the Llama vocabulary as action tokens. QLoRA adds trainable low-rank adapters to the attention projection matrices, training only 33.5M parameters (0.44% of 7.57B total).

## Hardware

Training was performed on the WashU SEAS research compute cluster (condo-cse5100 partition):

- GPU: NVIDIA A100-SXM4-80GB
- Peak VRAM usage: 5.78GB
- Training time: ~25 minutes (3,000 steps)
- Evaluation time: ~30 minutes per checkpoint (1,234 val steps)

## References

- [OpenVLA: An Open-Source Vision-Language-Action Model](https://openvla.github.io/)
- [BridgeData V2: Datasets for Robot Learning at Scale](https://rail-berkeley.github.io/bridgedata/)
- [QLoRA: Efficient Finetuning of Quantized LLMs](https://arxiv.org/abs/2305.14314)
- [Open X-Embodiment: Robotic Learning Datasets and RT-X Models](https://robotics-transformer-x.github.io/)

## Author

Andrew Haggstrom — Biomedical Engineering, Washington University in St. Louis  
[andrewhaggstrom0.github.io](https://andrewhaggstrom0.github.io)
