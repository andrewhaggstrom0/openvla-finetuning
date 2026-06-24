import os
import gc
import glob
import pickle
import torch
import numpy as np
from PIL import Image as PILImage
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from transformers import (
    AutoModelForVision2Seq,
    AutoProcessor,
    BitsAndBytesConfig,
    get_cosine_schedule_with_warmup,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, PeftModel

# ── Config ──────────────────────────────────────────────────────────────
DATA_PATH    = os.path.expanduser("~/openvla_project/data/bridge_500ep.pkl")
SAVE_DIR     = os.path.expanduser("~/openvla_project/checkpoints/full_run")
LOG_PATH     = os.path.expanduser("~/openvla_project/logs/train.log")
MODEL_ID     = "openvla/openvla-7b"
LR           = 5e-4
GRAD_ACCUM   = 8
MAX_STEPS    = 3000
WARMUP_STEPS = 200
SAVE_EVERY   = 250
LOG_EVERY    = 10
LORA_RANK    = 32
LORA_ALPHA   = 64
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# ── Logging ─────────────────────────────────────────────────────────────
os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)
os.makedirs(SAVE_DIR, exist_ok=True)

def log(msg):
    print(msg, flush=True)
    with open(LOG_PATH, "a") as f:
        f.write(msg + "\n")

# ── Checkpoint helpers ───────────────────────────────────────────────────
def find_latest_checkpoint(save_dir):
    """Return (path, step) of the most recent checkpoint, or (None, 0)."""
    checkpoints = glob.glob(os.path.join(save_dir, "step_*"))
    if not checkpoints:
        return None, 0
    # Extract step numbers and find the highest
    steps = []
    for ckpt in checkpoints:
        try:
            step = int(os.path.basename(ckpt).split("_")[1])
            steps.append((step, ckpt))
        except (IndexError, ValueError):
            continue
    if not steps:
        return None, 0
    latest_step, latest_path = max(steps, key=lambda x: x[0])
    return latest_path, latest_step

def save_checkpoint(model, optimizer, scheduler, step, save_dir):
    """Save LoRA weights + optimizer/scheduler state."""
    ckpt_dir = os.path.join(save_dir, f"step_{step}")
    os.makedirs(ckpt_dir, exist_ok=True)
    # Save LoRA adapter weights
    model.save_pretrained(ckpt_dir)
    # Save optimizer and scheduler state
    torch.save({
        'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict(),
        'step': step,
    }, os.path.join(ckpt_dir, "training_state.pt"))
    log(f"Checkpoint saved: {ckpt_dir}")

def load_checkpoint(model, optimizer, scheduler, ckpt_path):
    """Load optimizer/scheduler state and return the step to resume from."""
    state_path = os.path.join(ckpt_path, "training_state.pt")
    if not os.path.exists(state_path):
        log(f"No training_state.pt found in {ckpt_path}, starting fresh")
        return 0
    state = torch.load(state_path, map_location="cpu")
    optimizer.load_state_dict(state['optimizer'])
    scheduler.load_state_dict(state['scheduler'])
    step = state['step']
    log(f"Resumed from step {step}")
    return step

# ── Dataset ─────────────────────────────────────────────────────────────
class BridgeDataset(Dataset):
    def __init__(self, samples, processor):
        self.samples = samples
        self.processor = processor

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        img = PILImage.fromarray(sample['image'])
        action_token_ids = sample['action_tokens']
        action_text = "".join([
            self.processor.tokenizer.convert_ids_to_tokens(int(t))
            for t in action_token_ids
        ])
        full_text = sample['instruction'] + " " + action_text
        inputs = self.processor(full_text, img)
        input_ids = inputs['input_ids'].squeeze()
        labels = torch.full_like(input_ids, -100)
        labels[-7:] = input_ids[-7:]
        return {
            'input_ids': input_ids,
            'attention_mask': inputs['attention_mask'].squeeze(),
            'pixel_values': inputs['pixel_values'].squeeze(),
            'labels': labels,
        }

def collate_fn(batch, pad_token_id):
    input_ids = pad_sequence(
        [b['input_ids'] for b in batch],
        batch_first=True, padding_value=pad_token_id or 0
    )
    attention_mask = pad_sequence(
        [b['attention_mask'] for b in batch],
        batch_first=True, padding_value=0
    )
    labels = pad_sequence(
        [b['labels'] for b in batch],
        batch_first=True, padding_value=-100
    )
    pixel_values = torch.stack([b['pixel_values'] for b in batch])
    return {
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'pixel_values': pixel_values,
        'labels': labels,
    }

# ── Main ────────────────────────────────────────────────────────────────
def main():
    log("Loading processor...")
    processor = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)

    log("Loading model in 4-bit...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    model = AutoModelForVision2Seq.from_pretrained(
        MODEL_ID,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
    )
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)

    # Check for existing checkpoint to resume from
    ckpt_path, start_step = find_latest_checkpoint(SAVE_DIR)

    if ckpt_path:
        log(f"Found checkpoint at step {start_step}, loading LoRA weights...")
        model = PeftModel.from_pretrained(model, ckpt_path, is_trainable=True)
    else:
        log("No checkpoint found, initializing fresh LoRA adapters...")
        model = get_peft_model(model, LoraConfig(
            r=LORA_RANK, lora_alpha=LORA_ALPHA,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            lora_dropout=0.05, bias="none", task_type="CAUSAL_LM",
        ))

    model.print_trainable_parameters()

    log("Loading dataset...")
    with open(DATA_PATH, "rb") as f:
        all_samples = pickle.load(f)

    n_train = int(len(all_samples) * 0.9)
    train_dataset = BridgeDataset(all_samples[:n_train], processor)
    val_dataset   = BridgeDataset(all_samples[n_train:], processor)
    log(f"Train: {len(train_dataset)} steps | Val: {len(val_dataset)} steps")

    pad_id = processor.tokenizer.pad_token_id
    train_loader = DataLoader(
        train_dataset, batch_size=1, shuffle=True,
        num_workers=0, pin_memory=False,
        collate_fn=lambda b: collate_fn(b, pad_id)
    )

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=LR, weight_decay=0.01,
    )
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, num_warmup_steps=WARMUP_STEPS, num_training_steps=MAX_STEPS
    )

    # Resume optimizer/scheduler state if checkpoint exists
    if ckpt_path:
        start_step = load_checkpoint(model, optimizer, scheduler, ckpt_path)
    else:
        start_step = 0

    log(f"Starting training from step {start_step}/{MAX_STEPS}...")
    log(f"{'Step':>6} {'Loss':>8} {'LR':>10} {'VRAM':>8}")
    log("-" * 40)

    model.train()
    optimizer.zero_grad()
    step = start_step
    loader_iter = iter(train_loader)

    # Skip ahead in the dataloader if resuming mid-epoch
    if start_step > 0:
        skip = start_step % len(train_loader)
        log(f"Skipping {skip} batches to resume position...")
        for _ in range(skip):
            try:
                next(loader_iter)
            except StopIteration:
                loader_iter = iter(train_loader)

    while step < MAX_STEPS:
        try:
            batch = next(loader_iter)
        except StopIteration:
            loader_iter = iter(train_loader)
            batch = next(loader_iter)

        input_ids      = batch['input_ids'].to("cuda")
        attention_mask = batch['attention_mask'].to("cuda")
        pixel_values   = batch['pixel_values'].to("cuda")
        labels         = batch['labels'].to("cuda")

        with torch.autocast("cuda", dtype=torch.bfloat16):
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                pixel_values=pixel_values,
                labels=labels,
            )

        loss = outputs.loss / GRAD_ACCUM
        loss.backward()

        if (step + 1) % GRAD_ACCUM == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

        if step % LOG_EVERY == 0:
            vram = torch.cuda.memory_allocated() / 1e9
            lr   = optimizer.param_groups[0]['lr']
            log(f"{step:>6} {loss.item()*GRAD_ACCUM:>8.4f} {lr:>10.6f} {vram:>6.2f}GB")

        if step % SAVE_EVERY == 0 and step > 0:
            save_checkpoint(model, optimizer, scheduler, step, SAVE_DIR)

        step += 1

    # Final save
    save_checkpoint(model, optimizer, scheduler, step, SAVE_DIR)
    log("Training complete!")

if __name__ == "__main__":
    main()
