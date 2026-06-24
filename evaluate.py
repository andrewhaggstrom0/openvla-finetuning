import os
import sys
import pickle
import torch
import numpy as np
from PIL import Image as PILImage
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoModelForVision2Seq, AutoProcessor, BitsAndBytesConfig
from peft import PeftModel

# ── Config ───────────────────────────────────────────────────────────────
DATA_PATH  = os.path.expanduser("~/openvla_project/data/bridge_500ep.pkl")
MODEL_ID   = "openvla/openvla-7b"
N_BINS     = 256
VOCAB_SIZE = 32000
ACTION_TOKEN_START = VOCAB_SIZE - N_BINS

def decode_action_tokens(token_ids):
    bin_indices = token_ids - ACTION_TOKEN_START
    return (bin_indices / (N_BINS - 1)) * 2.0 - 1.0

class BridgeValDataset(Dataset):
    def __init__(self, samples, processor):
        self.samples = samples
        self.processor = processor

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        img = PILImage.fromarray(sample['image'])
        inputs = self.processor(sample['instruction'], img)
        return {
            'input_ids': inputs['input_ids'].squeeze(),
            'attention_mask': inputs['attention_mask'].squeeze(),
            'pixel_values': inputs['pixel_values'].squeeze(),
            'gt_action_tokens': torch.tensor(sample['action_tokens'], dtype=torch.long),
            'gt_action': torch.tensor(sample['action'], dtype=torch.float32),
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
    pixel_values = torch.stack([b['pixel_values'] for b in batch])
    gt_action_tokens = torch.stack([b['gt_action_tokens'] for b in batch])
    gt_action = torch.stack([b['gt_action'] for b in batch])
    return {
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'pixel_values': pixel_values,
        'gt_action_tokens': gt_action_tokens,
        'gt_action': gt_action,
    }

def main(ckpt_path, results_path):
    print(f"Evaluating checkpoint: {ckpt_path}")
    print("Loading processor...")
    processor = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)

    print("Loading model...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    base_model = AutoModelForVision2Seq.from_pretrained(
        MODEL_ID,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
    )

    if ckpt_path and os.path.exists(ckpt_path):
        print(f"Loading LoRA checkpoint from {ckpt_path}")
        model = PeftModel.from_pretrained(base_model, ckpt_path)
    else:
        print("No checkpoint found — evaluating base model (zero-shot)")
        model = base_model

    model.eval()

    print("Loading val dataset...")
    with open(DATA_PATH, "rb") as f:
        all_samples = pickle.load(f)
    n_train = int(len(all_samples) * 0.9)
    val_samples = all_samples[n_train:]
    print(f"Val set: {len(val_samples)} steps")

    val_dataset = BridgeValDataset(val_samples, processor)
    val_loader = DataLoader(
        val_dataset, batch_size=1, shuffle=False,
        num_workers=0, pin_memory=False,
        collate_fn=lambda b: collate_fn(b, processor.tokenizer.pad_token_id)
    )

    token_correct = 0
    token_total   = 0
    l1_errors     = []
    per_joint_l1  = [[] for _ in range(7)]

    print("Running evaluation...")
    print(f"{'Step':>6} {'Token Acc':>10} {'L1 Error':>10}")
    print("-" * 35)

    with torch.no_grad():
        for i, batch in enumerate(val_loader):
            input_ids      = batch['input_ids'].to("cuda")
            attention_mask = batch['attention_mask'].to("cuda")
            pixel_values   = batch['pixel_values'].to("cuda")
            gt_tokens      = batch['gt_action_tokens']
            gt_action      = batch['gt_action'].numpy()

            with torch.autocast("cuda", dtype=torch.bfloat16):
                generated = model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    pixel_values=pixel_values,
                    max_new_tokens=7,
                    do_sample=False,
                )

            pred_tokens  = generated[0, -7:].cpu().numpy()
            gt_tokens_np = gt_tokens[0].numpy()

            correct = (pred_tokens == gt_tokens_np).sum()
            token_correct += correct
            token_total   += 7

            pred_action     = decode_action_tokens(pred_tokens)
            gt_cont_clipped = np.clip(gt_action[0], -1.0, 1.0)
            step_l1         = np.abs(pred_action - gt_cont_clipped)
            l1_errors.append(step_l1.mean())

            for j in range(7):
                per_joint_l1[j].append(step_l1[j])

            if i % 100 == 0:
                running_acc = token_correct / token_total * 100
                running_l1  = np.mean(l1_errors)
                print(f"{i:>6} {running_acc:>9.1f}% {running_l1:>10.4f}")

    final_acc = token_correct / token_total * 100
    final_l1  = np.mean(l1_errors)
    joint_names = ["dx", "dy", "dz", "d_roll", "d_pitch", "d_yaw", "gripper"]

    print("\n" + "="*40)
    print("EVALUATION RESULTS")
    print("="*40)
    print(f"Checkpoint:         {ckpt_path}")
    print(f"Token accuracy:     {final_acc:.2f}%")
    print(f"Mean L1 error:      {final_l1:.4f}")
    print(f"\nPer-joint L1 error:")
    for j, name in enumerate(joint_names):
        print(f"  {name:>8}: {np.mean(per_joint_l1[j]):.4f}")

    os.makedirs(os.path.dirname(results_path), exist_ok=True)
    with open(results_path, "w") as f:
        f.write(f"checkpoint: {ckpt_path}\n")
        f.write(f"token_accuracy: {final_acc}\n")
        f.write(f"mean_l1: {final_l1}\n")
        f.write(f"per_joint_l1: {dict(zip(joint_names, [float(np.mean(per_joint_l1[j])) for j in range(7)]))}\n")
        f.write(f"n_samples: {len(val_samples)}\n")
    print(f"\nResults saved to {results_path}")

if __name__ == "__main__":
    ckpt  = sys.argv[1] if len(sys.argv) > 1 else ""
    rpath = sys.argv[2] if len(sys.argv) > 2 else os.path.expanduser("~/openvla_project/logs/eval_results.txt")
    main(ckpt_path=os.path.expanduser(ckpt) if ckpt else "", results_path=os.path.expanduser(rpath))
