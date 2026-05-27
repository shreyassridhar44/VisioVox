# training/train.py  -  VisioVox v4
# SI-SDR + PIT + Contrastive loss, SpecAugment, differential LR
# Tuned for RTX A5000 24GB with 500k mixtures

import os
import sys
import random
import logging
import numpy as np
import pandas as pd
from pathlib import Path

import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

sys.path.append(str(Path(__file__).resolve().parent.parent))
from models.fusion_model import VisioVox
from training.dataset_loader import VisioVoxDataset

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("training_log.txt", mode="w", encoding="utf-8"),
    ],
)
logger = logging.getLogger("VisioVox")

# ── Config ────────────────────────────────────────────────────────────────────
PROJECT_DIR    = Path("E:/visiovox")
METADATA_CSV   = Path("D:/visiovox_data/devA_training_metadata.csv")
CHECKPOINT_DIR = Path("E:/visiovox/checkpoints")
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

EPOCHS         = 200
BATCH_SIZE     = 32       # A5000 24GB handles this comfortably
GRAD_ACCUM     = 1        # no accumulation needed at batch 32
LR             = 3e-4
WEIGHT_DECAY   = 1e-3
NUM_WORKERS    = 8        # Xeon W-2295 has 18 cores
VAL_SPLIT      = 0.10
SAVE_EVERY     = 5

SISDR_W        = 0.5
MSE_W          = 0.3
CONTRASTIVE_W  = 2.0

VISUAL_DROPOUT = 0.2      # probability of dropping visual during training

# ── Device ────────────────────────────────────────────────────────────────────
if torch.cuda.is_available():
    device = torch.device("cuda")
    torch.backends.cudnn.benchmark = True
    logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
    logger.info(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
else:
    device = torch.device("cpu")
    logger.warning("CUDA not available - this will be extremely slow")

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)


# ── Losses ────────────────────────────────────────────────────────────────────

def si_sdr_loss(est, tgt, eps=1e-8):
    tgt    = tgt - tgt.mean(-1, keepdim=True)
    est    = est - est.mean(-1, keepdim=True)
    dot    = (tgt * est).sum(-1, keepdim=True)
    s_tgt  = dot / (tgt.pow(2).sum(-1, keepdim=True) + eps) * tgt
    e_nse  = est - s_tgt
    sdr    = 10 * torch.log10(s_tgt.pow(2).sum(-1) / (e_nse.pow(2).sum(-1) + eps) + eps)
    return -sdr.mean()


def reconstruction_loss_pit(mask_a, mask_b, mixed, tgt_a, tgt_b):
    sep_a = mask_a * mixed
    sep_b = mask_b * mixed

    def score(sa, sb, ta, tb):
        fa = sa.view(sa.size(0), -1); ta_f = ta.view(ta.size(0), -1)
        fb = sb.view(sb.size(0), -1); tb_f = tb.view(tb.size(0), -1)
        sdr = (si_sdr_loss(fa, ta_f) + si_sdr_loss(fb, tb_f)) / 2
        mse = (F.mse_loss(sa, ta) + F.mse_loss(sb, tb)) / 2
        return SISDR_W * sdr + MSE_W * mse

    l1 = score(sep_a, sep_b, tgt_a, tgt_b)
    l2 = score(sep_a, sep_b, tgt_b, tgt_a)
    return torch.min(l1, l2), sep_a, sep_b


def contrastive_loss(mask_a, mask_b, v_emb_a, v_emb_b):
    fa = mask_a.view(mask_a.size(0), -1)
    fb = mask_b.view(mask_b.size(0), -1)
    cos_masks  = F.cosine_similarity(fa, fb, dim=-1)
    hard_loss  = F.relu(cos_masks - 0.5).pow(2).mean()
    cos_visual = F.cosine_similarity(v_emb_a, v_emb_b, dim=-1).mean()
    total      = cos_masks.mean() + 2.0 * hard_loss + 0.5 * cos_visual
    return total, cos_masks.mean()


# ── Epoch runner ──────────────────────────────────────────────────────────────

def run_epoch(model, loader, optimizer, epoch, is_train=True):
    model.train() if is_train else model.eval()
    warmup  = min(epoch / 10.0, 1.0)
    c_w     = CONTRASTIVE_W * warmup

    tot_loss = tot_sdr = tot_cos = 0.0
    n = 0

    optimizer.zero_grad()
    ctx = torch.enable_grad() if is_train else torch.no_grad()
    with ctx:
        for i, batch in enumerate(tqdm(loader, desc="TRN" if is_train else "VAL", leave=False)):
            mixed, tgt, t_lips, itr, i_lips = [b.to(device) for b in batch]

            do_drop = is_train and random.random() < VISUAL_DROPOUT
            mask_a, v_emb_a = model(mixed, t_lips, drop_visual=False)
            mask_b, v_emb_b = model(mixed, i_lips, drop_visual=do_drop)

            recon, sep_a, sep_b = reconstruction_loss_pit(mask_a, mask_b, mixed, tgt, itr)
            c_loss, cos_val     = contrastive_loss(mask_a, mask_b, v_emb_a, v_emb_b)
            loss = recon + c_w * c_loss

            if is_train:
                (loss / GRAD_ACCUM).backward()
                if (i + 1) % GRAD_ACCUM == 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                    optimizer.step()
                    optimizer.zero_grad()

            with torch.no_grad():
                sdr_val = -si_sdr_loss(
                    sep_a.view(sep_a.size(0), -1),
                    tgt.view(tgt.size(0), -1)
                ).item()

            tot_loss += loss.item()
            tot_sdr  += sdr_val
            tot_cos  += cos_val.item()
            n        += 1

        if is_train and n % GRAD_ACCUM != 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            optimizer.zero_grad()

    return tot_loss / max(n, 1), tot_sdr / max(n, 1), tot_cos / max(n, 1)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    logger.info("=" * 65)
    logger.info("VisioVox v4 - A5000 24GB - 500k Mixtures")
    logger.info("=" * 65)

    if not METADATA_CSV.exists():
        logger.error(f"Metadata not found: {METADATA_CSV}")
        logger.error("Run preprocess_devA.py then create_mixtures_devA.py first.")
        sys.exit(1)

    full_meta = pd.read_csv(METADATA_CSV)
    logger.info(f"Total mixtures: {len(full_meta)}")

    val_n  = int(len(full_meta) * VAL_SPLIT)
    trn_n  = len(full_meta) - val_n
    logger.info(f"Train: {trn_n} | Val: {val_n}")

    indices = list(range(len(full_meta)))
    random.shuffle(indices)
    trn_idx = indices[:trn_n]
    val_idx = indices[trn_n:]

    full_ds     = VisioVoxDataset(str(METADATA_CSV), augment=False)
    trn_ds_aug  = VisioVoxDataset(str(METADATA_CSV), augment=True)
    trn_ds      = Subset(trn_ds_aug, trn_idx)
    val_ds      = Subset(full_ds,    val_idx)

    trn_loader = DataLoader(trn_ds, batch_size=BATCH_SIZE, shuffle=True,
                            num_workers=NUM_WORKERS, pin_memory=True,
                            persistent_workers=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False,
                            num_workers=NUM_WORKERS, pin_memory=True,
                            persistent_workers=True)

    model  = VisioVox().to(device)
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Parameters: {params:,}")

    visual_params = (list(model.visual_encoder.parameters()) +
                     list(model.visual_proj.parameters()))
    other_params  = (list(model.audio_unet.parameters()) +
                     list(model.audio_proj.parameters()) +
                     list(model.fusion_refine.parameters()))

    optimizer = optim.AdamW([
        {"params": visual_params, "lr": LR * 3.0},
        {"params": other_params,  "lr": LR},
    ], weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=LR / 100)

    best_sdr   = -999.0
    best_epoch = 0

    logger.info(f"Batch={BATCH_SIZE} | LR={LR} | Epochs={EPOCHS} | Workers={NUM_WORKERS}")
    logger.info(f"Targets: val_cos<0.70 by ep15 | val_sdr>0dB by ep40 | val_sdr>7dB by ep150")
    logger.info("=" * 65)

    for epoch in range(1, EPOCHS + 1):
        trn_loss, trn_sdr, trn_cos = run_epoch(model, trn_loader, optimizer, epoch, is_train=True)
        val_loss, val_sdr, val_cos = run_epoch(model, val_loader, optimizer, epoch, is_train=False)
        scheduler.step()
        lr_now = scheduler.get_last_lr()[0]
        gap    = val_loss / max(trn_loss, 1e-8)

        logger.info(
            f"Ep {epoch:03d}/{EPOCHS} | "
            f"trn loss={trn_loss:.4f} sdr={trn_sdr:.2f}dB cos={trn_cos:.4f} | "
            f"val loss={val_loss:.4f} sdr={val_sdr:.2f}dB cos={val_cos:.4f} | "
            f"gap={gap:.2f}x | lr={lr_now:.2e}"
        )

        if val_sdr > best_sdr:
            best_sdr   = val_sdr
            best_epoch = epoch
            p = CHECKPOINT_DIR / "visiovox_devA_best.pth"
            torch.save({
                "epoch":            epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state":  optimizer.state_dict(),
                "val_sisdr":        val_sdr,
                "val_cos":          val_cos,
            }, p)
            logger.info(f"  ** Best sdr={val_sdr:.2f}dB -> {p.name}")

        if epoch % SAVE_EVERY == 0:
            p = CHECKPOINT_DIR / f"visiovox_devA_ep{epoch:03d}.pth"
            torch.save({
                "epoch":            epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state":  optimizer.state_dict(),
                "val_sisdr":        val_sdr,
                "val_cos":          val_cos,
            }, p)
            logger.info(f"  Checkpoint saved: {p.name}")

        if epoch >= 10 and gap > 2.0:
            logger.warning(f"OVERFIT ALERT ep{epoch}: val/trn gap={gap:.2f}x > 2.0")
        if epoch == 15 and val_cos > 0.80:
            logger.warning(f"COLLAPSE ALERT ep{epoch}: val_cos={val_cos:.4f} > 0.80 -- increase CONTRASTIVE_W to 3.0")

    logger.info("=" * 65)
    logger.info(f"Done. Best val_sdr={best_sdr:.2f}dB at epoch {best_epoch}")


if __name__ == "__main__":
    main()