# train_local.py  -  VisioVox  -  Training Script v4
# ============================================================
# KEY CHANGES FROM v3:
#
#   PROBLEM IN v3: Severe overfitting
#     - 22M param model trained on only 4500 samples
#     - val_loss 7.0 vs trn_loss 3.0 (2.3x gap)
#     - val_cos stuck at 0.97 despite trn_cos reaching 0.04
#     - val_sdr barely moved from -5.3 to -5.0 in 89 epochs
#
#   FIX 1 - 50,000 mixtures (was 5,000) = 10x more training data
#     45,000 train / 5,000 val split
#     This is the single most impactful change.
#
#   FIX 2 - Dropout in U-Net decoder (p=0.2)
#     Prevents memorisation of training mixtures.
#
#   FIX 3 - SpecAugment data augmentation
#     Randomly mask frequency bands and time steps on spectrograms.
#     Forces model to be robust to partial information.
#
#   FIX 4 - Stronger weight decay 1e-4 -> 1e-3
#     L2 regularisation on all weights.
#
#   FIX 5 - Reduced visual dim 256 -> 128
#     Smaller model = less overfitting risk.
#
#   FIX 6 - Lower learning rate 2e-4 -> 1e-4
#     More stable training with larger dataset.
#
#   FIX 7 - Gradient accumulation (accumulate 4 batches)
#     Effective batch size = 4 * 4 = 16 without extra VRAM.
#     Larger effective batch = more stable gradients.
#
# EXPECTED BEHAVIOUR WITH 50k MIXTURES:
#   val_loss should track trn_loss closely (gap < 1.5x)
#   val_cos should drop below 0.60 by epoch 20
#   val_sdr should reach 0dB by epoch 40, 3-5dB by epoch 80

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
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision.models import resnet18, ResNet18_Weights
from tqdm import tqdm

# -- Logging ------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("training_log.txt", mode="w", encoding="utf-8"),
    ],
)
logger = logging.getLogger("VisioVox")

# -- Config -------------------------------------------------------------------
PROJECT_DIR    = Path("C:/Users/samar/Downloads/visiovoxnew_code/visiovoxnew")
METADATA_CSV   = PROJECT_DIR / "data" / "devA_training_metadata.csv"
CHECKPOINT_DIR = PROJECT_DIR / "checkpoints"
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

EPOCHS         = 100
BATCH_SIZE     = 4
GRAD_ACCUM     = 4        # effective batch = 4 * 4 = 16
LR             = 1e-4     # lower LR for stability
WEIGHT_DECAY   = 1e-3     # stronger regularisation (was 1e-4)
NUM_WORKERS    = 0
VAL_SPLIT      = 0.10     # 5000 val, 45000 train
SAVE_EVERY     = 5

SISDR_W        = 0.5
MSE_W          = 0.3
CONTRASTIVE_W  = 2.0

NUM_FRAMES     = 25
VISUAL_DIM     = 128      # reduced from 256 to reduce overfitting

VISUAL_DROPOUT = 0.2

# SpecAugment params
FREQ_MASK_MAX  = 30       # max freq bins to mask
TIME_MASK_MAX  = 30       # max time steps to mask
N_FREQ_MASKS   = 2
N_TIME_MASKS   = 2

SAMPLE_RATE    = 16000
N_FFT          = 510
HOP_LENGTH     = 160
WIN_LENGTH     = 400
TARGET_SAMPLES = int(3.0 * SAMPLE_RATE)

# -- Device -------------------------------------------------------------------
if torch.cuda.is_available():
    device = torch.device("cuda")
    torch.backends.cudnn.benchmark = True
    logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
    logger.info(f"VRAM: {torch.cuda.get_device_properties(0).total_memory/1024**3:.1f} GB")
else:
    device = torch.device("cpu")
    logger.warning("CUDA not available - training on CPU will be very slow!")

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)


# =============================================================================
# SPECAUGMENT
# =============================================================================

def spec_augment(spec):
    """
    Apply SpecAugment to a spectrogram tensor [1, F, T].
    Randomly masks frequency bands and time steps.
    Only applied during training.
    """
    spec = spec.clone()
    _, F, T = spec.shape

    # Frequency masking
    for _ in range(N_FREQ_MASKS):
        f = random.randint(0, FREQ_MASK_MAX)
        f0 = random.randint(0, max(F - f, 1))
        spec[:, f0:f0+f, :] = 0.0

    # Time masking
    for _ in range(N_TIME_MASKS):
        t = random.randint(0, TIME_MASK_MAX)
        t0 = random.randint(0, max(T - t, 1))
        spec[:, :, t0:t0+t] = 0.0

    return spec


# =============================================================================
# MODEL
# =============================================================================

class TemporalVisualEncoder(nn.Module):
    """25 lip frames -> 128-dim embedding. LayerNorm output for scale stability."""
    def __init__(self, embedding_dim=VISUAL_DIM):
        super().__init__()
        self.frontend3d = nn.Sequential(
            nn.Conv3d(1, 64, kernel_size=(5, 7, 7),
                      stride=(1, 2, 2), padding=(2, 3, 3), bias=False),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1)),
        )
        resnet       = resnet18(weights=ResNet18_Weights.DEFAULT)
        self.layer1  = resnet.layer1
        self.layer2  = resnet.layer2
        self.layer3  = resnet.layer3
        self.layer4  = resnet.layer4
        self.avgpool = resnet.avgpool
        self.fc      = nn.Linear(resnet.fc.in_features, embedding_dim)
        self.norm    = nn.LayerNorm(embedding_dim)
        self.drop    = nn.Dropout(p=0.3)   # regularise visual encoder

    def forward(self, x):
        B, T, C, H, W = x.shape
        x = x.permute(0, 2, 1, 3, 4)
        x = self.frontend3d(x)
        x = x.mean(dim=2)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.drop(x)
        x = self.fc(x)
        return self.norm(x)   # [B, 128]


class AudioUNet(nn.Module):
    """U-Net with dropout in decoder for regularisation."""
    def __init__(self, dropout_p=0.2):
        super().__init__()
        self.enc1       = self._blk(1,   32)
        self.enc2       = self._blk(32,  64)
        self.enc3       = self._blk(64,  128)
        self.enc4       = self._blk(128, 256)
        self.bottleneck = self._blk(256, 512)
        self.pool       = nn.MaxPool2d(2)
        self.drop       = nn.Dropout2d(p=dropout_p)   # decoder dropout

        self.up4  = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.dec4 = self._blk(512, 256)
        self.up3  = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec3 = self._blk(256, 128)
        self.up2  = nn.ConvTranspose2d(128,  64, 2, stride=2)
        self.dec2 = self._blk(128,  64)
        self.up1  = nn.ConvTranspose2d( 64,  32, 2, stride=2)
        self.dec1 = self._blk( 64,  32)
        self.out  = nn.Sequential(nn.Conv2d(32, 1, 1), nn.Sigmoid())

    def _blk(self, ic, oc):
        return nn.Sequential(
            nn.Conv2d(ic, oc, 3, padding=1), nn.BatchNorm2d(oc), nn.ReLU(inplace=True),
            nn.Conv2d(oc, oc, 3, padding=1), nn.BatchNorm2d(oc), nn.ReLU(inplace=True),
        )

    @staticmethod
    def _pad(x, ref):
        dh = ref.size(2) - x.size(2)
        dw = ref.size(3) - x.size(3)
        return F.pad(x, [dw//2, dw-dw//2, dh//2, dh-dh//2])

    def encode(self, x):
        e1 = self.enc1(x);       p1 = self.pool(e1)
        e2 = self.enc2(p1);      p2 = self.pool(e2)
        e3 = self.enc3(p2);      p3 = self.pool(e3)
        e4 = self.enc4(p3);      p4 = self.pool(e4)
        b  = self.bottleneck(p4)
        return b, [e1, e2, e3, e4]

    def decode(self, x, skips):
        e1, e2, e3, e4 = skips
        x = self.drop(x)
        x = self.dec4(torch.cat([self._pad(self.up4(x), e4), e4], 1))
        x = self.drop(x)
        x = self.dec3(torch.cat([self._pad(self.up3(x), e3), e3], 1))
        x = self.drop(x)
        x = self.dec2(torch.cat([self._pad(self.up2(x), e2), e2], 1))
        x = self.dec1(torch.cat([self._pad(self.up1(x), e1), e1], 1))
        return self.out(x)


class VisioVox(nn.Module):
    """
    Additive visual-audio fusion with LayerNorm-normalised visual embedding.
    Visual dim 128, audio projected to 128, then added for equal contribution.
    """
    def __init__(self):
        super().__init__()
        self.visual_encoder = TemporalVisualEncoder(embedding_dim=VISUAL_DIM)
        self.audio_unet     = AudioUNet(dropout_p=0.2)
        self.audio_proj     = nn.Sequential(
            nn.Conv2d(512, VISUAL_DIM, 1),
            nn.BatchNorm2d(VISUAL_DIM),
            nn.ReLU(inplace=True)
        )
        self.visual_proj    = nn.Linear(VISUAL_DIM, VISUAL_DIM)
        self.fusion_refine  = nn.Sequential(
            nn.Conv2d(VISUAL_DIM, 512, 3, padding=1),
            nn.BatchNorm2d(512), nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.BatchNorm2d(512), nn.ReLU(inplace=True),
        )

    def forward(self, mixed_spec, lip_frames, drop_visual=False):
        v_emb = self.visual_encoder(lip_frames)           # [B, 128]
        if drop_visual:
            v_emb = torch.zeros_like(v_emb)

        bottleneck, skips = self.audio_unet.encode(mixed_spec)
        a_proj = self.audio_proj(bottleneck)              # [B, 128, F', T']
        v_proj = self.visual_proj(v_emb)                  # [B, 128]
        v_spat = v_proj.unsqueeze(-1).unsqueeze(-1).expand_as(a_proj)

        fused  = a_proj + v_spat                          # additive fusion
        fused  = self.fusion_refine(fused)                # [B, 512, F', T']
        mask   = self.audio_unet.decode(fused, skips)     # [B, 1, F, T]
        return mask, v_emb


# =============================================================================
# LOSSES
# =============================================================================

def si_sdr_loss(est, tgt, eps=1e-8):
    tgt = tgt - tgt.mean(-1, keepdim=True)
    est = est - est.mean(-1, keepdim=True)
    dot    = (tgt * est).sum(-1, keepdim=True)
    s_tgt  = dot / (tgt.pow(2).sum(-1, keepdim=True) + eps) * tgt
    e_noise= est - s_tgt
    sdr    = 10 * torch.log10(
        s_tgt.pow(2).sum(-1) / (e_noise.pow(2).sum(-1) + eps) + eps)
    return -sdr.mean()


def reconstruction_loss_pit(mask_a, mask_b, mixed, tgt_a, tgt_b):
    """Permutation-invariant: try both assignments, use the better one."""
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
    """Cosine similarity + hard margin + visual embedding penalty."""
    fa = mask_a.view(mask_a.size(0), -1)
    fb = mask_b.view(mask_b.size(0), -1)
    cos_masks  = F.cosine_similarity(fa, fb, dim=-1)
    hard_loss  = F.relu(cos_masks - 0.5).pow(2).mean()
    cos_visual = F.cosine_similarity(v_emb_a, v_emb_b, dim=-1).mean()
    total      = cos_masks.mean() + 2.0 * hard_loss + 0.5 * cos_visual
    return total, cos_masks.mean()


# =============================================================================
# DATASET
# =============================================================================

class VisioVoxDataset(Dataset):
    def __init__(self, csv_path, augment=False):
        self.meta    = pd.read_csv(csv_path)
        self.window  = torch.hann_window(WIN_LENGTH)
        self.augment = augment
        logger.info(f"Dataset loaded: {len(self.meta)} samples "
                    f"({'with' if augment else 'no'} augmentation)")

    def __len__(self):
        return len(self.meta)

    def _spec(self, path):
        import librosa
        w, _ = librosa.load(str(path), sr=SAMPLE_RATE, mono=True)
        if len(w) >= TARGET_SAMPLES:
            w = w[:TARGET_SAMPLES]
        else:
            w = np.pad(w, (0, TARGET_SAMPLES - len(w)))
        t = torch.tensor(w, dtype=torch.float32)
        s = torch.stft(t, n_fft=N_FFT, hop_length=HOP_LENGTH,
                       win_length=WIN_LENGTH, window=self.window,
                       return_complex=True)
        spec = torch.abs(s).unsqueeze(0)   # [1, 256, T]
        if self.augment:
            spec = spec_augment(spec)
        return spec

    def _lips(self, d):
        paths  = sorted(Path(d).glob("*.jpg"))
        frames = []
        for p in paths[:NUM_FRAMES]:
            img = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
            frames.append(
                cv2.resize(img, (112, 112)).astype(np.float32) / 255.0
                if img is not None else np.zeros((112, 112), np.float32)
            )
        while len(frames) < NUM_FRAMES:
            frames.append(np.zeros((112, 112), np.float32))
        return torch.tensor(np.stack(frames)).unsqueeze(1)   # [25, 1, 112, 112]

    def __getitem__(self, idx):
        r = self.meta.iloc[idx]
        try:
            return (self._spec(r["mixed_audio_path"]),
                    self._spec(r["target_audio_path"]),
                    self._lips(r["target_lips_dir"]),
                    self._spec(r["interfere_audio"]),
                    self._lips(r["interfere_lips"]))
        except Exception as e:
            logger.warning(f"Bad idx {idx}: {e}")
            ds = torch.zeros(1, 256, 301)
            dl = torch.zeros(NUM_FRAMES, 1, 112, 112)
            return ds, ds, dl, ds, dl


# =============================================================================
# TRAINING
# =============================================================================

def run_epoch(model, loader, optimizer, epoch, is_train=True):
    model.train() if is_train else model.eval()

    warmup = min(epoch / 10.0, 1.0)
    c_w    = CONTRASTIVE_W * warmup

    tot_loss = tot_sdr = tot_cos = 0.0
    n = 0

    optimizer.zero_grad()   # for gradient accumulation

    ctx = torch.enable_grad() if is_train else torch.no_grad()
    with ctx:
        for i, batch in enumerate(tqdm(loader, desc="TRN" if is_train else "VAL",
                                       leave=False)):
            mixed, tgt, t_lips, itr, i_lips = [b.to(device) for b in batch]

            do_drop = is_train and random.random() < VISUAL_DROPOUT
            mask_a, v_emb_a = model(mixed, t_lips, drop_visual=False)
            mask_b, v_emb_b = model(mixed, i_lips, drop_visual=do_drop)

            recon, sep_a, sep_b = reconstruction_loss_pit(
                mask_a, mask_b, mixed, tgt, itr)
            c_loss, cos_val = contrastive_loss(mask_a, mask_b, v_emb_a, v_emb_b)
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
                    tgt.view(tgt.size(0), -1)).item()

            tot_loss += loss.item()
            tot_sdr  += sdr_val
            tot_cos  += cos_val.item()
            n        += 1

        # flush remaining gradients
        if is_train and n % GRAD_ACCUM != 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            optimizer.zero_grad()

    return tot_loss/max(n,1), tot_sdr/max(n,1), tot_cos/max(n,1)


def main():
    logger.info("=" * 65)
    logger.info("VisioVox v4 - 50k Mixtures + Regularisation + SpecAugment")
    logger.info("=" * 65)

    # Training dataset with augmentation, val without
    full_meta = pd.read_csv(METADATA_CSV)
    logger.info(f"Total mixtures: {len(full_meta)}")

    val_n  = int(len(full_meta) * VAL_SPLIT)
    trn_n  = len(full_meta) - val_n
    logger.info(f"Train: {trn_n} | Val: {val_n}")

    # Split indices first, then create datasets with correct augment flag
    indices    = list(range(len(full_meta)))
    random.seed(42)
    random.shuffle(indices)
    trn_idx    = indices[:trn_n]
    val_idx    = indices[trn_n:]

    from torch.utils.data import Subset
    full_ds    = VisioVoxDataset(METADATA_CSV, augment=False)
    trn_ds_aug = VisioVoxDataset(METADATA_CSV, augment=True)

    trn_ds = Subset(trn_ds_aug, trn_idx)
    val_ds = Subset(full_ds,    val_idx)

    trn_loader = DataLoader(trn_ds, batch_size=BATCH_SIZE, shuffle=True,
                            num_workers=NUM_WORKERS, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False,
                            num_workers=NUM_WORKERS, pin_memory=True)

    model  = VisioVox().to(device)
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Parameters: {params:,}")

    # Differential LR: visual 3x higher to learn lips faster
    visual_params = (list(model.visual_encoder.parameters()) +
                     list(model.visual_proj.parameters()))
    other_params  = (list(model.audio_unet.parameters()) +
                     list(model.audio_proj.parameters()) +
                     list(model.fusion_refine.parameters()))

    optimizer = optim.AdamW([
        {"params": visual_params, "lr": LR * 3.0},
        {"params": other_params,  "lr": LR},
    ], weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=EPOCHS, eta_min=LR/100)

    best_sdr   = -999.0
    best_epoch = 0

    logger.info(f"Epochs={EPOCHS} | batch={BATCH_SIZE} | "
                f"effective_batch={BATCH_SIZE*GRAD_ACCUM} | lr={LR}")
    logger.info(f"Weight decay={WEIGHT_DECAY} | SpecAugment ON | Decoder dropout=0.2")
    logger.info(f"Contrastive weight={CONTRASTIVE_W} (warmed up over 10 epochs)")
    logger.info("Targets: val_cos<0.60 by ep20 | val_sdr>0dB by ep40 | "
                "val/trn loss gap < 1.5x throughout")
    logger.info("=" * 65)

    for epoch in range(1, EPOCHS + 1):
        trn_loss, trn_sdr, trn_cos = run_epoch(
            model, trn_loader, optimizer, epoch, is_train=True)
        val_loss, val_sdr, val_cos = run_epoch(
            model, val_loader, optimizer, epoch, is_train=False)
        scheduler.step()
        lr_now = scheduler.get_last_lr()[0]

        # Overfitting indicator
        gap = val_loss / max(trn_loss, 1e-8)

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
            torch.save({"epoch": epoch,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state":  optimizer.state_dict(),
                        "val_sisdr": val_sdr,
                        "val_cos":   val_cos}, p)
            logger.info(f"  ** Best sdr={val_sdr:.2f}dB cos={val_cos:.4f} -> {p.name}")

        if epoch % SAVE_EVERY == 0:
            p = CHECKPOINT_DIR / f"visiovox_devA_ep{epoch:03d}.pth"
            torch.save({"epoch": epoch,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state":  optimizer.state_dict(),
                        "val_sisdr": val_sdr,
                        "val_cos":   val_cos}, p)
            logger.info(f"  Checkpoint: {p.name}")

        # Overfitting alert
        if epoch >= 10 and gap > 2.0:
            logger.warning(f"OVERFIT ALERT ep{epoch}: val/trn gap={gap:.2f}x > 2.0")

        # Collapse alert
        if epoch == 20 and val_cos > 0.80:
            logger.warning(f"COLLAPSE ALERT ep{epoch}: val_cos={val_cos:.4f} > 0.80")

    logger.info("=" * 65)
    logger.info(f"Done. Best val_sdr={best_sdr:.2f}dB at epoch {best_epoch}")
    logger.info("Next: update backend/app.py CHECKPOINT_PATH and "
                "update inference.py to use v4 model architecture")


if __name__ == "__main__":
    if not METADATA_CSV.exists():
        print(f"ERROR: {METADATA_CSV} not found. Run create_mixtures_devA.py first.")
        sys.exit(1)
    main()