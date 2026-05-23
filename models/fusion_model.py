# models/fusion_model.py  -  VisioVox v4
# Additive fusion: both audio and visual projected to 128-dim then ADDED
# This prevents the scale-mismatch problem where concatenation caused
# audio (512-dim) to completely dominate and lips to be ignored

import torch
import torch.nn as nn
from models.visual_model import TemporalVisualEncoder
from models.audio_model import AudioUNet

VISUAL_DIM = 128


class VisioVox(nn.Module):
    def __init__(self):
        super().__init__()
        self.visual_encoder = TemporalVisualEncoder(embedding_dim=VISUAL_DIM)
        self.audio_unet     = AudioUNet(dropout_p=0.2)
        self.audio_proj     = nn.Sequential(
            nn.Conv2d(512, VISUAL_DIM, 1),
            nn.BatchNorm2d(VISUAL_DIM),
            nn.ReLU(inplace=True),
        )
        self.visual_proj   = nn.Linear(VISUAL_DIM, VISUAL_DIM)
        self.fusion_refine = nn.Sequential(
            nn.Conv2d(VISUAL_DIM, 512, 3, padding=1),
            nn.BatchNorm2d(512), nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.BatchNorm2d(512), nn.ReLU(inplace=True),
        )

    def forward(self, mixed_spec, lip_frames, drop_visual=False):
        """
        mixed_spec : [B, 1, F, T]
        lip_frames : [B, 25, 1, 112, 112]   <-- 25 frames, not a single frame
        Returns    : (mask [B, 1, F, T], v_emb [B, 128])
        """
        v_emb = self.visual_encoder(lip_frames)           # [B, 128]
        if drop_visual:
            v_emb = torch.zeros_like(v_emb)

        bottleneck, skips = self.audio_unet.encode(mixed_spec)
        a_proj = self.audio_proj(bottleneck)              # [B, 128, F', T']
        v_proj = self.visual_proj(v_emb)                  # [B, 128]
        v_spat = v_proj.unsqueeze(-1).unsqueeze(-1).expand_as(a_proj)

        fused  = a_proj + v_spat                          # additive
        fused  = self.fusion_refine(fused)                # [B, 512, F', T']
        mask   = self.audio_unet.decode(fused, skips)     # [B, 1, F, T]
        return mask, v_emb