import torch
import random
import yaml
import pprint
import copy

from dataclasses import dataclass, asdict

from src.models import modules
from src.models.modules import text_encoder_model, x_t2i_module, vit_predictor

DEVICE_0 = 'cuda:0'
DEVICE_1 = 'cuda:1'

##################
with open('configs/in1k_vith14_ep300.yaml', 'r') as y_file:
    params = yaml.load(y_file, Loader=yaml.FullLoader)
    pp = pprint.PrettyPrinter(indent=4)
    # pp.pprint(params)

@dataclass
class ModelConfig:
    SIZE: int = 224
    PATCH_SIZE: int = params['mask']['patch_size']

    V_EMBED_DIM: int = 1280
    T_EMBED_DIM: int = 768
    H_EMBED_DIM: int = 1024
    PRED_EMBED_DIM: int = params['meta']['pred_emb_dim']

    DROP_RATE: float = 0.15
    ATTN_DROP_RATE: float = 0.15
    MLP_RATIO: float = 4.0

    PRED_ATTN_DEPTH: int = params['meta']['pred_depth']
    CROSS_ATTN_DEPTH: int = 4

    PRED_NUM_HEADS: int = 12
    CROSS_NUM_HEADS: int = 8

MODEL_CONFIG = ModelConfig()

def load(d, h):
    print(f"Init models...")

    # Target T2I Module
    target_crosser = x_t2i_module(
        text_embed_dim=MODEL_CONFIG.T_EMBED_DIM,
        vision_embed_dim=MODEL_CONFIG.V_EMBED_DIM,
        hidden_dim=MODEL_CONFIG.H_EMBED_DIM,
        depth=d,
        num_heads=h,
        mlp_ratio=MODEL_CONFIG.MLP_RATIO,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=MODEL_CONFIG.DROP_RATE,
        attn_drop_rate=MODEL_CONFIG.ATTN_DROP_RATE,
    ).to(DEVICE_0)
    target_crosser_total_params = sum(p.numel() for p in target_crosser.parameters())
    print(f"{target_crosser_total_params=}")

#load(4,8)
#load(6,10)
load(8,12)