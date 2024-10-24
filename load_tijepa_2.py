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
    pp.pprint(params)

@dataclass
class ModelConfig:
    SIZE: int = 224
    PATCH_SIZE: int = params['mask']['patch_size']

    V_EMBED_DIM: int = 1280
    T_EMBED_DIM: int = 768
    H_EMBED_DIM: int = 768
    PRED_EMBED_DIM: int = params['meta']['pred_emb_dim']

    DROP_RATE: float = 0.15
    ATTN_DROP_RATE: float = 0.15
    MLP_RATIO: float = 4.0

    PRED_ATTN_DEPTH: int = params['meta']['pred_depth']
    CROSS_ATTN_DEPTH: int = 4

    PRED_NUM_HEADS: int = 12
    CROSS_NUM_HEADS: int = 8

MODEL_CONFIG = ModelConfig()
##################

def inference(images, captions, text_encoder, vision_encoder, target_crosser, device=DEVICE_0):
    # Encode the text
    encoded_text, text_attn_mask = text_encoder(captions)
    
    # Encode the context patches
    encoded_image_full = vision_encoder(images)
    
    # Cross encode the text and context
    cross_encoded_target = target_crosser(encoded_text, encoded_image_full, text_attn_mask)

    # Average pooling the cross_encoded_target
    cross_encoded_target = cross_encoded_target.mean(dim=1)
    print(f"{cross_encoded_target.shape=}")
    
    return cross_encoded_target

def load(checkpoint_path):
    print(f"Init models...")
    # Text Encoder
    text_encoder = text_encoder_model(
        device=DEVICE_1
    )
    text_encoder_total_params = sum(p.numel() for p in text_encoder.parameters())
    print(f"{text_encoder_total_params=}")
    for p in text_encoder.parameters():
        p.requires_grad = False

    # Vision Encoder
    vision_encoder = modules.__dict__[params['meta']['model_name']](
        img_size=[MODEL_CONFIG.SIZE],
        patch_size=MODEL_CONFIG.PATCH_SIZE,
    ).to(DEVICE_0)
    context_vision_encoder_total_params = sum(p.numel() for p in vision_encoder.parameters())
    print(f"{context_vision_encoder_total_params=}")

    TAR_FILE = "checkpoints/vith.tar"
    print(f"Loading Vision Encoder {TAR_FILE}...")
    checkpoint = torch.load(TAR_FILE, map_location=torch.device(DEVICE_1))
    encoder_dict = checkpoint['target_encoder'] if 'target_encoder' in checkpoint else checkpoint['encoder']
    encoder_dict = {k.replace('module.', ''): v for k, v in encoder_dict.items()}
    msg = vision_encoder.load_state_dict(encoder_dict)
    print(f'loaded pretrained encoder from with msg: {msg}')
    for p in vision_encoder.parameters():
        p.requires_grad = False

    del checkpoint
    del encoder_dict

    # Target T2I Module
    target_crosser = x_t2i_module(
        text_embed_dim=MODEL_CONFIG.T_EMBED_DIM,
        vision_embed_dim=MODEL_CONFIG.V_EMBED_DIM,
        hidden_dim=MODEL_CONFIG.H_EMBED_DIM,
        depth=MODEL_CONFIG.CROSS_ATTN_DEPTH,
        num_heads=MODEL_CONFIG.CROSS_NUM_HEADS,
        mlp_ratio=MODEL_CONFIG.MLP_RATIO,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=MODEL_CONFIG.DROP_RATE,
        attn_drop_rate=MODEL_CONFIG.ATTN_DROP_RATE,
    ).to(DEVICE_0)
    target_crosser_total_params = sum(p.numel() for p in target_crosser.parameters())
    print(f"{target_crosser_total_params=}")

    print('\n\nDone init models\n\n')

    print(f"Loading from {checkpoint_path}...")
    saved_dict = torch.load(checkpoint_path, map_location='cpu')
    target_crosser.load_state_dict(saved_dict['target_crosser'])
    start_epoch = saved_dict['epoch']
    print(f"Loaded from epoch {start_epoch}")

    del saved_dict

    return text_encoder, vision_encoder, target_crosser

def inference(images, captions, text_encoder, vision_encoder, target_crosser):
    # Encode the text
    encoded_text, text_attn_mask = text_encoder(captions)
    
    # Encode the context patches
    encoded_image_full = vision_encoder(images)
    
    # Cross encode the text and context
    cross_encoded_target = target_crosser(encoded_text, encoded_image_full, text_attn_mask)

    # Average pooling the cross_encoded_target
    cross_encoded_target = cross_encoded_target.mean(dim=1)
    
    return cross_encoded_target
