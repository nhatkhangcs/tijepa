import torch
import random
import yaml
import pprint
import copy
import time
import os
import json

from dataclasses import dataclass, asdict

from src.models import modules
from src.models.modules import text_encoder_model, x_t2i_module, vit_predictor, MLP
from src.utils.tensors import apply_masks, repeat_interleave_batch
from src.helper import init_opt_fine_tune
from src.utils.losses import cosine_similarity_matrix, contrastive_loss, clip_loss, max_margin_loss, max_margin_loss_negative_only, weighted_max_margin_loss

from vqa_dataset import VQADataset
from src.masks.multiblock import MaskCollator
import torch.nn.functional as F
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import torch.nn as nn

from src.utils.visualizer import visualize_rectangle, print_tensor_with_precision, print_sample_of_tensor
from src.utils.saving import Saver

from metrics import calculate_metrics_from_logits, indices_to_one_hot

DEVICE_0 = 'cuda:0'
CHECKPOINT = "trains/VQA-1732161891/epoch-30.pt"

##################
with open('configs/in1k_vith16-448_ep300.yaml', 'r') as y_file:
    params = yaml.load(y_file, Loader=yaml.FullLoader)
    pp = pprint.PrettyPrinter(indent=4)
    pp.pprint(params)

@dataclass
class ModelConfig:
    SIZE: int = params['data']['crop_size']
    PATCH_SIZE: int = params['mask']['patch_size']

    V_EMBED_DIM: int = 1280
    T_EMBED_DIM: int = 768
    H_EMBED_DIM: int = 768
    PRED_EMBED_DIM: int = params['meta']['pred_emb_dim']

    DROP_RATE: float = 0. # 0.15
    ATTN_DROP_RATE: float = 0. # 0.15
    MLP_RATIO: float = 4.0 # 4.0

    PRED_ATTN_DEPTH: int = params['meta']['pred_depth']
    CROSS_ATTN_DEPTH: int = 4

    PRED_NUM_HEADS: int = 12
    CROSS_NUM_HEADS: int = 8

    MLP_HEAD_HIDDEN_DIM = 1536

MODEL_CONFIG = ModelConfig()
##################

# Text Encoder
text_encoder = text_encoder_model(
    device=DEVICE_0
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

TAR_FILE = "IN1K-vit.h.16-448px-300e.pth.tar"
print(f"Loading Vision Encoder {TAR_FILE}...")
checkpoint = torch.load(TAR_FILE, map_location=torch.device(DEVICE_0))
encoder_dict = checkpoint['target_encoder'] if 'target_encoder' in checkpoint else checkpoint['encoder']
encoder_dict = {k.replace('module.', ''): v for k, v in encoder_dict.items()}
msg = vision_encoder.load_state_dict(encoder_dict)
print(f'loaded pretrained encoder from with msg: {msg}')
for p in vision_encoder.parameters():
    p.requires_grad = False

del checkpoint
del encoder_dict

# Context T2I Module
crosser = x_t2i_module(
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
crosser_total_params = sum(p.numel() for p in crosser.parameters())
print(f"{crosser_total_params=}")

TIJEPA_file = "trains/SMALL-A100-448-10k-OBS-SCHEDULER/epoch-300.pt"
print(f"Loading TIJEPA crosser {TIJEPA_file}...")
checkpoint = torch.load(TIJEPA_file, map_location=torch.device(DEVICE_0))
crosser_dict = checkpoint['target_crosser']
# crosser_dict = {k.replace('module.', ''): v for k, v in crosser_dict.items()}
msg = crosser.load_state_dict(crosser_dict)
print(f'loaded pretrained crosser from with msg: {msg}')

del checkpoint
del crosser_dict

mlp_head = MLP(
    in_features=MODEL_CONFIG.H_EMBED_DIM,
    hidden_features=MODEL_CONFIG.MLP_HEAD_HIDDEN_DIM,
    out_features=3129,
).to(DEVICE_0)

NUM_PATCHES = vision_encoder.patch_embed.num_patches

dataset = VQADataset(
    batch_size=1,
    img_size=MODEL_CONFIG.SIZE,
    shuffle=False,
    max=None,
    max_val=5000,
)

saved_dict = torch.load(CHECKPOINT, map_location='cpu')
crosser.load_state_dict(saved_dict['crosser'])
mlp_head.load_state_dict(saved_dict['mlp_head'])

del saved_dict

# TEST
num_classes = 3129

crosser.eval()
mlp_head.eval()

RES = []
"""
RES = [
    {
        {"question_id": 88355000, "answer": "10"}
    },
    ...
]
"""

output_dir = "vqa_dataset/pred_test"
os.makedirs(output_dir, exist_ok=True)

with torch.no_grad():
    with tqdm(dataset.iter_test(), desc="Inference") as pbar:
        for i, (images, questions, ids) in enumerate(pbar): 
            
            encoded_text, text_attn_mask = text_encoder(questions)
            encoded_image_full = vision_encoder(images)  # Encode the context patches
            cross_encoded = crosser(encoded_text, encoded_image_full, text_attn_mask)
            pooled_encoded = cross_encoded.mean(dim=1)
            
            logits = mlp_head(pooled_encoded)  # (batch_size, 3129)
            
            predicted_indices = logits.argmax(dim=1)  
            
            for idx, question_id in enumerate(ids):
                answer = dataset.remapper[predicted_indices[idx].item()] 
                RES.append(
                    {
                        "question_id": question_id, 
                        "answer": answer,
                    }
                )

            with open(os.path.join(output_dir, f"{i}.json"), "w") as f:
                json.dump(RES, f)
            
            pbar.set_postfix(
                {"JSON LEN": len(RES)}
            )
print("==== Done ====")
