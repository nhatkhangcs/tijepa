import torch
import random
import yaml
import pprint
import copy
import time

from dataclasses import dataclass, asdict

from src.models import modules
from src.models.modules import text_encoder_model, x_t2i_module, vit_predictor
from src.utils.tensors import apply_masks, repeat_interleave_batch
from src.helper import init_opt
from src.utils.losses import similarity_matrix, contrastive_loss, clip_loss, max_margin_loss, max_margin_loss_negative_only, weighted_max_margin_loss

from create_dataset import ImageTextDataset
from src.masks.multiblock import MaskCollator
import torch.nn.functional as F
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import torch.nn as nn

from src.utils.visualizer import visualize_rectangle, print_tensor_with_precision, print_sample_of_tensor
from src.utils.saving import Saver
from eval_on_mvsa import train_simple_linear_module

DEVICE_0 = 'cuda:0'

##################
with open('configs/in1k_vith16-448_ep300.yaml', 'r') as y_file:
    params = yaml.load(y_file, Loader=yaml.FullLoader)
    pp = pprint.PrettyPrinter(indent=4)
    pp.pprint(params)

@dataclass
class ModelConfig:
    SIZE: int = 448 # => 28^2 patches # prev 224
    PATCH_SIZE: int = params['mask']['patch_size']

    V_EMBED_DIM: int = 1280
    T_EMBED_DIM: int = 768
    H_EMBED_DIM: int = 768
    PRED_EMBED_DIM: int = params['meta']['pred_emb_dim']

    DROP_RATE: float = 0.15
    ATTN_DROP_RATE: float = 0.15
    MLP_RATIO: float = 4.0

    PRED_ATTN_DEPTH: int = params['meta']['pred_depth']
    CROSS_ATTN_DEPTH: int = 8 # prev 4

    PRED_NUM_HEADS: int = 12
    CROSS_NUM_HEADS: int = 12 # prev 8

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

T, attn_mask = text_encoder(
    [
        "Hello",
        "hello",
        "hi",
        ""
    ], 
    verbose=True
)
print(T)
print(attn_mask)
