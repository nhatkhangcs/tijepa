import torch
import random
import yaml
import pprint
import copy
import time

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
from eval_on_mvsa import train_simple_linear_module

DEVICE_0 = 'cuda:0'

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

def train(num_epochs=1, max_images_per_epoch=10, batch_size=10, mini_batch_size=10, learning_rate=0.01, save_interval=1, resume_from=None):

    start_epoch = 0
    
    dataset = VQADataset(
        batch_size=batch_size,
        img_size=MODEL_CONFIG.SIZE,
        shuffle=False,
        max=max_images_per_epoch,
    )
    
    # -- OPTIMIZATION
    ipe_scale = params['optimization']['ipe_scale']  # scheduler scale factor (def: 1.0)
    print(f"{ipe_scale=}")
    wd = float(params['optimization']['weight_decay'])
    print(f"{wd=}")
    final_wd = float(params['optimization']['final_weight_decay'])
    print(f"{final_wd=}")
    warmup = params['optimization']['warmup']
    print(f"{warmup=}")
    start_lr = params['optimization']['start_lr']
    print(f"{start_lr=}")
    lr = learning_rate
    print(f"{lr=}")
    final_lr = params['optimization']['final_lr']
    print(f"{final_lr=}")
    ipe = len(dataset)

    optimizer, scaler, scheduler, wd_scheduler = init_opt_fine_tune(
        encoder=crosser,
        mlp_head=mlp_head,
        iterations_per_epoch=ipe,
        wd=wd,
        final_wd=final_wd,
        start_lr=start_lr,
        ref_lr=lr,
        final_lr=final_lr,
        warmup=warmup,
        num_epochs=num_epochs,
        ipe_scale=ipe_scale,
        use_bfloat16=True
    )

    previous_metrics = None
    if resume_from is not None:
        print(f"Resuming from {resume_from}...")
        saved_dict = torch.load(resume_from, map_location='cpu')
        crosser.load_state_dict(saved_dict['crosser'])
        mlp_head.load_state_dict(saved_dict['mlp_head'])

        optimizer.load_state_dict(saved_dict['opt'])
        scaler.load_state_dict(saved_dict['scaler'])
        start_epoch = saved_dict['epoch']
        print(f"Resumed from epoch {start_epoch}")

        for _ in range(start_epoch*ipe):
            scheduler.step()
            wd_scheduler.step()
        
        previous_metrics = {}
        previous_metrics['loss'] = saved_dict['loss']
        print(f"Loaded previous metrics: {len(previous_metrics['loss'])} records")

        del saved_dict

    saver = Saver(
        metrics = [
            'loss',
        ],
        folder_name = 'VQA',
        current_epoch = start_epoch,
        previous_metrics = previous_metrics,
        **(
            asdict(MODEL_CONFIG) |
            {
                'lr': learning_rate
            }
        )
    )

    last_time = time.time()

    loss_fn = torch.nn.CrossEntropyLoss()

    # start from start_epoch
    for epoch in range(start_epoch, num_epochs):

        # Set the models to train mode
        crosser.train()
        mlp_head.train()

        # Initialize tqdm for the dataset
        with tqdm(dataset, desc=f"Epoch {epoch+1}/{num_epochs}") as pbar:
            for images, questions, answers in pbar:

                start_time = time.time()
                print(f"Load 1 iter dataset in {start_time-last_time} secs")

                # Zero the gradients
                loss = 0

                n_mini_iter = len(images) // mini_batch_size

                optimizer.zero_grad()
                _new_lr = scheduler.step()
                saver.log(f"{_new_lr=}")
                _new_wd = wd_scheduler.step()
                saver.log(f"{_new_wd=}")

                # Loop through mini-batches
                for i in range(0, len(images), mini_batch_size):
                    mini_images = images[i:i+mini_batch_size]
                    mini_questions = questions[i:i+mini_batch_size]
                    mini_answers = answers[i:i+mini_batch_size]

                    with torch.cuda.amp.autocast(dtype=torch.bfloat16, enabled=True):
                    
                        with torch.no_grad():
                            encoded_text, text_attn_mask = text_encoder(mini_questions)  # Encode the text
                            # print(f"{encoded_text.shape=}")
                            # print(f"{encoded_text=}")
                            # print(f"{text_attn_mask.shape=}")
                            # print(f"{text_attn_mask=}")
                            encoded_image_full = vision_encoder(mini_images)  # Encode the context patches
                            # print(f"{encoded_image_full.shape=}")
                            # print(f"{encoded_image_full=}")

                            # start_time = time.time()
                            # for idx, record in enumerate(encoded_image_full):
                            #     print(idx, record[0][:3], record.shape)

                        # start_time = time.time()
                        # print(f"Cross encoding context...")
                        cross_encoded = crosser(encoded_text, encoded_image_full, text_attn_mask)  # Cross encode the text and context
                        # print(f"{cross_encoded_context.shape=}")
                        # print(f"{cross_encoded_context=}")
                        # print(f"{cross_encoded.shape=}")
                        # print(f"\tDone in {time.time() - start_time} seconds")
                        pooled_encoded = cross_encoded.mean(dim=1)
                        
                        logits = mlp_head(pooled_encoded)
                        # print(f"{logits.shape=}")
                        
                        mini_answers = torch.tensor(mini_answers, device=logits.device, dtype=torch.long)
                        # print(f"{mini_answers.shape=}")
                        
                        # Compute cross-entropy loss
                        ce_loss = loss_fn(logits, mini_answers)
                
                        loss += ce_loss.item()

                        # saver.log(target[1][20][:10])
                        # saver.log(predicted[1][20][:10])
                        saver.log('\n--')

                        saver.update_metric(
                            {
                                'loss': ce_loss.tolist(),
                            }
                        )
                        loss += ce_loss.item()
                        
                        # start_time = time.time()
                        # Backward pass
                        scaler.scale(ce_loss).backward()

                # Optimizer step
                scaler.step(optimizer)
                scaler.update()

                loss = loss / n_mini_iter

                saver.save_epoch(temp=True)
                saver.log(f"Finished 1 iter in {time.time() - start_time} seconds")
                print(f"Finished 1 iter in {time.time() - start_time} seconds")

                last_time = time.time()
                        
                # Update tqdm description with current loss values
                pbar.set_postfix(
                    {
                        'loss': loss
                    }
                )
        
        saver.save_epoch()

        if (epoch + 1) % save_interval == 0:
            save_dict = {
                'crosser': crosser.state_dict(),
                'mlp_head': mlp_head.state_dict(),
                'opt': optimizer.state_dict(),
                'scaler': None if scaler is None else scaler.state_dict(),
                'epoch': epoch + 1,
                'loss': loss
            }
            saver.save_checkpoint(save_dict, epoch=epoch+1)
            saver.log(f"Saved checkpoint: {save_dict['epoch']}, loss = {save_dict['loss']}")

def main():
    train(
        num_epochs=40, 
        max_images_per_epoch=None, # 50000 
        mini_batch_size=100, # 80
        batch_size=50*10, # 80*6
        learning_rate=1e-5,
        save_interval=5,
        resume_from=None,
    )

if __name__ == "__main__":
    main()
