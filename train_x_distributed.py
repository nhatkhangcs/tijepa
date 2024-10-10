import torch
import random
import yaml
import pprint
import copy

from dataclasses import dataclass, asdict

from src.models import modules
from src.models.modules import text_encoder_model, vision_encoder, x_t2i_module, vit_predictor
from src.utils.tensors import apply_masks, repeat_interleave_batch

from create_dataset import ImageTextDataset
from src.masks.multiblock import MaskCollator
import torch.nn.functional as F
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import torch.nn as nn

from src.utils.visualizer import visualize_rectangle
from src.utils.saving import Saver

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
    CROSS_ATTN_DEPTH: int = 6

    PRED_NUM_HEADS: int = 12
    CROSS_NUM_HEADS: int = 8

MODEL_CONFIG = ModelConfig()
##################

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

# Context T2I Module
context_crosser = x_t2i_module(
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
).to(DEVICE_1)
context_crosser_total_params = sum(p.numel() for p in context_crosser.parameters())
print(f"{context_crosser_total_params=}")

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

NUM_PATCHES = vision_encoder.patch_embed.num_patches

# Predictor
predictor = vit_predictor(
    embed_dim=MODEL_CONFIG.H_EMBED_DIM,
    depth=MODEL_CONFIG.PRED_ATTN_DEPTH,
    num_heads=MODEL_CONFIG.PRED_NUM_HEADS,
    predictor_embed_dim=MODEL_CONFIG.PRED_EMBED_DIM,
    num_patches=NUM_PATCHES,
    mlp_ratio=MODEL_CONFIG.MLP_RATIO,
    qkv_bias=True,
    qk_scale=None,
    drop_rate=MODEL_CONFIG.DROP_RATE,
    attn_drop_rate=MODEL_CONFIG.ATTN_DROP_RATE,
).to(DEVICE_1)
predictor_total_params = sum(p.numel() for p in predictor.parameters())
print(f"{predictor_total_params=}")

def train(num_epochs=1, max_images_per_epoch=10, batch_size=10, learning_rate=0.01):
    import time
    session = str(int(time.time()))
    
    # Optimizer
    optimizer = optim.Adam(
        list(context_crosser.parameters()) +
        list(predictor.parameters()), lr=learning_rate
    )
            
    dataset = ImageTextDataset(
        image_path='src/datasets/train', 
        caption_path='src/datasets/annotations/filename_caption_dict.json', 
        batch_size=batch_size,
        img_size=MODEL_CONFIG.SIZE,
        patch_size=MODEL_CONFIG.PATCH_SIZE,
        max=max_images_per_epoch,
        transform=transforms.Compose(
            [
                transforms.Resize((MODEL_CONFIG.SIZE, MODEL_CONFIG.SIZE)), 
                transforms.ToTensor()
            ]
        ),
        block_scale=(0.05, 0.1),
        block_aspect_ratio=(0.75, 1.5),
        device_image=DEVICE_0,
        device_context_masks=DEVICE_0,
        device_predict_masks=DEVICE_1,
    )
    
    saver = Saver(
        metrics = ['loss'],
        folder_name = 'xd',
        **asdict(MODEL_CONFIG)
    )

    # ema = (0.999, 1.0)
    ema = (0.996, 1.0)
    ipe_scale = 1.0
    momentum_scheduler = (
        ema[0] + i*(ema[1]-ema[0])/(len(dataset)*num_epochs*ipe_scale)
        for i in range(int(len(dataset)*num_epochs*ipe_scale)+1)
    )
        
    for epoch in range(num_epochs):
        # Set the models to train mode
        context_crosser.train()
        predictor.train()

        # Initialize tqdm for the dataset
        with tqdm(dataset, desc=f"Epoch {epoch+1}/{num_epochs}") as pbar:
            for images, captions, context_masks, predict_masks in pbar:
                # visualize_rectangle(
                #     context_masks[0].tolist(),
                #     predict_masks[0].tolist(),
                # )
                # print(images)
                # print(captions)
                # Zero the gradients
                optimizer.zero_grad()
                
                start_time = time.time()
                print(f"Encoding {len(images)} images and {len(captions)} captions...")
                encoded_text, text_attn_mask = text_encoder(captions)  # Encode the text
                encoded_image_full = vision_encoder(images)  # Encode the context patches
                print(f"\tDone in {time.time() - start_time} seconds")

                start_time = time.time()
                print(f"Moving images to {DEVICE_1}...")
                encoded_image_masked = apply_masks(encoded_image_full, context_masks).to(DEVICE_1)  # Apply context mask
                context_masks = context_masks.to(DEVICE_1)
                print(f"\tDone in {time.time() - start_time} seconds")  

                # print(f"{encoded_image.shape=}")
                # print(f"{masked_image.shape=}")
                # print(f"{encoded_text.shape=}")
                # print(f"{text_attn_mask.shape=}")

                start_time = time.time()
                print(f"Cross encoding context...")
                cross_encoded_context = context_crosser(encoded_text, encoded_image_masked, text_attn_mask)  # Cross encode the text and context
                # print(f"{cross_encoded.shape=}")
                print(f"\tDone in {time.time() - start_time} seconds")

                start_time = time.time()
                print(f"Predicting...")
                predicted = predictor(cross_encoded_context, context_masks, predict_masks)  # Generate predictions based on context
                # print(f"{predicted.shape=}")
                print(f"\tDone in {time.time() - start_time} seconds")

                start_time = time.time()
                print(f"Moving predictions to {DEVICE_0}...")   
                predicted = predicted.to(DEVICE_0)
                print(f"\tDone in {time.time() - start_time} seconds")

                start_time = time.time()   
                print(f"Moving masks and encoded texts to {DEVICE_0}...")
                text_attn_mask = text_attn_mask.to(DEVICE_0, non_blocking=True)
                predict_masks = predict_masks.to(DEVICE_0, non_blocking=True)
                encoded_text = encoded_text.to(DEVICE_0)
                print(f"\tDone in {time.time() - start_time} seconds")
                
                start_time = time.time()
                print(f"Cross encoding target...")
                cross_encoded_target = target_crosser(encoded_text, encoded_image_full, text_attn_mask)  # Cross encode the text and context
                # print(f"{cross_encoded_target.shape=}")
                print(f"\tDone in {time.time() - start_time} seconds")
                
                start_time = time.time()
                print(f"Normalizing target...")
                target = F.layer_norm(cross_encoded_target, (cross_encoded_target.size(-1),))  # Normalize the target
                # print(f"{target.shape=}")
                target = apply_masks(target, predict_masks)  # Apply predict mask
                # print(f"After mask: {target.shape=}")
                print(f"\tDone in {time.time() - start_time} seconds")

                # print(target[0][0][:7].tolist())
                # print(predicted[0][0][:7].tolist())
                # print()
                # print(target[0][1][:7].tolist())
                # print(predicted[0][1][:7].tolist())
                # print()
                # print(target[1][0][:7].tolist())
                # print(predicted[1][0][:7].tolist())
                # print()
                # print(target[1][1][:7].tolist())
                # print(predicted[1][1][:7].tolist())
                # print()
                
                # Calculate loss (L1 loss here)
                p_loss = F.smooth_l1_loss(predicted, target)
                
                # text_embeddings, image_embeddings = model(captions, images, context_masks)

                # Calculate the contrastive loss
                # simamese_loss = siamese_contrastive_loss(T_CLS, V_CLS)
                
                # print(predicted[0][0])
                # print(target[0][0])
                
                loss = p_loss # + c_loss
                saver.update_metric(
                    {
                        'loss': loss.tolist()
                    }
                )
                
                start_time = time.time()
                print(f"Optimizing...")
                # Backward pass and optimization
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                print(f"\tDone in {time.time() - start_time} seconds")

                start_time
                print(f"Updating target encoder...")
                # Step 3. momentum update of target encoder
                with torch.no_grad():
                    m = next(momentum_scheduler)
                    print(m)
                    for param_q, param_k in zip(context_crosser.parameters(), target_crosser.parameters()):
                        param_k.data.mul_(m).add_((1.-m) * param_q.to(DEVICE_0).detach().data)
                print(f"\tDone in {time.time() - start_time} seconds")
                        
                # Update tqdm description with current loss values
                pbar.set_postfix({
                    'MEM': torch.cuda.max_memory_allocated() / 1024.**3,
                    'P Loss': p_loss.item(),
                    # 'Clip Loss': c_loss.item()
                    # 'Siamese Loss': simamese_loss.item(),
                    # 'Hinge Loss': hinge_loss.item()
                    # 'Total Loss': loss.item()
                })
        
        saver.save_epoch()


def main():
    train(
        num_epochs=30, 
        max_images_per_epoch=260, 
        batch_size=48,
        learning_rate=0.0005
    )
    
if __name__ == "__main__":
    main()
