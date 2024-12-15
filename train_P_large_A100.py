# import torch
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
from src.utils.losses import cosine_similarity_matrix, contrastive_loss, clip_loss, max_margin_loss, max_margin_loss_negative_only, weighted_max_margin_loss

from create_dataset import ImageTextDatasetA100
from src.masks.multiblock import MaskCollator
# import torch.nn.functional as F
# import torch.optim as optim
# from torchvision import transforms
# from torch.utils.data import Dataset, DataLoader
# from tqdm import tqdm
# import torch.nn as nn

from src.utils.visualizer import visualize_rectangle, print_tensor_with_precision, print_sample_of_tensor
from src.utils.saving import Saver
from eval_on_mvsa import train_simple_linear_module

DEVICE_0 = 'cpu'

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

MODEL_CONFIG = ModelConfig()
##################

# Text Encoder
# text_encoder = text_encoder_model(
#     device=DEVICE_0
# )
# text_encoder_total_params = sum(p.numel() for p in text_encoder.parameters())
# print(f"{text_encoder_total_params=}")
# for p in text_encoder.parameters():
#     p.requires_grad = False

# # Vision Encoder
# vision_encoder = modules.__dict__[params['meta']['model_name']](
#     img_size=[MODEL_CONFIG.SIZE],
#     patch_size=MODEL_CONFIG.PATCH_SIZE,
# ).to(DEVICE_0)
# context_vision_encoder_total_params = sum(p.numel() for p in vision_encoder.parameters())
# print(f"{context_vision_encoder_total_params=}")

# TAR_FILE = "IN1K-vit.h.16-448px-300e.pth.tar"
# print(f"Loading Vision Encoder {TAR_FILE}...")
# checkpoint = torch.load(TAR_FILE, map_location=torch.device(DEVICE_0))
# encoder_dict = checkpoint['target_encoder'] if 'target_encoder' in checkpoint else checkpoint['encoder']
# encoder_dict = {k.replace('module.', ''): v for k, v in encoder_dict.items()}
# msg = vision_encoder.load_state_dict(encoder_dict)
# print(f'loaded pretrained encoder from with msg: {msg}')
# for p in vision_encoder.parameters():
#     p.requires_grad = False

# del checkpoint
# del encoder_dict

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
).to(DEVICE_0)
context_crosser_total_params = sum(p.numel() for p in context_crosser.parameters())
print(f"{context_crosser_total_params=}")
assert False
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
for p in target_crosser.parameters():
    p.requires_grad = False

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
).to(DEVICE_0)
predictor_total_params = sum(p.numel() for p in predictor.parameters())
print(f"{predictor_total_params=}")


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

def train(num_epochs=1, max_images_per_epoch=10, batch_size=10, mini_batch_size=10, learning_rate=0.01, save_interval=1, resume_from=None):

    # Optimizer
    # optimizer = optim.Adam(
    #     list(context_crosser.parameters()) +
    #     list(predictor.parameters()), lr=learning_rate
    # )
    start_epoch = 0
            
    dataset = ImageTextDatasetA100(
        image_path='src/datasets/train', 
        caption_path='src/datasets/annotations/filename_caption_dict.json', 
        batch_size=batch_size,
        img_size=MODEL_CONFIG.SIZE,
        patch_size=MODEL_CONFIG.PATCH_SIZE,
        max=max_images_per_epoch,
        transform=transforms.Compose(
            [
                transforms.ToTensor()
            ]
        ), 
        block_scale=(0.15, 0.2), # originally block_scale=(0.15, 0.2),
        block_aspect_ratio=(0.75, 1.5),
        device=DEVICE_0,
        shuffle=False,
        tensor_folder="src/datasets/train-tensor-448-10k",
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

    optimizer, scaler, scheduler, wd_scheduler = init_opt(
        encoder=context_crosser,
        predictor=predictor,
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

    # ema = (0.999, 1.0)
    ema = (0.996, 1.0)
    momentum_scheduler = (
        ema[0] + i*(ema[1]-ema[0])/(ipe*num_epochs*ipe_scale)
        for i in range(int(ipe*num_epochs*ipe_scale)+1)
    )

    previous_metrics = None
    if resume_from is not None:
        print(f"Resuming from {resume_from}...")
        saved_dict = torch.load(resume_from, map_location='cpu')
        context_crosser.load_state_dict(saved_dict['context_crosser'])
        predictor.load_state_dict(saved_dict['predictor'])
        target_crosser.load_state_dict(saved_dict['target_crosser'])

        optimizer.load_state_dict(saved_dict['opt'])
        scaler.load_state_dict(saved_dict['scaler'])
        start_epoch = saved_dict['epoch']
        print(f"Resumed from epoch {start_epoch}")

        for _ in range(start_epoch*ipe):
            scheduler.step()
            wd_scheduler.step()
            _m = next(momentum_scheduler)
        
        print(f"Momemtum: {_m}")

        previous_metrics = {}
        previous_metrics['loss'] = saved_dict['loss']
        print(f"Loaded previous metrics: {len(previous_metrics['loss'])} records")

        del saved_dict

    saver = Saver(
        metrics = [
            'loss', 
        ],
        folder_name = 'SMALL-A100-448-600-10k-OBS-SCHEDULER',
        current_epoch = start_epoch,
        previous_metrics = previous_metrics,
        **asdict(MODEL_CONFIG) 
    )

    last_time = time.time()

    # start from start_epoch
    for epoch in range(start_epoch, num_epochs):

        # Set the models to train mode
        context_crosser.train()
        predictor.train()

        # Initialize tqdm for the dataset
        with tqdm(dataset, desc=f"Epoch {epoch+1}/{num_epochs}") as pbar:
            for images, captions, context_masks, predict_masks in pbar:

                start_time = time.time()
                print(f"Load 1 iter dataset in {start_time-last_time} secs")

                visualize_rectangle(
                    context_masks[0].tolist(), 
                    predict_masks[0].tolist(),
                    p=MODEL_CONFIG.SIZE // MODEL_CONFIG.PATCH_SIZE
                )
                # print(images)
                # print(captions)
                # Zero the gradients
                loss = 0

                n_mini_iter = len(images) // mini_batch_size

                optimizer.zero_grad()
                _new_lr = scheduler.step()
                print(f"{_new_lr=}")
                _new_wd = wd_scheduler.step()
                print(f"{_new_wd=}")

                # Loop through mini-batches
                for i in range(0, len(images), mini_batch_size):
                    mini_images = images[i:i+mini_batch_size]
                    mini_captions = captions[i:i+mini_batch_size]
                    mini_context_masks = context_masks[i:i+mini_batch_size]
                    mini_predict_masks = predict_masks[i:i+mini_batch_size]
                    # print(f"{mini_images.shape=}")
                    # print(f"{mini_images=}")
                    # print(f"{len(mini_captions)=}")
                    # print(f"{mini_captions=}")
                    # print(f"{mini_context_masks.shape=}")
                    # print(f"{mini_context_masks=}")
                    # print(f"{mini_predict_masks.shape=}")
                    # print(f"{mini_predict_masks=}")

                    with torch.cuda.amp.autocast(dtype=torch.bfloat16, enabled=True):
                    
                        # print(f"Encoding {len(mini_images)} images and {len(mini_captions)} captions...")
                        with torch.no_grad():
                            encoded_text, text_attn_mask = text_encoder(mini_captions)  # Encode the text
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
                            encoded_image_masked = apply_masks(encoded_image_full, mini_context_masks) # Apply context mask
                            # print(f"{encoded_image_masked.shape=}")
                            # print(f"{encoded_image_masked=}")
                            # for idx, record in enumerate(encoded_image_masked):
                            #     print(idx, record[0][:3], record.shape)
                            # print(f"\tDone in {time.time() - start_time} seconds")  

                            # print(f"{encoded_image_full.shape=}")
                            # print(f"{encoded_image_masked.shape=}")
                            # print(f"{encoded_text.shape=}")
                            # print(f"{text_attn_mask.shape=}")

                        # start_time = time.time()
                        # print(f"Cross encoding context...")
                        cross_encoded_context = context_crosser(encoded_text, encoded_image_masked, text_attn_mask)  # Cross encode the text and context
                        # print(f"{cross_encoded_context.shape=}")
                        # print(f"{cross_encoded_context=}")
                        # print(f"{cross_encoded.shape=}")
                        # print(f"\tDone in {time.time() - start_time} seconds")

                        # start_time = time.time()
                        # print(f"Predicting...")
                        predicted = predictor(cross_encoded_context, mini_context_masks, mini_predict_masks)  # Generate predictions based on context
                        # print(f"{predicted.shape=}")
                        # print(f"{predicted=}")
                        # print(f"{predicted.shape=}")
                        # print(f"\tDone in {time.time() - start_time} seconds")

                        # start_time = time.time()
                        # print(f"Moving predictions to {DEVICE_0}...")   
                        # print(f"\tDone in {time.time() - start_time} seconds")

                        # start_time = time.time()   
                        # print(f"Moving masks and encoded texts to {DEVICE_0}...")
                        # print(f"\tDone in {time.time() - start_time} seconds")
                        
                        # start_time = time.time()
                        # print(f"Cross encoding target...")

                        with torch.no_grad():
                            cross_encoded_target = target_crosser(encoded_text, encoded_image_full, text_attn_mask)  # Cross encode the text and context
                            # print(f"{cross_encoded_target.shape=}")
                            # print(f"{cross_encoded_target=}")
                            # print(f"\tDone in {time.time() - start_time} seconds")
                            
                            # start_time = time.time()
                            # print(f"Normalizing target...")
                            target = F.layer_norm(cross_encoded_target, (cross_encoded_target.size(-1),))  # Normalize the target
                            # print(f"{target.shape=}")
                            # print(f"{target=}")
                            
    
                            # print(mini_predict_masks[0])
                            # print(len(mini_predict_masks[0]))
                            # print(f"{target.shape=}")
                            # for idx, patch in enumerate(target[0]):
                            #     print(idx, patch[:3])
                            # for idx, record in enumerate(target):
                            #     print(idx, record[0][:3])
                            
                            # print(mini_predict_masks)
                            target = apply_masks(target, mini_predict_masks)  # Apply predict mask
                            # print(f"{target.shape=}")
                            # print_tensor_with_precision(target[0][0][:10])
                            # print_tensor_with_precision(target[0][1][:10])
                            # print_tensor_with_precision(target[0][2][:10])
                            # print_tensor_with_precision(target[0][3][:10])
                            # print_tensor_with_precision(target[0][4][:10])
                            # print_tensor_with_precision(target[0][5][:10])
                            
                            # print_tensor_with_precision(target[1][0][:10])
                            # print_tensor_with_precision(target[1][1][:10])
                            # print_tensor_with_precision(target[1][2][:10])
                            # print_tensor_with_precision(target[1][3][:10])
                            # print_tensor_with_precision(target[1][4][:10])
                            # print_tensor_with_precision(target[1][5][:10])
                            # print()
    
                            # for idx, record in enumerate(target):
                            #     print(idx, record[0][:3])
    
                            # print(f"After mask: {target.shape=}")
                            # print(f"\tDone in {time.time() - start_time} seconds")
                            
                            # Calculate loss (L1 loss here)
                        p_loss = F.smooth_l1_loss(predicted, target)

                        saver.log(target[0][0][:10])
                        saver.log(predicted[0][0][:10])
                        # print_tensor_with_precision(cross_encoded_context[0][0][:10])
                        saver.log('\n')
                        saver.log(target[0][1][:10])
                        saver.log(predicted[0][1][:10])
                        # print_tensor_with_precision(cross_encoded_context[0][1][:10])
                        saver.log('\n')
                        saver.log(target[1][10][:10])
                        saver.log(predicted[1][10][:10])
                        # print_tensor_with_precision(cross_encoded_context[1][10][:10])
                        saver.log('\n')
                        saver.log(target[1][20][:10])
                        saver.log(predicted[1][20][:10])
                        # print_tensor_with_precision(cross_encoded_context[1][20][:10])
                        saver.log('\n--')

                        saver.log(
                            "TARGET: ",
                            *cosine_similarity_matrix(
                                target.mean(dim=1),
                            )[:8,:8].tolist()
                        )
                        saver.log(
                            "PREDICTED: ",
                            *cosine_similarity_matrix(
                                predicted.mean(dim=1),
                            )[:8,:8].tolist()
                        )

                        saver.update_metric(
                            {
                                'loss': p_loss.tolist(),
                            }
                        )
                        loss += p_loss.item()
                        
                        # start_time = time.time()
                        # Backward pass
                        scaler.scale(p_loss).backward()

                # Optimizer step
                scaler.step(optimizer)
                scaler.update()

                optimizer.zero_grad()
                # print(f"\tDone in {time.time() - start_time} seconds")

                # start_time = time.time()
                # print(f"Updating target encoder...")
                # Step 3. momentum update of target encoder
                with torch.no_grad():
                    m = next(momentum_scheduler)
                    saver.log(f"Momentum: {m}")
                    for param_q, param_k in zip(context_crosser.parameters(), target_crosser.parameters()):
                        param_k.data.mul_(m).add_((1.-m) * param_q.detach().data)
                # print(f"\tDone in {time.time() - start_time} seconds")

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
                'context_crosser': context_crosser.state_dict(),
                'predictor': predictor.state_dict(),
                'target_crosser': target_crosser.state_dict(),
                'opt': optimizer.state_dict(),
                'scaler': None if scaler is None else scaler.state_dict(),
                'epoch': epoch + 1,
                'loss': loss
            }
            target_crosser_only = {
                'target_crosser': target_crosser.state_dict(),
            }
            saver.save_checkpoint(save_dict, epoch=epoch+1)
            saver.save_checkpoint(target_crosser_only, epoch=epoch+1, target_crosser_only=True)
            saver.log(f"Saved checkpoint: {save_dict['epoch']}, loss = {save_dict['loss']}")


def main():
    train(
        num_epochs=600, 
        max_images_per_epoch=10000, # 50000 
        mini_batch_size=26, # 80
        batch_size=26*15, # 80*6
        learning_rate=0.001,
        save_interval=20,
        resume_from="trains/SMALL-A100-448-10k-OBS-SCHEDULER/epoch-300.pt",
    )

if __name__ == "__main__":
    main()
