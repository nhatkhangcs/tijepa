import torch
import random
import os
from src.models.vision_transformer import vit_test, vit_predictor_test
from src.models.modules import text_encoder_model, vision_encoder, crosser_module
from src.utils.tensors import apply_masks, repeat_interleave_batch
import torch.nn.functional as F
import torch.optim as optim

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from src.masks.multiblock import MaskCollator

# Assuming we have the MaskCollator class defined as in the previous code

# # Calculate the total number of batches
# total_batches = len(dataloader)
# print(f"Total number of batches: {total_batches}")

# # Iterate through the DataLoader
# for i, images in enumerate(dataloader):
#     print(f"Batch {i + 1}: {images[0].shape}")  # Print the shape of the image tensor

# for i, images in enumerate(dataloader):
#     if i < 10:  # Only take the first 10 images
#         print(images[0])
#     else:
#         break

# def main():
    
    # test_text_encoder()
    # create_context_vision_encoder()
    # target_vision_encoder()
    # test_crosser()

# create a dataloader class that has function return batches, each batch has input of batch_size and contains:
# - image
# - text (caption)
# - context_blocks (patches) of the image: List of List of Tensors
# - target_blocks (patches) of the image: List of Tensors

import os
import random
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import json
from tqdm import tqdm

from src.masks.custom_multiblock import MultiBlock

class ImageTextDataset(Dataset):
    def __init__(
        self, 
        image_path, 
        caption_path,
        batch_size,
        img_size,
        patch_size,
        _hidden_ratio,
        device='cuda',
        shuffle=False,
        block_scale=(0.05, 0.1),
        block_aspect_ratio=(0.75, 1.5),
        max=None | int,
        transform=None, 
        collator=None,
    ):
        self.batch_size = batch_size
        self.img_size = img_size
        self.patch_size = patch_size
        self._hidden_ratio = _hidden_ratio
        self.device = device
        
        assert img_size % patch_size == 0, f"img_size {img_size} is not divisible by patch_size {patch_size}"
        
        self.n_patches = (img_size // patch_size) ** 2
        self.all_patches = set(range(self.n_patches))
        
        self.folder_path = image_path
        with open(caption_path, 'r') as f:
            self.caption_dict = json.load(f) # caption_dict[filename] -> list[caption]
            
        self.transform = transform
        self.image_filenames = [f for f in os.listdir(image_path) if f.endswith(('.png', '.jpg', '.jpeg'))]
        
        if max:
            self.image_filenames = self.image_filenames[:max]
        
        if shuffle:
            random.shuffle(self.image_filenames)

        self.multiblock = MultiBlock(
            block_scale=block_scale,
            block_aspect_ratio=block_aspect_ratio,
            device='cuda',
        )
        self.collator = collator
        
    # def generate_random_predict_masks(self, num_predict_patches: int, current_batch_size: int):
    #     predict_masks = []
    #     for _ in range(current_batch_size):
    #         random_patches = random.sample(
    #             list(self.all_patches),
    #             num_predict_patches
    #         )
    #         predict_masks.append(random_patches)
    #     return torch.tensor(predict_masks).to(self.device)

    # def generate_context_masks(self, predict_masks: torch.Tensor):
    #     context_masks = []
    #     for i in range(len(predict_masks)):
    #         remaining_patches = list(self.all_patches - set(predict_masks[i].tolist()))
    #         context_masks.append(remaining_patches)
    #     return torch.tensor(context_masks).to(self.device)     

    def generate_random_predict_masks(self, num_predict_patches: int, current_batch_size: int):
        predict_masks = []
        max_len = 0
        
        # Generate masks and track the maximum length
        for _ in range(current_batch_size):
            random_patches = random.sample(
                list(self.all_patches),
                num_predict_patches
            )
            predict_masks.append(random_patches)
            max_len = max(max_len, len(random_patches))
        
        # Pad all masks to the maximum length
        for i in range(len(predict_masks)):
            padding_len = max_len - len(predict_masks[i])
            predict_masks[i].extend([-1] * padding_len)  # Use -1 as padding
        
        return torch.tensor(predict_masks).to(self.device)

    def generate_context_masks(self, predict_masks: torch.Tensor):
        context_masks = []
        max_len = 0
        
        # Generate context masks and track the maximum length
        for i in range(len(predict_masks)):
            remaining_patches = list(self.all_patches - set(predict_masks[i].tolist()))
            context_masks.append(remaining_patches)
            max_len = max(max_len, len(remaining_patches))
        
        # Pad all context masks to the maximum length
        for i in range(len(context_masks)):
            padding_len = max_len - len(context_masks[i])
            context_masks[i].extend([-1] * padding_len)  # Use -1 as padding
        
        return torch.tensor(context_masks).to(self.device)


    def get_image(self, idx):
        img_path = os.path.join(self.folder_path, self.image_filenames[idx])
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        return image

    def get_text(self, idx):
        filename = self.image_filenames[idx]
        captions = self.caption_dict[filename]
        return random.choice(captions)
    
    def __len__(self):
        import math
        return int(math.ceil(len(self.image_filenames) / self.batch_size))
    
    def __iter__(self):
        self.current_idx = 0
        
        while self.current_idx < len(self.image_filenames):
            images = []
            captions = []

            for idx in range(self.current_idx, self.current_idx + self.batch_size):
                if idx >= len(self.image_filenames): 
                    break
                images.append(self.get_image(idx))
                captions.append(self.get_text(idx))
                
            current_batch_size = len(captions)
            
            context_masks, predict_masks = self.multiblock(current_batch_size)
            
            self.current_idx += self.batch_size

            yield (
                torch.stack(images).to(self.device),
                captions,
                context_masks,
                predict_masks,
            )
            
# collator = MaskCollator()
# dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, collate_fn=collator, num_workers=0)
# data = dataset.batching(10)
# print(data[0])

if __name__ == "__main__":

    SIZE = 224
    PATCH_SIZE = 16
    HIDDEN_RATIO = (0.4, 0.5)

    BATCH_SIZE = 21

    #usage
    dataset = ImageTextDataset(
        image_path='src/datasets/train', 
        caption_path='src/datasets/annotations/filename_caption_dict.json', 
        batch_size=BATCH_SIZE,
        img_size=SIZE,
        patch_size=PATCH_SIZE,
        _hidden_ratio=HIDDEN_RATIO,
        max=100,
        transform=transforms.Compose(
            [
                transforms.Resize((SIZE, SIZE)), 
                transforms.ToTensor()
            ]
        )
    )
    
    for images, captions, context_masks, predict_masks in tqdm(dataset, desc="Loading Dataset"):
        print(f"{images.shape=}")
        print(f"{images[0][0][0]=}")
        print(f"{len(captions)=}")
        print(f"{captions[0]=}")
        print(f"{context_masks.shape=}")
        print(f"{predict_masks.shape=}")
        print()
