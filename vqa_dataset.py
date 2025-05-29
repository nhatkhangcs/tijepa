import torch
import random
import os
from src.models.modules import text_encoder_model, vision_encoder, crosser_module
from src.utils.tensors import apply_masks, repeat_interleave_batch
import torchvision.transforms.functional as F
import torch.optim as optim

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

import os
import random
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import json
from tqdm import tqdm

class VQADataset(Dataset):
    def __init__(
        self,
        batch_size,
        img_size,
        shuffle=False,
        max=None,
        max_val=5000,
        val_batch_size=100,
        transform=transforms.Compose(
            [
                transforms.ToTensor()
            ]
        ), 
        collator=None,
    ):
        self.batch_size = batch_size
        self.img_size = img_size
        self.device = "cuda:0"
        self.val_batch_size = val_batch_size
        
        self.tensor_folder = f"vqa_dataset/vqa_tensor_{img_size}" 
        
        with open(os.path.join("vqa_dataset", "combined_data_train.json"), 'r') as f:
            self.train_dict = json.load(f)
            print(f"Loaded {len(self.train_dict)=}")
            
        with open(os.path.join("vqa_dataset", "combined_data_val.json"), 'r') as f:
            self.val_dict = json.load(f)
            print(f"Loaded {len(self.val_dict)=}")
            
        with open(os.path.join("vqa_dataset", "data_test.json"), 'r') as f:
            self.test_dict = json.load(f)
            print(f"Loaded {len(self.test_dict)=}")

        with open(os.path.join("vqa_dataset", "answer_mapping.json"), 'r') as f:
            self.mapper = json.load(f)
            print(f"Loaded {len(self.mapper)=}")
            self.A3129 = list(self.mapper.keys())
            print(f"Loaded {len(self.A3129)=}")
        
        with open(os.path.join("vqa_dataset", "answer_reverse_mapping.json"), 'r') as f:
            self.remapper = json.load(f)
            self.remapper = [v for k, v in self.remapper.items()]
            print(f"Loaded {len(self.remapper)=}")

        self.train_dict = [
            row for row in self.train_dict if row['multiple_choice_answer'] in self.A3129
        ]
        print(f"Filtered {len(self.train_dict)=}")
        self.val_dict = [
            row for row in self.val_dict if row['multiple_choice_answer'] in self.A3129
        ]
        
        print(f"Filtered {len(self.val_dict)=}")

        """
        {'questions': 'What is this photo taken looking through?',
         'question_type': 'what is this',
         'multiple_choice_answer': 'net',
         'answers': [{'answer': 'net', 'answer_confidence': 'maybe', 'answer_id': 1},
          {'answer': 'net', 'answer_confidence': 'yes', 'answer_id': 2},
          {'answer': 'net', 'answer_confidence': 'yes', 'answer_id': 3},
          {'answer': 'netting', 'answer_confidence': 'yes', 'answer_id': 4},
          {'answer': 'net', 'answer_confidence': 'yes', 'answer_id': 5},
          {'answer': 'net', 'answer_confidence': 'yes', 'answer_id': 6},
          {'answer': 'mesh', 'answer_confidence': 'maybe', 'answer_id': 7},
          {'answer': 'net', 'answer_confidence': 'yes', 'answer_id': 8},
          {'answer': 'net', 'answer_confidence': 'yes', 'answer_id': 9},
          {'answer': 'net', 'answer_confidence': 'yes', 'answer_id': 10}],
         'image_id': 458752,
         'answer_type': 'other',
         'question_id': 458752000}
        """
            
        self.transform = transform
        
        if max:
            self.train_dict = self.train_dict[:max]
        
        if shuffle:
            random.shuffle(self.train_dict)

        self.max_val = max_val
        self.collator = collator

    def get_V(self, idx, split="train"):
        """ Load tensorized image from disk. """
        set = self.train_dict if split == "train" else self.val_dict if split == "val" else self.test_dict
        tensor_path = os.path.join(
            self.tensor_folder, 
            f"{set[idx]['image_id']:012d}.pt"  # Format as 12-digit number with leading zeros
        )
        return torch.load(tensor_path, map_location=self.device)

    def get_Q(self, idx, split="train"):
        """ Get a random caption for a given image. """
        set = self.train_dict if split == "train" else self.val_dict if split == "val" else self.test_dict
        return set[idx]['questions']

    def get_A(self, idx, split="train"):
        set = self.train_dict if split == "train" else self.val_dict
        return set[idx]['multiple_choice_answer']
    
    def get_ID(self, idx):
        # Just for test
        return self.test_dict[idx]['question_id']
        
    def __len__(self):
        """ Number of batches in the dataset. """
        return (len(self.train_dict) + self.batch_size - 1) // self.batch_size

    def __iter__(self):
        """ Iterator to yield batches of images and captions. """
        random.shuffle(self.train_dict)
            
        self.current_idx = 0
        while self.current_idx < len(self.train_dict):
            batch_indices = range(self.current_idx, min(self.current_idx + self.batch_size, len(self.train_dict)))
            images = [self.get_V(i) for i in batch_indices]
            questions = [self.get_Q(i) for i in batch_indices]
            answers = [self.get_A(i) for i in batch_indices]

            self.current_idx += self.batch_size

            yield torch.stack(images), questions, [self.mapper[ans] for ans in answers]

    def iter_val(self):
        """ Iterator to yield batches of images and captions. """
            
        self.current_idx = 0
        while self.current_idx < self.max_val:
            batch_indices = range(self.current_idx, min(self.current_idx + self.val_batch_size, self.max_val))
            images = [self.get_V(i, 'val') for i in batch_indices]
            questions = [self.get_Q(i, 'val') for i in batch_indices]
            answers = [self.get_A(i, 'val') for i in batch_indices]

            self.current_idx += self.val_batch_size

            yield torch.stack(images), questions, [self.mapper[ans] for ans in answers]

    def iter_test(self):
        """ Iterator to yield batches of images and captions. """
            
        self.current_idx = 0
        while self.current_idx < len(self.test_dict):
            batch_indices = range(self.current_idx, min(self.current_idx + self.val_batch_size, len(self.test_dict)))
            images = [self.get_V(i, 'val') for i in batch_indices]
            questions = [self.get_Q(i, 'val') for i in batch_indices]
            ids = [self.get_ID(i) for i in batch_indices]
            
            self.current_idx += self.val_batch_size

            yield torch.stack(images), questions, ids

            
# collator = MaskCollator()
# dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, collate_fn=collator, num_workers=0)
# data = dataset.batching(10)
# print(data[0])

if __name__ == "__main__":

    SIZE = 448
    BATCH_SIZE = 1

    dataset = VQADataset(
        batch_size=BATCH_SIZE,
        img_size=SIZE,
        shuffle=False,
        max=1,
    )
    
    for images, questions, answers in tqdm(dataset, desc="Loading Dataset"):
        print(f"{images.shape=}")
        print(f"{images=}")
        print(f"{images[0][0][0]=}")
        print(f"{len(questions)=}")
        print(f"{questions=}")
        print(f"{len(answers)=}")
        print(f"{answers=}")
        print([dataset.remapper[str(ans_id)] for ans_id in answers])
