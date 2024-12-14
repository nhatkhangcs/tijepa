import random
from src.models.modules import SimpleLinear
import os
import torch
from PIL import Image
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
from torchvision import transforms
import torchvision.transforms.functional as F
import time

from load_tijepa import load, inference
from load_tijepa_448 import load_448, inference_448

from tqdm import tqdm

from metrics import calculate_metrics_from_logits

class MVSA:
    MVSA_SINGLE_PATH = "src/datasets/mvsa/mvsa_single"
    MVSA_MULTIPLE_PATH = "src/datasets/mvsa/mvsa_multiple"

    def __init__(
            self, 
            batch_size, 
            img_size,
            device = 'cuda:0',
            transform = None,
            tensor_folder=None
        ):
        self.mvsa_dict = {
            'single': {
                # 'positive': [],
                # 'neutral': [],
                # 'negative': [],
            },
            'multiple': {

            }
        }
        self.batch_size = batch_size
        self.img_size = img_size
        self.device = device
        self.transform = transform
        self.tensor_folder = tensor_folder

        for cls in os.listdir(MVSA.MVSA_SINGLE_PATH):
            images_list = []
            local_path = os.path.join(MVSA.MVSA_SINGLE_PATH, cls, 'image')
            for file_name in os.listdir(local_path):
                images_list.append(os.path.join(local_path, file_name).replace('/', '='))
            self.mvsa_dict['single'][cls] = images_list
        
        for cls in os.listdir(MVSA.MVSA_MULTIPLE_PATH):
            images_list = []
            local_path = os.path.join(MVSA.MVSA_MULTIPLE_PATH, cls, 'image')
            for file_name in os.listdir(local_path):
                images_list.append(os.path.join(local_path, file_name).replace('/', '='))
            self.mvsa_dict['multiple'][cls] = images_list
        
        # for cls in self.mvsa_dict['single']:
        #     print(f"{cls}: {len(self.mvsa_dict['single'][cls])=}")
        # for cls in self.mvsa_dict['multiple']:
        #     print(f"{cls}: {len(self.mvsa_dict['multiple'][cls])=}")

        self.dataset = []
        for cls in self.mvsa_dict['single']:
            images_list = self.mvsa_dict['single'][cls]
            self.dataset.extend([(img, cls) for img in images_list])
        
        for cls in self.mvsa_dict['multiple']:
            images_list = self.mvsa_dict['multiple'][cls]
            self.dataset.extend([(img, cls) for img in images_list])

        print(f"Total dataset after init: {len(self.dataset)=}")

        self.preload_images()
        

    def process_and_save_image(self, image_file):
        save_path = os.path.join(self.tensor_folder, os.path.splitext(image_file)[0] + '.pt')
        
        if os.path.exists(save_path):
            print("skip")
            return
        
        # Open and process the image
        image = Image.open(image_file.replace('=', '/')).convert("RGB")
        tensor = self.transform(image).unsqueeze(0)  # Apply transformations and add batch dimension
        tensor = F.resize(tensor, (self.img_size, self.img_size))  # Resize
        torch.save(tensor.squeeze(0), save_path)

    def preload_images(self):
        """ Preload images as tensors and save them to disk as .pt files (if not already done). """
        if self.tensor_folder is not None and os.path.exists(self.tensor_folder):
            return

        self.tensor_folder = f"src/datasets/mvsa-tensor-448"
        os.makedirs(self.tensor_folder, exist_ok=True)

        for i, (image_file, cls) in enumerate(self.dataset):
            try:
                self.process_and_save_image(image_file)
            except Exception as e:
                print(f"FAIL {image_file}: {e}")
            print(f"Processing {image_file} ({i+1}/{len(self.dataset)})", end='\r')
                            
        print(
            f"{len(os.listdir(self.tensor_folder))}/{len(self.dataset)}"
            " images processed and saved as tensors."
        ) 

    def shuffle(self, seed = 69):
        random.seed(seed)
        random.shuffle(self.dataset)
        
    def split(self, ratio = (0.8, 0.1, 0.1)):
        train_len = int(len(self.dataset) * ratio[0])
        val_len = int(len(self.dataset) * ratio[1])
        test_len = len(self.dataset) - train_len - val_len

        print(f"Split to {train_len=}, {val_len=}, {test_len=}")

        self.train_set = self.dataset[:train_len]
        self.val_set = self.dataset[train_len:train_len + val_len]
        self.test_set = self.dataset[train_len + val_len:]

    def __len__(self): # length of the dataset
        return len(self.dataset) // self.batch_size + 1

    def upsampling(self):
        """
        Upsample the dataset to balance the number of samples across classes.
        Assumes `self.train_set` is a list of tuples (image_path, class).
    
        Returns:
            None: Updates `self.train_set` in-place with upsampled data.
        """
        from collections import Counter
    
        # Count the number of samples for each class
        class_counts = Counter([cls for _, cls in self.train_set])
        max_count = max(class_counts.values())  # Find the maximum class count
    
        # Group samples by class
        class_to_samples = {cls: [] for cls in class_counts}
        for image_path, cls in self.train_set:
            class_to_samples[cls].append((image_path, cls))
    
        # Upsample each class to have the same number of samples as the majority class
        upsampled_train_set = []
        for cls, samples in class_to_samples.items():
            num_samples = len(samples)
            if num_samples < max_count:
                # Randomly duplicate samples to reach the maximum count
                additional_samples = random.choices(samples, k=max_count - num_samples)
                upsampled_train_set.extend(samples + additional_samples)
            else:
                upsampled_train_set.extend(samples)
    
        # Shuffle the upsampled dataset for randomness
        random.shuffle(upsampled_train_set)
    
        # Update the train_set with the upsampled dataset
        self.train_set = upsampled_train_set

        # Count again
        class_counts = Counter([cls for _, cls in self.train_set])
        print("After upsampling: ", class_counts)
    
    def __iter__(self): # iter on the dataset
        self.current_idx = 0

        while self.current_idx < len(self.dataset):
            images = []
            captions = []
            images_paths = []

            # Load a batch of data
            for idx in range(self.current_idx, self.current_idx + self.batch_size):
                if idx >= len(self.dataset):
                    break

                try:
                    # Load image
                    image_path, _ = self.dataset[idx]
                    text_path = image_path.replace('image', 'text').replace('jpg', 'txt')
                    
                    image = torch.load(os.path.join(self.tensor_folder, image_path.replace('jpg', 'pt')))
                    
                    # Load text
                    with open(text_path.replace('=', '/'), 'r', encoding='unicode_escape') as file:
                        text = file.read()

                except Exception as e: 
                    continue

                images.append(image)
                captions.append(text)
                images_paths.append(image_path)
            
            self.current_idx += self.batch_size

            # Stack images into a single tensor for the batch and move to the appropriate device
            yield (
                torch.stack(images).to(self.device),  # Stack images into a batch tensor
                captions,  # Captions can be processed later
                images_paths,  # Image paths can be used for debugging
            )


    
    def iter_path(self, split='train'):
        if split == 'train':
            data = self.train_set
        elif split == 'val':
            data = self.val_set
        elif split == 'test':
            data = self.test_set

        # Shuffle the data
        random.shuffle(data)

        self.current_idx = 0

        while self.current_idx < len(data):
            images_paths = []
            class_labels = []

            # Load a batch of data
            for idx in range(self.current_idx, self.current_idx + self.batch_size):
                if idx >= len(data):
                    break

                try:
                    # Load image
                    image_path, cls = data[idx]

                    if cls == 'positive':
                        cls = 0
                    elif cls == 'neutral':
                        cls = 1
                    elif cls == 'negative':
                        cls = 2

                except Exception as e: 
                    continue

                images_paths.append(image_path)
                class_labels.append(cls)

            
            self.current_idx += self.batch_size

            # Stack images into a single tensor for the batch and move to the appropriate device
            yield (
                torch.tensor(class_labels).to(self.device),  # Convert class labels to a tensor
                images_paths,  # Image paths can be used for debugging
            )


def simple_linear_sentiment_module(hidden_size, num_classes):
    # Create a simple linear module
    return SimpleLinear(hidden_size, num_classes)

def encode_dataset(
        checkpoint_path,
        batch_size,
        device,
        save_path,
        crosser_type='target',
        tensor_folder=None,
    ):
    # Load the models
    text_encoder, vision_encoder, crosser = load_448(checkpoint_path, crosser_type)

    # Move the models to the device
    text_encoder = text_encoder.to(device)
    text_encoder.device = device
    vision_encoder = vision_encoder.to(device)
    crosser = crosser.to(device)

    # Set them to eval
    text_encoder.eval()
    vision_encoder.eval()
    crosser.eval()

    transform=transforms.Compose(
        [
            transforms.ToTensor()
        ]
    )

    # Create dataset
    ds = MVSA(
        batch_size = batch_size,
        img_size = 224,
        device = device,
        transform = transform,
        tensor_folder=tensor_folder
    )

    with torch.no_grad():
        # Encode the dataset
        with tqdm(ds, desc=f"Embedding pairs.") as pbar:
            for images, captions, images_paths in pbar:
                embeddings = inference_448(
                    images, captions, text_encoder, vision_encoder, crosser
                )
                for embedding, image_path in zip(embeddings, images_paths):
                    torch.save(embedding, os.path.join(save_path, image_path.replace('jpg', 'pt')))

            pbar.set_postfix({
                'MEM': torch.cuda.max_memory_allocated() / 1024.**3,
                'len': len(images_paths),
            })

from src.utils.saving import Saver

def train_simple_linear_module(
        save_path,
        hidden_size,
        device='cuda:0',
        lr=1e-3,
        epochs=5,
        batch_size=500,
        seed=69
    ):

    """
    {
        "accuracy": accuracy,
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "macro_f1": macro_f1,
        "weighted_precision": weighted_precision,
        "weighted_recall": weighted_recall,
        "weighted_f1": weighted_f1,
        "per_class_precision": precision.tolist(),
        "per_class_recall": recall.tolist(),
        "per_class_f1": f1_score.tolist(),
    }
    """

    saver = Saver(
        metrics=[
            'loss',
            
            "tr-accuracy",
            "tr-macro_precision",
            "tr-macro_recall",
            "tr-macro_f1",
            "tr-weighted_precision",
            "tr-weighted_recall",
            "tr-weighted_f1",
            "tr-per_class_precision",
            "tr-per_class_recall",
            "tr-per_class_f1",

            "v-accuracy",
            "v-macro_precision",
            "v-macro_recall",
            "v-macro_f1",
            "v-weighted_precision",
            "v-weighted_recall",
            "v-weighted_f1",
            "v-per_class_precision",
            "v-per_class_recall",
            "v-per_class_f1",

            "t-accuracy",
            "t-macro_precision",
            "t-macro_recall",
            "t-macro_f1",
            "t-weighted_precision",
            "t-weighted_recall",
            "t-weighted_f1",
            "t-per_class_precision",
            "t-per_class_recall",
            "t-per_class_f1",
        ],
        folder_name='MVSA',
        **{
            "save_path": save_path
        }
    )

    # Create dataset
    ds = MVSA(
        batch_size = batch_size,
        img_size = 224,
        device = device,
        tensor_folder=f"src/datasets/mvsa-tensor"
    )

    ds.shuffle(seed=seed)
    ds.split()

    print(f"{len(ds.train_set)=}")
    print(f"{len(ds.val_set)=}")
    print(f"{len(ds.test_set)=}")

    ds.upsampling()

    # Create a simple linear module
    linear_module = simple_linear_sentiment_module(hidden_size, 3).to(device)

    # Define a simple training loop
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        linear_module.parameters(), 
        lr=lr
    )
    # optimizer = torch.optim.SGD(
    #     linear_module.parameters(), 
    #     lr=lr, 
    #     momentum=0.9
    # )
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.9)
    
    # Train the model
    for epoch in range(epochs):
        # Train the model
        linear_module.train()

        total_loss = 0
        ALL_PREDICTED_LOGITS = torch.empty(0, 3).to(device)
        ALL_GROUND_TRUTH = torch.empty(0, dtype=torch.long).to(device)

        with tqdm(ds.iter_path('train'), desc=f"Epoch {epoch+1}/{epochs}") as pbar:
            for class_labels, images_paths in pbar:
                # Zero the gradients
                optimizer.zero_grad()

                # Check if the images are in the save_path
                for idx, image_path in enumerate(images_paths):
                    if not os.path.exists(os.path.join(save_path, image_path.replace('jpg', 'pt'))):
                        saver.log(f"{image_path} not found")

                        images_paths.pop(idx)
                        class_labels = torch.cat([
                            class_labels[:idx],
                            class_labels[idx+1:]
                        ], dim=0)

                # Embed
                embeddings = torch.stack([
                    torch.load(os.path.join(save_path, image_path.replace('jpg', 'pt')))
                    for image_path in images_paths
                ]).to(device)

                # Predict
                predictions = linear_module(embeddings)
                saver.log(f"{predictions[:5]=}")
                saver.log(f"{predictions.argmax(dim=1)[:5]}")
                
                class_labels = torch.tensor(class_labels, dtype=torch.long).to(device)

                # Calculate loss
                loss = criterion(predictions, class_labels)

                ALL_PREDICTED_LOGITS = torch.cat((ALL_PREDICTED_LOGITS, predictions), dim=0)
                ALL_GROUND_TRUTH = torch.cat((ALL_GROUND_TRUTH, class_labels), dim=0)
            
                # Backpropagate
                loss.backward()
                optimizer.step()
    
                # Optionally, print or log the learning rate
                current_lr = scheduler.get_last_lr()[0]

                # Calculate accuracy
                total_loss += loss.item()
                
                metrics = calculate_metrics_from_logits(ALL_PREDICTED_LOGITS, ALL_GROUND_TRUTH)
                
                saver.update_metric(
                    {
                        'loss': loss.item(),
                        'tr-accuracy': metrics['accuracy'],
                        'tr-weighted_precision': metrics['weighted_precision'],
                        'tr-weighted_recall': metrics['weighted_recall'],
                        'tr-weighted_f1': metrics['weighted_f1'],
                        "tr-per_class_precision": metrics['per_class_precision'],
                        "tr-per_class_recall": metrics['per_class_recall'],
                        "tr-per_class_f1": metrics['per_class_f1'],
                    }
                )
                saver.save_epoch(temp=True)

                pbar.set_postfix(
                    loss=loss.item(),
                    tr_accuracy=metrics['accuracy'],
                    tr_weighted_precision=metrics['weighted_precision'],
                    tr_weighted_recall=metrics['weighted_recall'],
                    tr_weighted_f1=metrics['weighted_f1'],
                    lr=current_lr,
                )

        scheduler.step()

        with torch.no_grad():
            # Validate the model
            linear_module.eval()
    
            total_loss = 0
            ALL_PREDICTED_LOGITS = torch.empty(0, 3).to(device)
            ALL_GROUND_TRUTH = torch.empty(0, dtype=torch.long).to(device)
    
            with tqdm(ds.iter_path('val'), desc=f"Validation") as pbar:
                for class_labels, images_paths in pbar:
    
                     # Check if the images are in the save_path
                    for idx, image_path in enumerate(images_paths):
                        if not os.path.exists(os.path.join(save_path, image_path.replace('jpg', 'pt'))):
                            print(f"{image_path} not found")
    
                            images_paths.pop(idx)
                            class_labels = torch.cat([
                                class_labels[:idx],
                                class_labels[idx+1:]
                            ], dim=0)
    
                    # Embed
                    embeddings = torch.stack([
                        torch.load(os.path.join(save_path, image_path.split('/')[-1].replace('jpg', 'pt')))
                        for image_path in images_paths
                    ]).to(device)
    
                    # Predict
                    predictions = linear_module(embeddings)
                    # print(f"{predictions[:5]=}")
                    # print(f"{predictions.argmax(dim=1)[:5]}")
    
                    class_labels = torch.tensor(class_labels, dtype=torch.long).to(device)
    
                    # Calculate loss
                    loss = criterion(predictions, class_labels)
    
                    ALL_PREDICTED_LOGITS = torch.cat((ALL_PREDICTED_LOGITS, predictions), dim=0)
                    ALL_GROUND_TRUTH = torch.cat((ALL_GROUND_TRUTH, class_labels), dim=0)
    
                    # Calculate accuracy
                    total_loss += loss.item()
    
                    metrics = calculate_metrics_from_logits(ALL_PREDICTED_LOGITS, ALL_GROUND_TRUTH)
                    
                    saver.update_metric(
                        {
                            'v-accuracy': metrics['accuracy'],
                            'v-weighted_precision': metrics['weighted_precision'],
                            'v-weighted_recall': metrics['weighted_recall'],
                            'v-weighted_f1': metrics['weighted_f1'],
                            "v-per_class_precision": metrics['per_class_precision'],
                            "v-per_class_recall": metrics['per_class_recall'],
                            "v-per_class_f1": metrics['per_class_f1'],
                        }
                    )
                    saver.save_epoch(temp=True)
    
                    pbar.set_postfix(
                        v_accuracy=metrics['accuracy'],
                        v_weighted_precision=metrics['weighted_precision'],
                        v_weighted_recall=metrics['weighted_recall'],
                        v_weighted_f1=metrics['weighted_f1'],
                    )
                    
            # Test the model
            linear_module.eval()
        
            total_loss = 0
            ALL_PREDICTED_LOGITS = torch.empty(0, 3).to(device)
            ALL_GROUND_TRUTH = torch.empty(0, dtype=torch.long).to(device)
        
            with tqdm(ds.iter_path('test'), desc=f"Testing") as pbar:
                for class_labels, images_paths in pbar:
                    # Check if the images are in the save_path
                    for idx, image_path in enumerate(images_paths):
                        if not os.path.exists(os.path.join(save_path, image_path.replace('jpg', 'pt'))):
                            print(f"{image_path} not found")
        
                            images_paths.pop(idx)
                            class_labels = torch.cat([
                                class_labels[:idx],
                                class_labels[idx+1:]
                            ], dim=0)
        
                    # Embed
                    embeddings = torch.stack([
                        torch.load(os.path.join(save_path, image_path.split('/')[-1].replace('jpg', 'pt')))
                        for image_path in images_paths
                    ]).to(device)
        
                    # Predict
                    predictions = linear_module(embeddings)
                    
                    class_labels = torch.tensor(class_labels, dtype=torch.long).to(device)
        
                    # Calculate loss
                    loss = criterion(predictions, class_labels)
    
                    ALL_PREDICTED_LOGITS = torch.cat((ALL_PREDICTED_LOGITS, predictions), dim=0)
                    ALL_GROUND_TRUTH = torch.cat((ALL_GROUND_TRUTH, class_labels), dim=0)
        
                    # Calculate accuracy
                    total_loss += loss.item()
        
                    metrics = calculate_metrics_from_logits(ALL_PREDICTED_LOGITS, ALL_GROUND_TRUTH)
                    
                    saver.update_metric(
                        {
                            't-accuracy': metrics['accuracy'],
                            't-weighted_precision': metrics['weighted_precision'],
                            't-weighted_recall': metrics['weighted_recall'],
                            't-weighted_f1': metrics['weighted_f1'],
                            "t-per_class_precision": metrics['per_class_precision'],
                            "t-per_class_recall": metrics['per_class_recall'],
                            "t-per_class_f1": metrics['per_class_f1'],
                        }
                    )
                    saver.save_epoch(temp=True)
    
                    pbar.set_postfix(
                        t_accuracy=metrics['accuracy'],
                        t_weighted_precision=metrics['weighted_precision'],
                        t_weighted_recall=metrics['weighted_recall'],
                        t_weighted_f1=metrics['weighted_f1'],
                    )

        saver.save_epoch()

        save_dict = {
            'linear_module': linear_module.state_dict(),
            'epoch': epoch + 1,
            'loss': loss
        }
        saver.save_checkpoint(save_dict, epoch=epoch+1)


    