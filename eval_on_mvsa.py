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
                        cls = [1, 0, 0]
                    elif cls == 'neutral':
                        cls = [0, 1, 0]
                    elif cls == 'negative':
                        cls = [0, 0, 1]

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

    # Encode the dataset
    with tqdm(ds, desc=f"Embedding pairs.") as pbar:
        for images, captions, images_paths in pbar:
            with torch.no_grad():
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

    saver = Saver(
        metrics=[
            'loss', 
            'train-accuracy', 
            'train-precision',
            'train-recall',
            'train-f1',
            'val-accuracy',
            'val-precision',
            'val-recall',
            'val-f1',
            'test-accuracy',
            'test-precision',
            'test-recall',
            'test-f1',
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
    # ds.upsampling()
    ds.shuffle(seed=seed)
    ds.split()

    print(f"{len(ds.train_set)=}")
    print(f"{len(ds.val_set)=}")
    print(f"{len(ds.test_set)=}")

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
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.75)
    
    # Train the model
    for epoch in range(epochs):
        # Train the model
        linear_module.train()

        total_loss = 0
        total_samples = 0
        total_true_negative = 0
        total_true_positive = 0
        total_false_negative = 0
        total_false_positive = 0

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


                # Calculate loss
                loss = criterion(predictions, class_labels.argmax(dim=1))

                # Backpropagate
                loss.backward()
                optimizer.step()
    
                # Optionally, print or log the learning rate
                current_lr = scheduler.get_last_lr()[0]

                # Calculate accuracy
                total_loss += loss.item()
                

                pred_classes = predictions.argmax(dim=1)      # Get predicted class indices
                true_classes = class_labels.argmax(dim=1)     # Get true class indices (assuming one-hot encoded)

                # False Positives: Predicted class is 0, but true class is not 0
                total_false_positive += ((pred_classes == 0) & (true_classes != 0)).sum().item()
                saver.log("total_false_positive", total_false_positive)
                # False Negatives: Predicted class is not 0, but true class is 0
                total_false_negative += ((pred_classes != 0) & (true_classes == 0)).sum().item()
                saver.log("total_false_negative", total_false_negative)
                # True Positives: Predicted class is 0 and true class is also 0
                total_true_positive += ((pred_classes == 0) & (true_classes == 0)).sum().item()
                saver.log("total_true_positive", total_true_positive)
                # True Negatives: Predicted class is not 0 and true class is also not 0
                total_true_negative += ((pred_classes != 0) & (true_classes != 0)).sum().item()
                saver.log("total_true_negative", total_true_negative)
                
                total_samples += len(class_labels)
                saver.log("total_samples", total_samples)
                
                saver.update_metric(
                    {
                        'loss': total_loss / total_samples,
                        'train-accuracy': (total_true_positive + total_true_negative) / total_samples,
                        'train-precision': total_true_positive / (total_true_positive + total_false_positive),
                        'train-recall': total_true_positive / (total_true_positive + total_false_negative),
                        'train-f1': 2 * total_true_positive / (2 * total_true_positive + total_false_positive + total_false_negative),
                    }
                )
                saver.save_epoch(temp=True)

                pbar.set_postfix(
                    loss=total_loss / total_samples,
                    accuracy=(total_true_positive + total_true_negative) / total_samples,
                    precision=total_true_positive / (total_true_positive + total_false_positive),
                    recall=total_true_positive / (total_true_positive + total_false_negative),
                    f1=2 * total_true_positive / (2 * total_true_positive + total_false_positive + total_false_negative),
                    lr=current_lr
                )


        scheduler.step()
        
        # Validate the model
        linear_module.eval()

        total_loss = 0
        total_samples = 0
        total_true_negative = 0
        total_true_positive = 0
        total_false_negative = 0
        total_false_positive = 0

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

                # Calculate loss
                loss = criterion(predictions, class_labels.argmax(dim=1))

                # Calculate accuracy
                total_loss += loss.item()

                pred_classes = predictions.argmax(dim=1)      # Get predicted class indices
                true_classes = class_labels.argmax(dim=1)     # Get true class indices (assuming one-hot encoded)

                # False Positives: Predicted class is 0, but true class is not 0
                total_false_positive += ((pred_classes == 0) & (true_classes != 0)).sum().item()

                # False Negatives: Predicted class is not 0, but true class is 0
                total_false_negative += ((pred_classes != 0) & (true_classes == 0)).sum().item()

                # True Positives: Predicted class is 0 and true class is also 0
                total_true_positive += ((pred_classes == 0) & (true_classes == 0)).sum().item()

                # True Negatives: Predicted class is not 0 and true class is also not
                total_true_negative += ((pred_classes != 0) & (true_classes != 0)).sum().item()
                
                total_samples += len(class_labels)

                saver.update_metric(
                    {
                        'val-accuracy': (total_true_positive + total_true_negative) / total_samples,
                        'val-precision': total_true_positive / (total_true_positive + total_false_positive),
                        'val-recall': total_true_positive / (total_true_positive + total_false_negative),
                        'val-f1': 2 * total_true_positive / (2 * total_true_positive + total_false_positive + total_false_negative),
                    }
                )

                pbar.set_postfix(
                    loss=total_loss / total_samples,
                    accuracy=(total_true_positive + total_true_negative) / total_samples,
                    precision=total_true_positive / (total_true_positive + total_false_positive),
                    recall=total_true_positive / (total_true_positive + total_false_negative),
                    f1=2 * total_true_positive / (2 * total_true_positive + total_false_positive + total_false_negative),
                )
                
        # Test the model
        linear_module.eval()
    
        total_loss = 0
        total_samples = 0
        total_true_negative = 0
        total_true_positive = 0
        total_false_negative = 0
        total_false_positive = 0
    
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
    
                # Calculate loss
                loss = criterion(predictions, class_labels.argmax(dim=1))
    
                # Calculate accuracy
                total_loss += loss.item()
    
                pred_classes = predictions.argmax(dim=1)      # Get predicted class indices
                true_classes = class_labels.argmax(dim=1)
    
                # False Positives: Predicted class is 0, but true class is not 0
                total_false_positive += ((pred_classes == 0) & (true_classes != 0)).sum().item()
    
                # False Negatives: Predicted class is not 0, but true class is 0
                total_false_negative += ((pred_classes != 0) & (true_classes == 0)).sum().item()
    
                # True Positives: Predicted class is 0 and true class is also 0
                total_true_positive += ((pred_classes == 0) & (true_classes == 0)).sum().item()
    
                # True Negatives: Predicted class is not 0 and true class is also not
                total_true_negative += ((pred_classes != 0) & (true_classes != 0)).sum().item()
    
                total_samples += len(class_labels)
    
                saver.update_metric(
                    {
                        'test-accuracy': (total_true_positive + total_true_negative) / total_samples,
                        'test-precision': total_true_positive / (total_true_positive + total_false_positive),
                        'test-recall': total_true_positive / (total_true_positive + total_false_negative),
                        'test-f1': 2 * total_true_positive / (2 * total_true_positive + total_false_positive + total_false_negative),
                    }
                )
    
                pbar.set_postfix(
                    loss=total_loss / total_samples,
                    accuracy=(total_true_positive + total_true_negative) / total_samples,
                    precision=total_true_positive / (total_true_positive + total_false_positive),
                    recall=total_true_positive / (total_true_positive + total_false_negative),
                    f1=2 * total_true_positive / (2 * total_true_positive + total_false_positive + total_false_negative),
                )

        saver.save_epoch()


    
