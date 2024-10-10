import random
from src.models.modules import SimpleLinear
import os
import torch
from PIL import Image
from torchvision import transforms

from load_tijepa import load, inference
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
        ):
        self.mvsa_dict = {
            'single': {},
            'multiple': {}
        }
        self.batch_size = batch_size
        self.img_size = img_size
        self.device = device
        self.transform = transform

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
        
    def upsampling(self):

        self.dataset = []

        max_len_single = max([len(self.mvsa_dict['single'][cls]) for cls in self.mvsa_dict['single']])
        max_len_multiple = max([len(self.mvsa_dict['multiple'][cls]) for cls in self.mvsa_dict['multiple']])

        print(f"Will upsampling `single` to {max_len_single=}")
        print(f"Will upsampling `multiple` to {max_len_multiple=}")

        for cls in self.mvsa_dict['single']:
            
            images_list = self.mvsa_dict['single'][cls]
            print(f"single: {cls}: original: {len(images_list)=}")

            while len(images_list) < max_len_single:
                images_list.extend(images_list)

            images_list = images_list[:max_len_single]
            print(f"single: {cls}: after upsampling: {len(images_list)=}")

            self.dataset.extend([(img, cls) for img in images_list])
        
        for cls in self.mvsa_dict['multiple']:

            images_list = self.mvsa_dict['multiple'][cls]
            print(f"multiple: {cls}: original: {len(images_list)=}")

            while len(images_list) < max_len_multiple:
                images_list.extend(images_list)

            images_list = images_list[:max_len_multiple]
            print(f"multiple: {cls}: after upsampling: {len(images_list)=}")

            self.dataset.extend([(img, cls) for img in images_list])

        print(f"Total dataset: {len(self.dataset)=}")

        all_tensors = os.listdir('tensors/tenk/10-epoch')
        for img, cls in self.dataset:
            if img.replace('jpg', 'pt') not in all_tensors:
                print(f"{img} not in tensors/tenk/10-epoch")
                self.dataset.remove((img, cls))


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
                    
                    image = Image.open(image_path.replace('=', '/')).convert('RGB')  # Open image and convert to RGB
                    image = self.transform(image)  # Apply the transformation
                    
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
    ):
    # Load the models
    text_encoder, vision_encoder, target_crosser = load(checkpoint_path)

    # Move the models to the device
    text_encoder = text_encoder.to(device)
    text_encoder.device = device
    vision_encoder = vision_encoder.to(device)
    target_crosser = target_crosser.to(device)

    # Set them to eval
    text_encoder.eval()
    vision_encoder.eval()
    target_crosser.eval()

    transform=transforms.Compose(
        [
            transforms.Resize((224, 224)), 
            transforms.ToTensor()
        ]
    )

    # Create dataset
    ds = MVSA(
        batch_size = batch_size,
        img_size = 224,
        device = device,
        transform = transform,
    )

    # Encode the dataset
    with tqdm(ds, desc=f"Embedding pairs.") as pbar:
        for images, captions, images_paths in pbar:
            with torch.no_grad():
                embeddings = inference(
                    images, captions, text_encoder, vision_encoder, target_crosser
                )
                for embedding, image_path in zip(embeddings, images_paths):
                    torch.save(embedding, os.path.join(save_path, image_path.replace('jpg', 'pt')))

            pbar.set_postfix({
                'MEM': torch.cuda.max_memory_allocated() / 1024.**3,
                'len': len(images_paths),
            })

def train_simple_linear_module(
        save_path,
        hidden_size,
        device='cuda:0',
        lr=1e-3,
        epochs=5,
        batch_size=500,
    ):

    # Create dataset
    ds = MVSA(
        batch_size = batch_size,
        img_size = 224,
        device = device,
    )
    ds.upsampling()
    ds.shuffle()
    ds.split()

    # Create a simple linear module
    linear_module = simple_linear_sentiment_module(hidden_size, 3).to(device)

    # Define a simple training loop
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        linear_module.parameters(), 
        lr=lr
    )
    
    # Train the model
    for epoch in range(epochs):
        # Train the model
        linear_module.train()

        total_loss = 0
        total_correct = 0
        total_samples = 0
        with tqdm(ds.iter_path('train'), desc=f"Epoch {epoch+1}/{epochs}") as pbar:
            for class_labels, images_paths in pbar:
                # Zero the gradients
                optimizer.zero_grad()

                # Embed
                embeddings = torch.stack([
                    torch.load(os.path.join(save_path, image_path.replace('jpg', 'pt')))
                    for image_path in images_paths
                ]).to(device)

                # Predict
                predictions = linear_module(embeddings)

                # Calculate loss
                loss = criterion(predictions, class_labels.argmax(dim=1))

                # Backpropagate
                loss.backward()
                optimizer.step()

                # Calculate accuracy
                total_loss += loss.item()
                total_correct += (predictions.argmax(dim=1) == class_labels.argmax(dim=1)).sum().item()
                total_samples += len(class_labels)

                pbar.set_postfix(
                    loss=total_loss / total_samples,
                    accuracy=total_correct / total_samples
                )
    
        # Validate the model
        linear_module.eval()

        total_loss = 0
        total_correct = 0
        total_samples = 0

        with tqdm(ds.iter_path('val'), desc=f"Validation") as pbar:
            for class_labels, images_paths in pbar:
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
                total_correct += (predictions.argmax(dim=1) == class_labels.argmax(dim=1)).sum().item()
                total_samples += len(class_labels)

                pbar.set_postfix(
                    loss=total_loss / total_samples,
                    accuracy=total_correct / total_samples
                )
    
    # Test the model
    linear_module.eval()

    total_loss = 0
    total_correct = 0
    total_samples = 0

    with tqdm(ds.iter_path('test'), desc=f"Testing") as pbar:
        for class_labels, images_paths in pbar:
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
            total_correct += (predictions.argmax(dim=1) == class_labels.argmax(dim=1)).sum().item()
            total_samples += len(class_labels)

            pbar.set_postfix(
                loss=total_loss / total_samples,
                accuracy=total_correct / total_samples
            )

    
