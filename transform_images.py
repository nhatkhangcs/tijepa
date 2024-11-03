import os
import torch
from PIL import Image
from torchvision import transforms
import torch.nn.functional as F

# Configuration
SOURCE_DIR = 'src/datasets/train'  # Folder containing images
IMAGE_SIZE = 224  # Replace this with the size you need
BATCH_SIZE = 16  # How many images to process in parallel (useful for GPU)
DEST_DIR = f'src/datasets/train-tensors-{IMAGE_SIZE}'  # Folder to save transformed tensors

# Create the transformation pipeline
transform = transforms.Compose([
    transforms.ToTensor()  # Convert PIL image to PyTorch tensor
])

# Ensure the destination folder exists
os.makedirs(DEST_DIR, exist_ok=True)

# Function to load and transform an image, then save it as a .pt file
def process_and_save_image(image_path, save_path, device):
    # Open the image
    image = Image.open(image_path).convert("RGB")
    
    # Apply the transformation (on CPU)
    tensor = transform(image).unsqueeze(0)  # Add batch dimension
    
    # Move tensor to GPU (or CUDA-enabled device)
    tensor = tensor.to(device)

    # Optionally, resize the image on GPU (if needed)
    tensor = F.interpolate(tensor, size=(IMAGE_SIZE, IMAGE_SIZE))
    
    # Remove batch dimension and save tensor as .pt
    torch.save(tensor.squeeze(0), save_path)

# Check if GPU is available, otherwise fallback to CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Get a list of image files in the source directory
image_files = [f for f in os.listdir(SOURCE_DIR) if f.endswith(('.jpg', '.jpeg', '.png'))]

# Process the images in batches
for i, image_file in enumerate(image_files[:1000]):
    image_path = os.path.join(SOURCE_DIR, image_file)
    save_path = os.path.join(DEST_DIR, os.path.splitext(image_file)[0] + '.pt')
    
    print(f"Processing {image_file} ({i+1}/{len(image_files)})")
    
    # Process and save the image tensor
    process_and_save_image(image_path, save_path, device)

print("All images processed and saved as tensors.")
