import os
import sys
from PIL import Image
import torch
import torchvision.transforms.functional as F
from torchvision import transforms

# Folder to save tensor files
tensor_folder = "vqa_tensor_448"

# Transformation to apply to each image
transform = transforms.Compose([transforms.ToTensor()])

# Function to process and save an image as a tensor
def process_and_save_image(image_path, save_folder=tensor_folder, transform=transform):
    # Derive the save path
    image_file = os.path.basename(image_path)
    save_path = os.path.join(save_folder, os.path.splitext(image_file)[0][-12:] + ".pt")

    # Skip if the tensor file already exists
    if os.path.exists(save_path):
        return

    # Open, transform, and resize the image
    image = Image.open(image_path).convert("RGB")
    tensor = transform(image).unsqueeze(0)  # Add batch dimension
    tensor = F.resize(tensor, (448, 448))  # Resize
    torch.save(tensor.squeeze(0), save_path)  # Save without batch dimension
    
# Main function
def main():
    # Get folder path and start index from command-line arguments
    if len(sys.argv) != 3:
        print("Usage: python to_tensor.py <folder_path> <start_index>")
        return

    folder_path = sys.argv[1]
    start_index = int(sys.argv[2])
    batch_size = 100000  # Number of images to process in this batch
    final_index = start_index + batch_size

    # Get all image file paths
    image_files = sorted(
        [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.lower().endswith((".png", ".jpg", ".jpeg"))]
    )

    # Determine the range to process
    total_images = len(image_files)
    
    if start_index >= total_images:
        print(f"Start index {start_index} exceeds the number of images ({total_images}).")
        return

    end_index = min(final_index, total_images)  # Ensure we don't exceed the folder size

    print(f"Processing images {start_index} to {end_index} of {total_images}...")

    # Process images in the specified range
    for idx in range(start_index, end_index):
        real_idx = idx + 1
        image_path = image_files[idx]
        process_and_save_image(image_path)
        print(f"\rProcessed {real_idx}/{total_images} {image_path}", end='')

    print("Processing complete.")

if __name__ == "__main__":
    main()
