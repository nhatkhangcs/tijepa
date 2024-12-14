import os
import requests

# Define a function to download files
def download_file(url, save_path):
    """
    Downloads a file from the given URL and saves it to the specified path.
    """
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()  # Raise an error for bad responses
        total_size = int(response.headers.get('content-length', 0))
        chunk_size = 1024  # 1 KB

        print(f"Downloading {save_path}...")

        with open(save_path, 'wb') as file:
            for data in response.iter_content(chunk_size=chunk_size):
                file.write(data)

        print(f"Download complete: {save_path}")
    except requests.exceptions.RequestException as e:
        print(f"Error downloading {url}: {e}")

# URLs and desired filenames
files_to_download = {
    "train_annotation.zip": "https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Annotations_Train_mscoco.zip",
    "test_annotation.zip": "https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Annotations_Val_mscoco.zip"
}

# Directory to save the files
save_dir = "vqa_dataset"
os.makedirs(save_dir, exist_ok=True)

# Download each file
for filename, url in files_to_download.items():
    save_path = os.path.join(save_dir, filename)
    download_file(url, save_path)
