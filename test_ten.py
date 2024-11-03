from src.models.modules import text_encoder_model, x_t2i_module, vit_predictor
from create_dataset import ImageTextDatasetA100
from torchvision import transforms

DEVICE_0 = "cuda"
TE = text_encoder_model(device=DEVICE_0)

dataset = ImageTextDatasetA100(
    image_path='src/datasets/train', 
    caption_path='src/datasets/annotations/filename_caption_dict.json', 
    batch_size=20000,
    img_size=224,
    patch_size=14,
    max=None,
    transform=transforms.Compose(
        [
            transforms.ToTensor()
        ]
    ), 
    block_scale=(0.15, 0.2), # originally block_scale=(0.15, 0.2),
    block_aspect_ratio=(0.75, 1.5),
    device=DEVICE_0,
    shuffle=False,
    tensor_folder="src/datasets/train-tensor-ALL",
)

L = []

for i, (images, captions, context_masks, predict_masks) in enumerate(dataset):
    res = TE.tokenizer(captions)
    lens = [len(ids) for ids in res['input_ids']]
    L += lens
    print(f"\r{i}/{len(dataset)}")

import matplotlib.pyplot as plt
# Define bin edges to group lengths by 5
bins = range(0, max(L) + 5, 5)

# Plot histogram
plt.figure(figsize=(10, 6))
plt.hist(L, bins=bins, edgecolor='black', alpha=0.7)
plt.xlabel('Sequence Length')
plt.ylabel('Frequency')
plt.title('Frequency of Sequence Lengths (Grouped by 100s)')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.xticks(bins)

# Save the plot to 'freq.png'
plt.savefig('freq.png', format='png')
