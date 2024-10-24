import torch
import torchvision

imagenet_data = torchvision.datasets.ImageNet('src/datasets/imagenet/', split='val')
data_loader = torch.utils.data.DataLoader(imagenet_data,
                                          batch_size=4,
                                          shuffle=True,
                                          num_workers=3)