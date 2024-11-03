from eval_on_mvsa import encode_dataset, train_simple_linear_module, MVSA
import os
import sys

def to_tensor():
    import torchvision.transforms.functional as F
    from torchvision import transforms

    # Tensorize by set tensor_folder=None
    MVSA(
        batch_size=1, 
        img_size=448,
        device='cuda:0',
        transform=transforms.Compose(
            [
                transforms.ToTensor()
            ]
        ),
        tensor_folder=None
    )
    
def encode_all():
    os.makedirs("trains/SMALL-A100-448-10k-OBS-SCHEDULER/tensors")
    os.makedirs("trains/SMALL-A100-448-10k-OBS-SCHEDULER/300-epoch-target")
    os.makedirs("trains/SMALL-A100-448-10k-OBS-SCHEDULER/300-epoch-context")

    encode_dataset(
        checkpoint_path="trains/SMALL-A100-448-10k-OBS-SCHEDULER/epoch-300.pt",
        batch_size=200,
        device='cuda:0',
        save_path="trains/SMALL-A100-448-10k-OBS-SCHEDULER/300-epoch-target",
        crosser_type='target',
        tensor_folder='src/datasets/mvsa-tensor-448',
    )
    encode_dataset(
        checkpoint_path="trains/SMALL-A100-448-10k-OBS-SCHEDULER/epoch-300.pt",
        batch_size=200,
        device='cuda:0',
        save_path="trains/SMALL-A100-448-10k-OBS-SCHEDULER/300-epoch-context",
        crosser_type='context',
        tensor_folder='src/datasets/mvsa-tensor-448',
    )

def train():
    train_simple_linear_module(
        save_path="trains/SMALL-A100-448-10k-OBS-SCHEDULER/300-epoch-target",
        hidden_size=768,
        batch_size=512,
        epochs=50,
        device='cuda:0',
        seed=100
    )
    train_simple_linear_module(
        save_path="trains/SMALL-A100-448-10k-OBS-SCHEDULER/300-epoch-context",
        hidden_size=768,
        batch_size=512,
        epochs=50,
        device='cuda:0',
        seed=100
    )

if __name__ == "__main__":
    encode_all()
    train()
