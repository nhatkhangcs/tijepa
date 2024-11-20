from eval_on_mvsa import encode_dataset, train_simple_linear_module, MVSA
import os
import sys

TRAIN = "SMALL-A100-448-600-10k-OBS-SCHEDULER"
EP = 600

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
    
    # os.makedirs(f"trains/{TRAIN}/tensors")
    # os.makedirs(f"trains/{TRAIN}/{EP}-epoch-target")
    # os.makedirs(f"trains/{TRAIN}/{EP}-epoch-context")

    encode_dataset(
        checkpoint_path=f"trains/{TRAIN}/epoch-{EP}.pt",
        batch_size=200,
        device='cuda:0',
        save_path=f"trains/{TRAIN}/{EP}-epoch-target",
        crosser_type='target',
        tensor_folder='src/datasets/mvsa-tensor-448',
    )
    encode_dataset(
        checkpoint_path=f"trains/{TRAIN}/epoch-{EP}.pt",
        batch_size=200,
        device='cuda:0',
        save_path=f"trains/{TRAIN}/{EP}-epoch-context",
        crosser_type='context',
        tensor_folder='src/datasets/mvsa-tensor-448',
    )

def train():
    train_simple_linear_module(
        save_path=f"trains/{TRAIN}/{EP}-epoch-target",
        hidden_size=768,
        batch_size=512,
        epochs=50,
        device='cuda:0',
        seed=100,
        # lr=1e-5
    )
    # train_simple_linear_module(
    #     save_path=f"trains/{TRAIN}/{EP}-epoch-context",
    #     hidden_size=768,
    #     batch_size=512,
    #     epochs=50,
    #     device='cuda:0',
    #     seed=100
    # )

if __name__ == "__main__":
                
            train()
