from eval_on_mvsa import encode_dataset, train_simple_linear_module
import os
import sys
def encode_all():
    # os.makedirs("tensors/P-ONLY")
    os.makedirs("tensors/P-ONLY/300-epoch-context")
    os.makedirs("tensors/P-ONLY/300-epoch-target")


    encode_dataset(
        checkpoint_path="trains/P-10k-1729506531/epoch-300.pt",
        batch_size=270,
        device='cuda:0',
        save_path="tensors/P-ONLY/300-epoch-context",
        crosser_type='context'
    )

    encode_dataset(
        checkpoint_path="trains/P-10k-1729506531/epoch-300.pt",
        batch_size=270,
        device='cuda:0',
        save_path="tensors/P-ONLY/300-epoch-target",
        crosser_type='target'
    )

def train():
    train_simple_linear_module(
        save_path="tensors/P-ONLY/300-epoch-target",
        hidden_size=768,
        batch_size=512,
        epochs=40,
        device='cuda:0'
    )

if __name__ == "__main__":
    train()
