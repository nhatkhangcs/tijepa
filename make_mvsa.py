from eval_on_mvsa import encode_dataset, train_simple_linear_module
import os
def encode_all():
    # os.makedirs("tensors/50k")
    # os.makedirs("tensors/50k/2-epoch")

    encode_dataset(
        checkpoint_path="trains/50k/epoch-2.pt",
        batch_size=270,
        device='cuda:0',
        save_path="tensors/50k/2-epoch"
    )

def train():
    train_simple_linear_module(
        save_path="tensors/50k/2-epoch",
        hidden_size=768,
        batch_size=6000,
        epochs=100,
        device='cuda:0'
    )

if __name__ == "__main__":
    encode_all()
