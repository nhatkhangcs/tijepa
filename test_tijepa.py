import torch
import random

from src.models.vision_transformer import vit_test, vit_predictor_test
from src.models.modules import text_encoder_model, vision_encoder, crosser_module
from src.utils.tensors import apply_masks, repeat_interleave_batch
import torch.nn.functional as F
import torch.optim as optim

import random
import string

def generate_random_string(length):
    """Generates a random string of the specified length.

    Args:
        length (int): The desired length of the string.

    Returns:
        str: The generated random string.
    """

    characters = string.ascii_letters + string.digits
    return ''.join(random.choice(characters) for _ in range(length))


def cross_entropy(preds, targets, reduction='none'):
    log_softmax = torch.nn.LogSoftmax(dim=-1)
    loss = (-targets * log_softmax(preds)).sum(1)
    if reduction == "none":
        return loss
    elif reduction == "mean":
        return loss.mean()
    
# Calculating the Loss
def contrastive_loss(text_embeddings, image_embeddings, temperature=1.0):
    logits = (text_embeddings @ image_embeddings.T) / temperature
    print(f"{logits=}")
    images_similarity = image_embeddings @ image_embeddings.T
    texts_similarity = text_embeddings @ text_embeddings.T
    targets = F.softmax(
        (images_similarity + texts_similarity) / 2 * temperature, dim=-1
    )
    print(f"{targets=}")
    texts_loss = cross_entropy(logits, targets, reduction='none')
    print(f"{texts_loss=}")
    images_loss = cross_entropy(logits.T, targets.T, reduction='none')
    print(f"{images_loss=}")
    loss =  (images_loss + texts_loss) / 2.0 # shape: (batch_size)
    return loss.mean()

import torch.nn.functional as F

def clip_loss(text_embeddings, image_embeddings, temperature=1.0):
    # Normalize embeddings for cosine similarity
    text_embeddings = F.normalize(text_embeddings, p=2, dim=-1)
    image_embeddings = F.normalize(image_embeddings, p=2, dim=-1)

    # Compute logits (cosine similarity scaled by temperature)
    logits = (text_embeddings @ image_embeddings.T) / temperature
    # print(f"{logits=}")

    # Contrastive targets: identity matrix for (text, image) pairs
    targets = torch.eye(text_embeddings.shape[0], device=text_embeddings.device)

    # Calculate the contrastive loss for both texts and images
    texts_loss = F.cross_entropy(logits, targets, reduction='none')
    images_loss = F.cross_entropy(logits.T, targets.T, reduction='none')

    # Final contrastive loss
    loss = (texts_loss + images_loss) / 2.0  # shape: (batch_size)
    
    return loss.mean()  # Reduce the mean for the final loss


def generate_random_predict_mask(batch_size, ALL_PATCHES, num_predict_patches=12, device='cuda'):
    predict_mask = []
    for _ in range(batch_size):
        # Randomly select 12 unique patches from ALL_PATCHES
        random_patches = random.sample(list(ALL_PATCHES), num_predict_patches)
        predict_mask.append(random_patches)
    return torch.tensor(predict_mask).to(device)

def generate_context_masks(ALL_PATCHES, predict_mask, device='cuda'):
    context_masks = []
    for i in range(len(predict_mask)):
        # Remaining patches are the ones not in predict_mask
        remaining_patches = list(ALL_PATCHES - set(predict_mask[i].tolist()))
        context_masks.append(remaining_patches)
    return torch.tensor(context_masks).to(device)

def test_text_encoder():
    text_encoder = text_encoder_model(device='cuda')
    
    input_texts = [
        "what is the capital of China?",
        "how to implement quick sort in python?",
        "Beijing",
        "sorting algorithms"
    ]
    
    embeddings, attention_mask = text_encoder(input_texts)
    print(f"{embeddings.shape=}")
    print(f"{attention_mask.shape=}")
    print(f"{attention_mask=}")
    
SIZE = 100
PATCH_SIZE = 20
HIDDEN_PATCH = 13

EMBED_DIM = 128
PREDICTOR_EMBED_DIM = 64
DEPTH = 6
NUM_HEADS = 4

BATCH_SIZE = 7

def test_context_vision_encoder(full=True):
    
    context_vision_encoder = vision_encoder(
        patch_size=PATCH_SIZE,
        embed_dim=EMBED_DIM,
        img_size=[SIZE],
    )
    
    NUM_PATCHES = context_vision_encoder.patch_embed.num_patches
    ALL_PATCHES = set(range(NUM_PATCHES))
    
    if full:
        context_masks = [
            torch.tensor(list(ALL_PATCHES))
            for _ in range(BATCH_SIZE)
        ]
    else:
        predict_mask = generate_random_predict_mask(BATCH_SIZE, list(ALL_PATCHES), num_predict_patches=HIDDEN_PATCH)
        context_masks = generate_context_masks(ALL_PATCHES, predict_mask)
    
    tensor = torch.randn(BATCH_SIZE, 3, SIZE, SIZE) # (B, C, W, H)
    
    context = context_vision_encoder(tensor, masks=context_masks)
    print(f"{context.shape=}")
    

def test_target_vision_encoder():
    
    target_vision_encoder = vision_encoder(
        patch_size=PATCH_SIZE,
        embed_dim=EMBED_DIM,
        img_size=[SIZE],
    )
    
    NUM_PATCHES = target_vision_encoder.patch_embed.num_patches
    ALL_PATCHES = set(range(NUM_PATCHES))

    tensor = torch.randn(BATCH_SIZE, 3, SIZE, SIZE) # (B, C, W, H)
    
    context = target_vision_encoder(tensor)
    print(f"{context.shape=}")
    
def test_crosser():
    
    crosser = crosser_module(
        text_embed_dim=768,
        vision_embed_dim=100,
        hidden_dim=EMBED_DIM,
        depth=DEPTH,
        num_heads=NUM_HEADS,
    )
    
    text_embeddings = torch.randn(BATCH_SIZE, 20, 768)
    vision_embeddings = torch.randn(BATCH_SIZE, 12, 100)
    
    T, V = crosser(text_embeddings, vision_embeddings)
    print(f"{T.shape=}")
    print(f"{V.shape=}")
    
    

def train(num_epochs=1, learning_rate=0.01):
    text_encoder = text_encoder_model(device='cuda')
    context_vision_encoder = vision_encoder(
        patch_size=PATCH_SIZE,
        embed_dim=EMBED_DIM,
        img_size=[SIZE],
    ).to('cuda')
    target_vision_encoder = vision_encoder(
        patch_size=PATCH_SIZE,
        embed_dim=EMBED_DIM,
        img_size=[SIZE],
    ).to('cuda')
    crosser = crosser_module(
        text_embed_dim=768,
        vision_embed_dim=EMBED_DIM,
        hidden_dim=EMBED_DIM,
        depth=DEPTH,
        num_heads=NUM_HEADS,
    ).to('cuda')
    
    NUM_PATCHES = context_vision_encoder.patch_embed.num_patches
    ALL_PATCHES = set(range(NUM_PATCHES))

    predictor = vit_predictor_test(
        embed_dim=EMBED_DIM,
        depth=DEPTH,
        num_heads=NUM_HEADS,
        predictor_embed_dim=PREDICTOR_EMBED_DIM,
        num_patches=NUM_PATCHES,
    ).to('cuda')
    
    # Optimizer
    optimizer = optim.Adam(
        list(text_encoder.parameters()) +
        list(context_vision_encoder.parameters()) +
        list(target_vision_encoder.parameters()) +
        list(crosser.parameters()) +
        list(predictor.parameters()), lr=learning_rate
    )
    
    # Use the same batch tensor & makss to train over and over
    texts = [
        generate_random_string(40)
        for _ in range(BATCH_SIZE)
    ]
    images = torch.randn(BATCH_SIZE, 3, SIZE, SIZE).to('cuda') # (B, C, W, H)
    # Predict 7 patches per sample, filling in a combination of patches

    for epoch in range(num_epochs):
        # Set models to train mode
        text_encoder.eval()
        context_vision_encoder.train()
        target_vision_encoder.train()
        crosser.train()
        predictor.train()

        # Zero the gradients
        optimizer.zero_grad()
        
        # Generate new random predict_mask at the start of each epoch
        predict_mask = generate_random_predict_mask(BATCH_SIZE, ALL_PATCHES, num_predict_patches=HIDDEN_PATCH, device='cuda')
        
        # Generate context_masks based on predict_mask
        context_masks = generate_context_masks(ALL_PATCHES, predict_mask, device='cuda')
        # print(context_masks)

        # Forward pass
        encoded_text, text_attn_mask = text_encoder(texts)
        # print(f"{encoded_text.shape=}")
        # print(f"{text_attn_mask.shape=}")
        encoded_image_context = context_vision_encoder(images, context_masks)  # Encode the context patches
        # print(f"{encoded_image_context.shape=}")
        T_context, V_context = crosser(encoded_text, encoded_image_context, text_attn_mask)
        # print(f"{T_context.shape=}")
        
        predicted = predictor(V_context, context_masks, predict_mask)  # Generate predictions based on context
        # print(f"{predicted.shape=}")
        
        encoded_image_target = target_vision_encoder(images)  # Encode the full tensor
        _, V_target = crosser(encoded_text, encoded_image_target)
        # print(f"{V_target.shape=}")
        
        T_CLS = T_context[:, 0, :]
        V_CLS = V_target[:, 0, :]
        c_loss = clip_loss(T_CLS, V_CLS, temperature=1)
        
        target = F.layer_norm(V_target, (V_target.size(-1),))  # Normalize the target
        target = apply_masks(target, predict_mask)  # Apply predict mask
        # print(f"{V_target.shape=}")
        
        # Calculate loss (L1 loss here)
        l1_loss = F.smooth_l1_loss(predicted, target)
        
        # Print loss
        print(f"Epoch [{epoch + 1}/{num_epochs}], L1 Loss: {l1_loss.item():.4f}, Cons Loss: {c_loss.item():.4f}")
        
        loss =  c_loss# + l1_loss * 0.01 l1_loss
        
        # Backward pass and optimization
        loss.backward()
        optimizer.step()


def main():
    # test_text_encoder()
    # test_context_vision_encoder()
    # test_target_vision_encoder()
    # test_crosser()
    train(num_epochs=100)
    
    
if __name__ == "__main__":
    main()
