import torch
import random

from src.models.vision_transformer import vit_test, vit_predictor_test
from src.models.modules import text_encoder_model, vision_encoder, crosser_module
from src.utils.tensors import apply_masks, repeat_interleave_batch
from create_dataset import ImageTextDataset
import torch.nn.functional as F
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import torch.nn as nn

class SiameseTextVisionModel(nn.Module):
    def __init__(self, text_encoder, vision_encoder):
        super(SiameseTextVisionModel, self).__init__()
        self.text_encoder = text_encoder
        self.vision_encoder = vision_encoder

    def forward(self, text, images, masks=None):
        text_embeddings, _ = self.text_encoder(text)
        image_embeddings = self.vision_encoder(images, masks)
        return text_embeddings, image_embeddings
    
def mean_pooling(embeddings):
    # Average across the patch dimension (second dimension)
    return embeddings.mean(dim=1)

def siamese_contrastive_loss(text_embeddings, image_embeddings, margin=1.0):
    # Normalize embeddings for cosine similarity
    text_norm = F.normalize(text_embeddings, p=2, dim=-1)
    image_norm = F.normalize(image_embeddings, p=2, dim=-1)
    
    # Pool embeddings to reduce dimensions
    text_pooled = mean_pooling(text_norm)
    image_pooled = mean_pooling(image_norm)
    
    # Compute cosine similarity
    similarity_matrix = torch.mm(text_pooled, image_pooled.mT)
    
    # Identity matrix as targets for positive pairs
    targets = torch.eye(text_pooled.size(0), device=text_pooled.device)

    # Contrastive loss calculation
    loss = F.binary_cross_entropy_with_logits(similarity_matrix, targets)
    return loss



# def margin_function(text_features, contrastive_features):
#     """
#     Compute the margin based on text and contrastive features.
    
#     Args:
#         text_features: Tensor of text features (positive samples) of shape (batch_size * seq_length, feature_dim).
#         contrastive_features: Tensor of contrastive points (negative samples) of shape (batch_size * seq_length, feature_dim).
    
#     Returns:
#         Computed margin.
#     """
#     return torch.norm(text_features - contrastive_features, dim=1) * 0.1

# def generate_contrastive_features(features, temperature=1):
#     """
#     Generate contrastive features using other samples in the batch.
    
#     Args:
#         features: Tensor of shape (batch_size, sequence_length, feature_dim)
#         temperature: Temperature parameter for softmax
    
#     Returns:
#         Tensor of contrastive features
#     """
#     batch_size, seq_length, feature_dim = features.size()
    
#     # Reshape features to (batch_size * sequence_length, feature_dim)
#     features_flat = features.view(-1, feature_dim)
    
#     # Compute similarity matrix
#     sim_matrix = torch.matmul(features_flat, features_flat.t()) / temperature
    
#     # Remove self-similarities from the matrix
#     mask = torch.eye(batch_size * seq_length, device=features.device)
#     sim_matrix = sim_matrix - mask * 1e9
    
#     # Sample negative features
#     neg_indices = torch.multinomial(F.softmax(sim_matrix, dim=1), num_samples=1).squeeze()
#     contrastive_features_flat = features_flat[neg_indices]
    
#     # Reshape back to original dimensions
#     contrastive_features = contrastive_features_flat.view(batch_size, seq_length, feature_dim)
    
#     return contrastive_features

# def pairwise_hinge_loss(image_features, text_features, contrastive_features, margin=0.1):
#     """
#     Pairwise hinge loss function for 3D tensors.

#     Args:
#         image_features: Tensor of image features (positive samples) of shape (batch_size, seq_length, feature_dim).
#         text_features: Tensor of text features (positive samples) of shape (batch_size, seq_length, feature_dim).
#         contrastive_features: Tensor of contrastive points (negative samples) of shape (batch_size, seq_length, feature_dim).
#         margin: Margin for the hinge loss.

#     Returns:
#         Computed hinge loss.
#     """
#     batch_size, seq_length, feature_dim = image_features.size()

#     # Compute cosine similarity
#     def cosine_similarity(x, y):
#         return F.cosine_similarity(x.unsqueeze(1), y.unsqueeze(0), dim=2)

#     # Compute similarities
#     pos_sim = cosine_similarity(image_features.view(-1, feature_dim), text_features.view(-1, feature_dim))
#     neg_sim = cosine_similarity(image_features.view(-1, feature_dim), contrastive_features.view(-1, feature_dim))

#     # Compute loss
#     loss = F.relu(neg_sim - pos_sim.diag().unsqueeze(1) + margin)

#     return loss.mean()


def cross_entropy(preds, targets, reduction='none'):
    log_softmax = torch.nn.LogSoftmax(dim=-1)
    loss = (-targets * log_softmax(preds)).sum(1)
    if reduction == "none":
        return loss
    elif reduction == "mean":
        return loss.mean()
    
# Calculating the Loss
def contrastive_loss(text_embeddings, image_embeddings, temperature=0.07):
    logits = (text_embeddings @ image_embeddings.T) / temperature
    print(f"{logits=}")
    images_similarity = image_embeddings @ image_embeddings.T
    texts_similarity = text_embeddings @ text_embeddings.T
    targets = F.softmax(
        (images_similarity + texts_similarity) / 2 * temperature, dim=-1
    )
    # print(f"{targets=}")
    texts_loss = cross_entropy(logits, targets, reduction='none')
    # print(f"{texts_loss=}")
    images_loss = cross_entropy(logits.T, targets.T, reduction='none')
    # print(f"{images_loss=}")
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

def contrastive_loss(anchor, positive, negative, margin=1.0):
    distance_positive = F.pairwise_distance(anchor, positive)
    distance_negative = F.pairwise_distance(anchor, negative)
    losses = torch.relu(distance_positive - distance_negative + margin)
    return losses.mean()

##################
SIZE = 224
PATCH_SIZE = 16

EMBED_DIM = 768
PREDICTOR_EMBED_DIM = 384
DEPTH = 6
NUM_HEADS = 4

def train(num_epochs=1, max_images_per_epoch=10, batch_size=10, learning_rate=0.01):
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
    context_crosser = crosser_module(
        text_embed_dim=768,
        vision_embed_dim=EMBED_DIM,
        hidden_dim=EMBED_DIM,
        depth=DEPTH,
        num_heads=NUM_HEADS,
    ).to('cuda')
    target_crosser = crosser_module(
        text_embed_dim=768,
        vision_embed_dim=EMBED_DIM,
        hidden_dim=EMBED_DIM,
        depth=DEPTH,
        num_heads=NUM_HEADS,
    ).to('cuda')
    
    NUM_PATCHES = context_vision_encoder.patch_embed.num_patches

    predictor = vit_predictor_test(
        embed_dim=EMBED_DIM,
        depth=DEPTH,
        num_heads=NUM_HEADS,
        predictor_embed_dim=PREDICTOR_EMBED_DIM,
        num_patches=NUM_PATCHES,
    ).to('cuda')
    
    model = SiameseTextVisionModel(text_encoder, context_vision_encoder).to('cuda')
    # Optimizer
    optimizer = optim.Adam(
        list(text_encoder.parameters()) +
        list(context_vision_encoder.parameters()) +
        list(target_vision_encoder.parameters()) +
        list(context_crosser.parameters()) +
        list(target_crosser.parameters()) +
        list(predictor.parameters()), lr=learning_rate
    )
    
    HIDDEN_RATIO = (0.4, 0.5)
        
    dataset = ImageTextDataset(
        image_path='src/datasets/train', 
        caption_path='src/datasets/annotations/filename_caption_dict.json', 
        batch_size=batch_size,
        img_size=SIZE,
        patch_size=PATCH_SIZE,
        _hidden_ratio=HIDDEN_RATIO,
        max=max_images_per_epoch,
        transform=transforms.Compose(
            [
                transforms.Resize((SIZE, SIZE)), 
                transforms.ToTensor()
            ]
        )
    )
    
    for epoch in range(num_epochs):
        # Set models to train mode
        text_encoder.eval()
        context_vision_encoder.train()
        target_vision_encoder.train()
        context_crosser.train()
        target_crosser.train()
        predictor.train()

        # Initialize tqdm for the dataset
        with tqdm(dataset, desc=f"Epoch {epoch+1}/{num_epochs}") as pbar:
            for images, captions, context_masks, predict_masks in pbar:
                # print(images)
                # print(captions)
                # Zero the gradients
                optimizer.zero_grad()
                
                # Forward pass
                encoded_text, text_attn_mask = text_encoder(captions)
                # print(f"{encoded_text.shape=}")
                # print(f"{text_attn_mask.shape=}")
                encoded_image_context = context_vision_encoder(images, context_masks)  # Encode the context patches
                # print(f"{encoded_image_context.shape=}")
                T_context, V_context = context_crosser(encoded_text, encoded_image_context, text_attn_mask)
                # print(f"{T_context.shape=}")
                
                predicted = predictor(V_context, context_masks, predict_masks)  # Generate predictions based on context
                # print(f"{predicted.shape=}")
                
                encoded_image_target = target_vision_encoder(images)  # Encode the full tensor
                _, V_target = target_crosser(encoded_text, encoded_image_target)
                # print(f"{V_target.shape=}")
                
                T_CLS = T_context[:, 0, :]
                V_CLS = V_target[:, 0, :]
                # c_loss = clip_loss(T_CLS, V_CLS, temperature=1)

                # # generate contrastive features
                # contrastive_features = generate_contrastive_features(T_context)

                # # Calculate the contrastive loss
                # hinge_loss = pairwise_hinge_loss(T_context, V_context, contrastive_features)
                
                target = F.layer_norm(V_target, (V_target.size(-1),))  # Normalize the target
                target = apply_masks(target, predict_masks)  # Apply predict mask
                
                # Calculate loss (L1 loss here)
                l1_loss = F.smooth_l1_loss(predicted, target)
                
                # text_embeddings, image_embeddings = model(captions, images, context_masks)

                # Calculate the contrastive loss
                simamese_loss = siamese_contrastive_loss(encoded_text, encoded_image_context)
                
                # print(predicted[0][0])
                # print(target[0][0])
                
                loss = 0.5 * l1_loss +  0.5 * simamese_loss
                
                
                # Backward pass and optimization
                loss.backward()
                optimizer.step()

                
                # Update tqdm description with current loss values
                pbar.set_postfix({
                    'L1 Loss': l1_loss.item(),
                    # 'Clip Loss': c_loss.item(),
                    'Siamese Loss': simamese_loss.item()
                    # 'Hinge Loss': hinge_loss.item()
                })


def main():
    train(
        num_epochs=10, 
        max_images_per_epoch=10000, 
        batch_size=5
    )
    
    
if __name__ == "__main__":
    main()
