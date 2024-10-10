import torch

def similarity_matrix(lst):
    return lst @ lst.T

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

    images_similarity = similarity_matrix(image_embeddings)
    texts_similarity = similarity_matrix(text_embeddings)
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
    # print(f"{text_embeddings=}")
    # print(f"{image_embeddings=}")
    # Normalize embeddings for cosine similarity
    text_embeddings = F.normalize(text_embeddings, p=2, dim=-1)
    image_embeddings = F.normalize(image_embeddings, p=2, dim=-1)

    # Compute logits (cosine similarity scaled by temperature)
    logits = (text_embeddings @ image_embeddings.T) / temperature
    # print(f"{logits=}")

    # Contrastive targets: identity matrix for (text, image) pairs
    targets = torch.eye(text_embeddings.shape[0], device=text_embeddings.device)
    # print(f"{targets=}")

    # Calculate the contrastive loss for both texts and images
    texts_loss = F.cross_entropy(logits, targets, reduction='none')
    # print(f"{texts_loss=}")
    images_loss = F.cross_entropy(logits.T, targets.T, reduction='none')
    # print(f"{images_loss=}")

    # Final contrastive loss
    loss = (texts_loss + images_loss) / 2.0  # shape: (batch_size)
    
    return loss.mean()  # Reduce the mean for the final loss

# def contrastive_loss(anchor, positive, negative, margin=1.0):
#     distance_positive = F.pairwise_distance(anchor, positive)
#     distance_negative = F.pairwise_distance(anchor, negative)
#     losses = torch.relu(distance_positive - distance_negative + margin)
#     return losses.mean()
