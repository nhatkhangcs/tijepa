import torch

def similarity_matrix(lst):
    return lst @ lst.T

def cosine_similarity_matrix(embeddings):
    """Compute the cosine similarity matrix for a set of embeddings."""
    # Normalize each vector to have unit length (L2 normalization)
    normalized_embeddings = F.normalize(embeddings, p=2, dim=-1)
    
    # Compute the cosine similarity matrix using matrix multiplication of normalized vectors
    cosine_sim = normalized_embeddings @ normalized_embeddings.T  # Cosine similarity matrix
    
    return cosine_sim

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

def clip_loss(text_embeddings, image_embeddings, temperature=1.0, off_diagonal_penalty_weight=None):
    # print(f"{text_embeddings=}")
    # print(f"{image_embeddings=}")
    # Normalize embeddings for cosine similarity
    text_embeddings = F.normalize(text_embeddings, p=2, dim=-1)
    image_embeddings = F.normalize(image_embeddings, p=2, dim=-1)

    # Compute logits (cosine similarity scaled by temperature)
    logits = (text_embeddings @ image_embeddings.T) / temperature
    print(f"{logits=}")
    # average of diagonal
    print(f"Diag: {torch.diagonal(logits).mean()}")
    # average of off-diagonal
    print(f"OffD: {(logits.sum() - torch.diagonal(logits).sum()) / (logits.numel() - logits.size(0))}")

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
    loss = loss.mean()

    # Add off-diagonal penalty to the loss
    if off_diagonal_penalty_weight is not None:
        # Compute the sum of all logits
        total_logits_sum = logits.sum()

        # Compute the sum of diagonal logits (i.e., correct pairs)
        diagonal_sum = torch.diagonal(logits).sum()

        # Calculate the number of off-diagonal elements
        num_off_diagonal_elements = logits.numel() - logits.size(0)

        # Compute the average off-diagonal penalty
        off_diagonal_penalty = (total_logits_sum - diagonal_sum) / num_off_diagonal_elements
        print(f"{off_diagonal_penalty=}")

        # Add the off-diagonal penalty to the loss with the specified weight
        loss += off_diagonal_penalty_weight * off_diagonal_penalty
    
    return loss

def contrastive_l1_loss(text_embeddings, image_embeddings, temperature=1.0):
    # Normalize embeddings for cosine similarity
    text_embeddings = F.normalize(text_embeddings, p=2, dim=-1)
    image_embeddings = F.normalize(image_embeddings, p=2, dim=-1)

    # Compute logits (cosine similarity scaled by temperature)
    logits = (text_embeddings @ image_embeddings.T) / temperature

    # Contrastive targets: identity matrix for (text, image) pairs
    targets = torch.eye(text_embeddings.shape[0], device=text_embeddings.device)

    loss = F.l1_loss(logits, targets, reduction='mean')

    return loss


def max_margin_loss(text_embeddings, image_embeddings, margin=1.0):
    """
    Compute the max-margin loss (contrastive loss) between text and image embeddings.
    Args:
        text_embeddings: Embeddings for the text inputs (batch_size, embedding_dim).
        image_embeddings: Embeddings for the image inputs (batch_size, embedding_dim).
        margin: The margin for dissimilar pairs. Defaults to 1.0.
    Returns:
        loss: The contrastive loss.
    """
    # Normalize embeddings to compute cosine similarity
    text_embeddings = F.normalize(text_embeddings, p=2, dim=-1)
    image_embeddings = F.normalize(image_embeddings, p=2, dim=-1)

    # Compute pairwise similarity (cosine similarity)
    similarity = text_embeddings @ image_embeddings.T
    print(f"{similarity=}")

    # Get positive and negative pairs
    positive_pairs = torch.diagonal(similarity)  # Similar pairs (diagonal elements)
    negative_pairs = similarity - torch.diag(positive_pairs)  # Dissimilar pairs (off-diagonal elements)
    

    # Max-margin loss: hinge loss formula
    positive_loss = F.relu(margin - positive_pairs).mean()  # Penalize similar pairs being too far apart
    negative_loss = F.relu(negative_pairs + margin).mean()  # Penalize dissimilar pairs being too close
    print(f"Avg +: {positive_pairs.mean()} | {positive_loss=}")
    print(f"Avg -: {negative_pairs.mean()} | {negative_loss=}")

    # Total loss: mean of positive and negative losses
    loss = positive_loss + negative_loss

    return loss

def weighted_max_margin_loss(text_embeddings, image_embeddings, margin=1.0):
    """
    Compute the max-margin loss (contrastive loss) between text and image embeddings.
    Args:
        text_embeddings: Embeddings for the text inputs (batch_size, embedding_dim).
        image_embeddings: Embeddings for the image inputs (batch_size, embedding_dim).
        margin: The margin for dissimilar pairs. Defaults to 1.0.
    Returns:
        loss: The contrastive loss.
    """
    # Normalize embeddings to compute cosine similarity
    text_embeddings = F.normalize(text_embeddings, p=2, dim=-1)
    image_embeddings = F.normalize(image_embeddings, p=2, dim=-1)
    batch_size = text_embeddings.size(0)
    wp = batch_size / batch_size**2
    wn = (batch_size**2 - batch_size) / batch_size**2

    # Compute pairwise similarity (cosine similarity)
    similarity = text_embeddings @ image_embeddings.T
    print(f"{similarity=}")

    # Get positive and negative pairs
    positive_pairs = torch.diagonal(similarity)  # Similar pairs (diagonal elements)
    negative_pairs = similarity - torch.diag(positive_pairs)  # Dissimilar pairs (off-diagonal elements)
    

    # Max-margin loss: hinge loss formula
    positive_loss = F.relu(margin - positive_pairs).mean()  # Penalize similar pairs being too far apart
    negative_loss = F.relu(negative_pairs + margin).mean()  # Penalize dissimilar pairs being too close
    print(f"Avg +: {positive_pairs.mean()} | {positive_loss=}")
    print(f"Avg -: {negative_pairs.mean()} | {negative_loss=}")

    # Total loss: mean of positive and negative losses
    loss = positive_loss * wp + negative_loss * wn

    return loss

def max_margin_loss_negative_only(text_embeddings, image_embeddings, margin=1.0):
    """
    Compute the max-margin loss (contrastive loss) between text and image embeddings.
    Args:
        text_embeddings: Embeddings for the text inputs (batch_size, embedding_dim).
        image_embeddings: Embeddings for the image inputs (batch_size, embedding_dim).
        margin: The margin for dissimilar pairs. Defaults to 1.0.
    Returns:
        loss: The contrastive loss.
    """
    # Normalize embeddings to compute cosine similarity
    text_embeddings = F.normalize(text_embeddings, p=2, dim=-1)
    image_embeddings = F.normalize(image_embeddings, p=2, dim=-1)

    # Compute pairwise similarity (cosine similarity)
    similarity = text_embeddings @ image_embeddings.T
    print(f"{similarity=}")

    # Get positive and negative pairs
    positive_pairs = torch.diagonal(similarity)  # Similar pairs (diagonal elements)
    negative_pairs = similarity - torch.diag(positive_pairs)  # Dissimilar pairs (off-diagonal elements)
    
    # Max-margin loss: hinge loss formula
    print(f"Avg -: {negative_pairs.mean()}")
    negative_loss = F.relu(negative_pairs + margin).mean()  # Penalize dissimilar pairs being too close
    print(f"Loss: {negative_loss}")

    # Total loss: mean of positive and negative losses
    loss = negative_loss

    return loss


def semantic_soft_clip_loss(
        original_text_embeddings,
        original_image_embeddings,
        predicted_embeddings,
        target_embeddings,
        temperature=1.0,
    ):
    """
    Contrastive loss between predicted and target embeddings where the target 
    is based on the similarity of the original embeddings.
    """
    # print(f"{original_text_embeddings.device=}")
    # print(f"{original_image_embeddings.device=}")
    # print(f"{predicted_embeddings.device=}")
    # print(f"{target_embeddings.device=}")

    # Normalize predicted and target embeddings
    predicted_embeddings = F.normalize(predicted_embeddings, p=2, dim=-1)
    target_embeddings = F.normalize(target_embeddings, p=2, dim=-1)

    # Compute similarity matrices for original embeddings
    text_similarity = cosine_similarity_matrix(original_text_embeddings)
    print(f"{text_similarity=}")
    image_similarity = cosine_similarity_matrix(original_image_embeddings)
    print(f"{image_similarity=}")

    # Compute logits for predicted-target pairs
    logits = predicted_embeddings @ target_embeddings.T
    print(f"{logits=}")

    # Average similarity matrix to form soft targets
    soft_targets = F.softmax((text_similarity + image_similarity) / 2 / temperature, dim=-1)
    print(f"{soft_targets=}")

    # Calculate cross-entropy loss using the soft target similarity matrix
    pred_to_target_loss = F.cross_entropy(logits, soft_targets, reduction='none')
    print(f"{pred_to_target_loss=}")

    target_to_pred_loss = F.cross_entropy(logits.T, soft_targets.T, reduction='none')
    print(f"{target_to_pred_loss=}")

    # Final loss is the average of the forward and reverse losses
    loss = (pred_to_target_loss + target_to_pred_loss) / 2.0

    return loss.mean()

# def contrastive_loss(anchor, positive, negative, margin=1.0):
#     distance_positive = F.pairwise_distance(anchor, positive)
#     distance_negative = F.pairwise_distance(anchor, negative)
#     losses = torch.relu(distance_positive - distance_negative + margin)
#     return losses.mean()
