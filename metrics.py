import torch

def indices_to_one_hot(class_indices, num_classes=3129):
    """
    Convert a list or tensor of class indices to a tensor of one-hot encoded vectors.

    Args:
        class_indices (list or torch.Tensor): List or 1D tensor of class indices.
        num_classes (int): Total number of classes (size of the one-hot vector).

    Returns:
        torch.Tensor: A tensor of shape (len(class_indices), num_classes) with one-hot encoding.
    """
    # Convert input to a tensor if it's a list
    class_indices = torch.tensor(class_indices, dtype=torch.long)

    # Create a zero tensor with shape (len(class_indices), num_classes)
    one_hot = torch.zeros(len(class_indices), num_classes, dtype=torch.float)

    # Scatter 1s into the appropriate indices
    one_hot.scatter_(1, class_indices.unsqueeze(1), 1.0)

    return one_hot

def calculate_metrics_from_logits(logits, ground_truth):
    """
    Calculate accuracy, precision, recall, and F1-score for multiclass classification.

    Args:
        logits (torch.Tensor): Model output logits of shape (batch_size, num_classes).
        ground_truth (torch.Tensor): Ground truth class labels of shape (batch_size).

    Returns:
        dict: Dictionary containing accuracy, macro precision, macro recall, macro F1,
              weighted precision, weighted recall, weighted F1, and per-class metrics.
    """
    logits = logits.to('cpu')
    ground_truth = ground_truth.to('cpu')
    
    # Convert logits to predicted classes
    pred_classes = torch.argmax(logits, dim=1)

    # Number of classes
    num_classes = logits.size(1)

    # Initialize metrics
    true_positive = torch.zeros(num_classes)
    false_positive = torch.zeros(num_classes)
    false_negative = torch.zeros(num_classes)
    true_negative = torch.zeros(num_classes)

    # Calculate TP, FP, FN, TN for each class
    for k in range(num_classes):
        true_positive[k] = ((pred_classes == k) & (ground_truth == k)).sum()
        false_positive[k] = ((pred_classes == k) & (ground_truth != k)).sum()
        false_negative[k] = ((pred_classes != k) & (ground_truth == k)).sum()
        true_negative[k] = ((pred_classes != k) & (ground_truth != k)).sum()

    # Precision, Recall, and F1-Score for each class
    precision = true_positive / (true_positive + false_positive + 1e-8)
    recall = true_positive / (true_positive + false_negative + 1e-8)
    f1_score = 2 * (precision * recall) / (precision + recall + 1e-8)

    # Replace NaNs with 0 for classes with no samples
    precision[torch.isnan(precision)] = 0
    recall[torch.isnan(recall)] = 0
    f1_score[torch.isnan(f1_score)] = 0

    # Accuracy
    accuracy = (pred_classes == ground_truth).sum().item() / ground_truth.size(0)

    # Macro-Averaged Metrics
    macro_precision = precision.mean().item()
    macro_recall = recall.mean().item()
    macro_f1 = f1_score.mean().item()

    # Weighted-Averaged Metrics
    support = torch.bincount(ground_truth, minlength=num_classes).float()  # Count of true instances per class
    weighted_precision = (precision * support).sum().item() / support.sum().item()
    weighted_recall = (recall * support).sum().item() / support.sum().item()
    weighted_f1 = (f1_score * support).sum().item() / support.sum().item()

    return {
        "accuracy": accuracy,
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "macro_f1": macro_f1,
        "weighted_precision": weighted_precision,
        "weighted_recall": weighted_recall,
        "weighted_f1": weighted_f1,
        "per_class_precision": precision.tolist(),
        "per_class_recall": recall.tolist(),
        "per_class_f1": f1_score.tolist(),
    }
