from torch.nn import functional as F
from torchmetrics import F1Score
import torch

f1_score = F1Score(task="multiclass", num_classes=5).to('cuda')

def collate_fn(inputs):
    return inputs

def get_prediction_from_logits(logits):
    probabilities = F.softmax(logits, dim=-1)
    predictedClass = torch.argmax(probabilities, dim=-1)
    
    return predictedClass

def get_accuracy_from_logits(logits, query_labels):
    with torch.no_grad():
        predictions = get_prediction_from_logits(logits)
        accuracy = torch.sum(predictions == query_labels)
        return accuracy / query_labels.numel()

