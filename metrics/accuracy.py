import torch
from typing import Union, Tuple


def compute_accuracy(predictions: torch.Tensor, 
                    targets: torch.Tensor,
                    topk: Union[int, Tuple[int]] = 1) -> float:
    """
    Compute classification accuracy
    
    Args:
        predictions: Model predictions [batch_size, num_classes]
        targets: Ground truth labels [batch_size]
        topk: Compute top-k accuracy (default: top-1)
    
    Returns:
        Accuracy as float between 0 and 1
    """
    
    with torch.no_grad():
        # Handle different prediction formats
        if predictions.dim() == 1:
            # Single prediction per sample
            if predictions.size(0) != targets.size(0):
                return 0.0
            pred_classes = predictions.round().long()
        else:
            # Multiple classes - use argmax
            pred_classes = predictions.argmax(dim=-1)
        
        # Ensure targets are long tensor
        targets = targets.long()
        
        # Compute accuracy
        correct = (pred_classes == targets).float()
        accuracy = correct.mean().item()
        
        return accuracy


def accuracy(outputs, labels, topk=(1,)):
    with torch.no_grad():
        maxk = max(topk)
        batch_size = labels.size(0)

        _, pred = outputs.topk(maxk, 1, True, True)  # [batch_size, topk]
        pred = pred.t()  # [topk, batch_size]
        correct = pred.eq(labels.view(1, -1).expand_as(pred))  # [topk, batch_size]

        res = []
        for k in topk:
            correct_k = correct[:k].float().sum()
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


class ClassAccuracy:
    def __init__(self):
        self.sum = {}
        self.count = {}

    def update(self, outputs, labels):
        _, pred = outputs.max(dim=1)
        correct = pred.eq(labels)

        for b, label in enumerate(labels):
            label = label.item()
            if label not in self.sum.keys():
                self.sum[label] = 0
                self.count[label] = 0
            self.sum[label] += correct[b].item()
            self.count[label] += 1

    def __call__(self):
        self.sum = dict(sorted(self.sum.items()))
        self.count = dict(sorted(self.count.items()))
        return [s / c * 100 for s, c in zip(self.sum.values(), self.count.values())]

    def __getitem__(self, item):
        return self.__call__()[item]

    def list(self):
        return self.__call__()

    def __str__(self):
        fmtstr = '{}:{:6.2f}'
        result = '\n'.join([fmtstr.format(l, a) for l, a in enumerate(self.__call__())])
        return result
