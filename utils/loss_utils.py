import torch
import torch.nn as nn
import torch.nn.functional as F
import warnings
warnings.filterwarnings("ignore")


class MultiClassFocalLoss(nn.Module):
    def __init__(self, gamma=2, reduction='mean'):
        super(MultiClassFocalLoss, self).__init__()
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        """
        Args:
            inputs (torch.Tensor): Predicted logits (before softmax).
            targets (torch.Tensor): Ground truth labels (class indices).
        """
        # Apply softmax to get probabilities
        probs = F.softmax(inputs, dim=1)
        
        # Gather the probabilities of the true classes
        class_probs = probs.gather(1, targets.view(-1, 1))
        
        # Compute focal loss
        focal_loss = -((1 - class_probs) ** self.gamma) * torch.log(class_probs)
        
        # Apply reduction
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss