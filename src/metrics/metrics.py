

class MeanPixelAccuracy:
    def __init__(self):
        super(MeanPixelAccuracy, self).__init__()

    def __call__(self, pred, target):
        # Calculate pixel-wise accuracy by comparing predicted and ground truth labels
        correct_pixels = (pred.argmax(dim=1) == target.to(pred.device)).float()
        mean_accuracy = correct_pixels.mean()
        return mean_accuracy

class IoUMetric:
    def __init__(self, smooth=1e-6):
        self.smooth = smooth  # Smooth term to prevent division by zero

    def __call__(self, pred, target):
        # pred shape: (batch_size, num_classes, height, width)
        # target shape: (batch_size, height, width)

        # Apply argmax to get predicted class per pixel
        pred = pred.argmax(dim=1)  # shape (batch_size, height, width)

        # Flatten predictions and target for calculation
        pred = pred.view(-1)       # shape (batch_size * height * width,)
        target = target.view(-1)   # shape (batch_size * height * width,)

        # Create a tensor to store IoU for each class
        num_classes = pred.max().item() + 1  # Determine the number of classes from pred
        ious = []

        for cls in range(num_classes):
            pred_cls = (pred == cls).float()  # 1 for pixels in the predicted class, else 0
            target_cls = (target == cls).float()  # 1 for pixels in the target class, else 0

            # Calculate intersection and union
            intersection = (pred_cls * target_cls).sum()
            union = pred_cls.sum() + target_cls.sum() - intersection

            # Compute IoU and add to the list
            iou = (intersection + self.smooth) / (union + self.smooth)
            ious.append(iou)

        # Calculate mean IoU over all classes
        mean_iou = torch.stack(ious).mean()
        
        return mean_iou


class DiceMetric:
    def __init__(self, smooth=1e-6):
        self.smooth = smooth  # Smooth term to prevent division by zero

    def __call__(self, pred, target):
        # pred shape: (batch_size, num_classes, height, width)
        # target shape: (batch_size, height, width)

        # Apply softmax to get probabilities for each class
        pred = pred.softmax(dim=1)  # shape (batch_size, num_classes, height, width)
        
        # Flatten predictions and target across spatial dimensions
        pred = pred.view(pred.size(0), pred.size(1), -1)  # (batch_size, num_classes, height * width)
        target = target.view(target.size(0), 1, -1)       # (batch_size, 1, height * width)

        # Create one-hot encoding for target
        target_one_hot = torch.zeros_like(pred).scatter_(1, target, 1)  # (batch_size, num_classes, height * width)

        # Calculate Dice score for each class
        intersection = (pred * target_one_hot).sum(dim=2)  # sum over pixels
        pred_sum = pred.sum(dim=2)
        target_sum = target_one_hot.sum(dim=2)

        # Compute Dice score for each class
        dice_score = (2 * intersection + self.smooth) / (pred_sum + target_sum + self.smooth)
        
        # Average Dice score over classes
        mean_dice = dice_score.mean()
        
        return mean_dice
