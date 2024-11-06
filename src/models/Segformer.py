from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor
import torch.nn as nn
import torch.nn.functional as F


class Segformer(nn.Module):
    def __init__(self):
        super(Segformer, self).__init__()
        self.segformer = SegformerForSemanticSegmentation.from_pretrained('nvidia/segformer-b1-finetuned-ade-512-512')
        self.segformer.decode_head.classifier = nn.Conv2d(256,1,kernel_size=1)

    def forward(self, image):
        batch_size = len(image)
        mask = self.segformer(image).logits
        mask = F.interpolate(mask, image.shape[-2:], mode="bilinear", align_corners=True)
        mask = mask.squeeze(1)
        
        return mask