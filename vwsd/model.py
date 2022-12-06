import torch
import torch.nn as nn
from transformers import CLIPModel


class CLIPZeroShotBaseline(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.model = CLIPModel.from_pretrained(config.get('architecture'))

    def forward(self, images, text, *args, **kwargs):
        B, N, C, H, W = images.shape[:2]
        images = images.view(B*N, C, H, W)
        image_features = self.model.get_image_features(images)
        text_features = self.model.get_text_features(text)
        raise NotImplementedError('WIP')
        