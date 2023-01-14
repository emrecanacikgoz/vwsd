import torch
import torch.nn as nn
from transformers import CLIPModel


class CLIPZeroShotBaseline(nn.Module):
    def __init__(self, architecture='openai/clip-vit-large-patch14'):
        super().__init__()
        self.model = CLIPModel.from_pretrained(architecture)

    def _expand(self, input, N):
        B, T = input.shape
        input = input.unsqueeze(1)
        input = input.expand(B, N, T)
        input = input.reshape(B*N, T)
        return input

    def forward(self, pixel_values, input_ids, attention_mask, **kwargs):
        B, N, C, H, W = pixel_values.shape
        pixel_values = pixel_values.reshape(B*N, C, H, W)
        outputs = self.model(
            pixel_values=pixel_values,
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        logits = outputs.logits_per_text  # logits_per_text: (B, B*N)
        logits = logits.reshape(B, B, N)
        I = torch.arange(B)
        logits = logits[I, I, :]
        return logits