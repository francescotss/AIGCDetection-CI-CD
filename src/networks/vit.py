import torch.nn as nn
from transformers import ViTForImageClassification


model_urls = {
    'base16-224': 'google/vit-base-patch16-224-in21k',
}

class ViT(nn.Module):

    def __init__(self, weights, num_classes=2):
        super(ViT, self).__init__()
        self.vit = ViTForImageClassification.from_pretrained(weights,
                                                             num_labels=num_classes)
                
    def forward(self, x):
        out = self.vit(pixel_values=x)
        return out.logits
    
    def save(self, dir):
        self.vit.save_pretrained(dir,safe_serialization=False)


def vit(weights='', num_classes=2):
    if weights == '':
        weights = model_urls['base16-224']

    model = ViT(weights,num_classes=num_classes)
    return model
