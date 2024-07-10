import torch.nn as nn
from transformers import ViTForImageClassification


model_urls = {
    'base16-224': 'google/vit-base-patch16-224-in21k',
    'small16-224': 'WinKawaks/vit-small-patch16-224',
    'tiny16-224': 'WinKawaks/vit-tiny-patch16-224'
}

class ViT(nn.Module):

    def __init__(self, weights, num_classes=2, ignore_mismatched_sizes=False):
        super(ViT, self).__init__()
        self.vit = ViTForImageClassification.from_pretrained(weights,
                                                             num_labels=num_classes, 
                                                             ignore_mismatched_sizes=ignore_mismatched_sizes)
                
    def forward(self, x):
        out = self.vit(pixel_values=x) # type: ignore
        return out.logits
    
    def save(self, dir):
        self.vit.save_pretrained(dir,safe_serialization=False) # type: ignore


def vit(weights='', num_classes=2):
    ignore_mismatched_sizes=False
    if not weights:
        weights = model_urls['small16-224']
        ignore_mismatched_sizes=True

    model = ViT(weights,num_classes=num_classes, ignore_mismatched_sizes=ignore_mismatched_sizes)
    return model
