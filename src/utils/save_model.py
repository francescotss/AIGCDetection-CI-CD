import os, torch

def save_checkpoint(state, filepath=''):
    os.makedirs(os.path.dirname(filepath),exist_ok=True)
    torch.save(state, filepath)