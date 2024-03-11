import os, torch

def save_checkpoint(state, path=''):
    filepath = os.path.join(path, 'model.pth')
    os.makedirs(os.path.dirname(filepath),exist_ok=True)
    torch.save(state, filepath)