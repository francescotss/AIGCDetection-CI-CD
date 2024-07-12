import os, torch

def save_checkpoint(state=None, model=None, path=''):
    assert state or model, "Save error: state or model not provided"

    if state:
        filepath = os.path.join(path, 'model_best_accuracy.pth')
        os.makedirs(os.path.dirname(filepath),exist_ok=True)
        torch.save(state, filepath)
    
    if model:
        model.save(path)