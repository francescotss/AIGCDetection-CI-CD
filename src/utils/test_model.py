import torch
from utils.train_utils import AverageMeter


def test_model(data_loader, model, criterion, device='cpu', source_name = ''): #Accuracy

    print(f'===> Starting the dataset {source_name}' if source_name else '===> Starting TEST')

    correct, total =0,0
    losses = AverageMeter()

    model.eval()
    model.to(device)
    with torch.no_grad():

        for inputs, targets in data_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            _, predicted = torch.max(outputs, 1)
            correct += (predicted == targets).sum().item()
            total += len(targets)
            losses.update(loss.data.tolist(), inputs.size(0))

        print('\nTest results | Loss:{loss:.4f} | acc:{top:.4f}'.format(loss=losses.avg, top = correct/total*100))
    return (losses.avg, correct/total*100)