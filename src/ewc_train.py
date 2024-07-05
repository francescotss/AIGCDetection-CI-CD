from copy import deepcopy
import argparse, configparser

import torch
from torch import nn
from torch.nn import functional as F
import torch.utils.data
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR

from utils.save_model import save_checkpoint
from utils.test_model import test_model
from utils.data_loader import create_dataloader
from utils.model_loader import load_models

def parse_args():
    parser = argparse.ArgumentParser("ewc_train")
    parser.add_argument("-c", "--config_file", default="ewc_model_config.conf", type=str, help='Config file')
    parser.add_argument("--network", help="Supported Networks: ResNet, ResNet18, Xception, MobileNet2")
    parser.add_argument("--input_model")
    parser.add_argument("--output_dir")
    parser.add_argument("--source_datasets", help="comma-separated list of directories. Example: /datasets/ds1,/datasets/ds2")
    parser.add_argument("--target_dataset", help="Target dataset directory")
    parser.add_argument("--use_comet", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--comet_name", help="Experiment name for comet logging")

    args = parser.parse_args()
    assert args.config_file

    config = configparser.ConfigParser()
    config.read(args.config_file)
    defaults = {}
    defaults.update(dict(config.items("Defaults")))
    
    parser.set_defaults(**defaults)
    args = parser.parse_args()

    print(args)
    return args



class EWC(object):
    def __init__(self, model: nn.Module, datasets, optimizer , criterion, device, n_sample_batches):

        self.model = model
        self.datasets = datasets
        self.optimizer = optimizer
        self.device = device
        self.criterion = criterion
        self.n_sample_batches = n_sample_batches

        self.params = {n: p for n, p in self.model.named_parameters() if p.requires_grad}
        self._means = {}
        self._precision_matrices = self._diag_fisher()

        for n, p in deepcopy(self.params).items():
            self._means[n] = p.data.to(self.device)

    # Compute the diagonal Fisher information matrix
    def _diag_fisher(self):
        self.model.eval()
        precision_matrices = {}
        for n, p in deepcopy(self.params).items():
            p.data.zero_()
            precision_matrices[n] = p.data.to(self.device)

        for dataset in self.datasets:
          for batch_idx, (input, target) in enumerate(dataset):
            if batch_idx < self.n_sample_batches:
              input = input.to(self.device)
              target = target.to(self.device)
              self.optimizer.zero_grad()
              output = self.model(input)
              loss = self.criterion(output, target)
              loss.backward()

              for n, p in self.model.named_parameters():
                if p.grad is not None:
                  precision_matrices[n].data += p.grad.data.clone().pow(2) / len(dataset)

        precision_matrices = {n: p for n, p in precision_matrices.items()}
        self.model.train()
        return precision_matrices

    # Compute the EWC penalty
    def penalty(self, model: nn.Module):
        loss = 0.0
        for n, p in model.named_parameters():
          if n not in self._precision_matrices:
            continue
          _loss = self._precision_matrices[n] * ((p - self._means[n]) ** 2)
          loss += _loss.sum()
        return loss





def ewc_train(args):

    # Init
    device = 'cuda' if args.num_gpu else 'cpu'
    epochs = int(args.epochs)
    lr = float(args.lr)
    early_stop = int(args.early_stop)
    sample_batch = int(args.sample_batch)
    importance = float(args.importance)


    # Logger
    logger = None
    if args.use_comet:
        comet_ml.init()
        logger = comet_ml.Experiment()
        logger.set_name(args.comet_name)
        logger.log_parameters(parameters=vars(args))

    # Load datasets and models
    target_loaders, sources_loaders = create_dataloader(args, train=True)
    print("Dataset available in target_loaders: ", " / ".join([n for n in target_loaders]))
    print("Dataset available in val_loaders: ", " / ".join([n for n in sources_loaders]))
    _, model = load_models(args.input_model, args.network, num_gpu = args.num_gpu)
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.1)

    # Learning rate scheduler
    if args.lr_schedule == "cosine":
        print("Apply Cosine learning rate schedule")
        lr_scheduler = CosineAnnealingLR(optimizer=optimizer,
                                        T_max=10,
                                        eta_min=1e-5,
                                        verbose=True)
    else:
        print(f"Input: {args.lr_schedule}, No learning rate schedule applied ... ")
        

    # Pre-evaluation
    _, test_acc = test_model(target_loaders['val'], model, criterion, device=device) 
    if logger: logger.log_metric('start_acc',test_acc,0)
    print("Start Target Validation ACC: {:.2f}%".format(test_acc))


    # Get data loaders
    train_loader = target_loaders['train']
    old_tasks_loaders = list(sources_loaders.values())

    # Compute weight importance
    ewc = EWC(model, old_tasks_loaders, optimizer, criterion, device, sample_batch)


    # ------- START TRAINING ------- #
    best_acc = 0
    cur_patience = 0
    print(f"Start training in {epochs} epochs")

    for epoch in range(epochs):
        print(f"\n\n---------- Starting epoch {epoch} ----------")
        correct,total = 0,0
        tot_ewc_loss, tot_task_loss = 0.0, 0.0
        model.train()

        for batch_idx, (inputs, targets) in enumerate(train_loader):
            # Load data
            step = (batch_idx+1) * (epoch+1)
            inputs = inputs.to(device).to(torch.float32)
            targets = targets.to(device).to(torch.long)

            # Forward
            outputs = model(inputs)

            # Losses
            task_loss = criterion(outputs, targets)
            ewc_loss = importance * ewc.penalty(model)
            loss = task_loss + ewc_loss

            # Display
            print("Train Epoch: {e:03d} Batch: {batch:05d}/{size:05d} | Loss: {loss:.4f} | EWC Loss: {ewc:.4f} | CE Loss: {ce:.4f}"
                            .format(e=epoch, batch=batch_idx+1, size=len(train_loader), loss=loss.item(), ewc=ewc_loss, ce=task_loss))

            # Learn
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Predict
            tot_task_loss += task_loss.item()
            tot_ewc_loss += ewc_loss.item() # type: ignore
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == targets).sum().item()
            total += len(targets)


        # Metrics
        epoch_task_loss = tot_task_loss / len(train_loader)
        epoch_ewc_loss = tot_ewc_loss / len(train_loader)
        tot_loss = epoch_task_loss + epoch_ewc_loss
        acc = (correct / total) * 100

        # Logging
        if logger:
            logger.log_metric('losses/loss', tot_loss, epoch)
            logger.log_metric('losses/loss_main', epoch_task_loss, epoch)
            logger.log_metric('losses/loss_ewc', epoch_ewc_loss, epoch)
            logger.log_metric('acc/train_acc', acc, epoch)
        print(f"Train Epoch: {epoch:03d} |Acc {acc:0.5f} | Loss: {tot_loss:.5f} | Task Loss {epoch_task_loss:.5f} | EWC Loss: {epoch_ewc_loss:.5f}")

        # Learning rate scheduler step
        if lr_scheduler: lr_scheduler.step()


        # ---- START VALIDATION ---- #

        # Target (Current Task) 
        _, test_acc = test_model(target_loaders['val'], model, criterion, device=device)
        total_acc = test_acc
        print("[VAL Acc] Target: {:.2f}%".format(test_acc))
        if logger: logger.log_metric('acc/target_val_acc', test_acc, epoch=epoch)

        # Sources (Past tasks)
        cnt = 1
        for source_name in sources_loaders:
                _, source_acc = test_model(sources_loaders[source_name], model, criterion, device=device, source_name=source_name)
                total_acc += source_acc
                print("[VAL Acc] Source {}: {:.2f}%".format(source_name, source_acc))
                if logger: logger.log_metric(f'acc/{source_name}_val_acc', source_acc, epoch=epoch)
                cnt += 1
        print("[VAL Acc] Avg {:.2f}%".format(total_acc / cnt))
        if logger: logger.log_metric('acc/val_acc', total_acc/cnt, epoch=epoch)

        # Evaluate performances
        is_best_acc = total_acc > best_acc
        if is_best_acc:
          print("VAL Acc improve from {:.2f}% to {:.2f}%".format(best_acc/cnt, total_acc/cnt))
          cur_patience = 0
        else:
          cur_patience += 1

        # Save
        best_acc = max(total_acc,best_acc)
        if  is_best_acc or epoch==0:
          save_checkpoint({
              'epoch': epoch + 1,
              'state_dict': model.state_dict(),
              'best_acc': best_acc,
              'optimizer': optimizer.state_dict()},
            path=args.output_dir
          )
          
          print('Save best model')

        # Early stop
        if cur_patience == early_stop:
            print("Early stopping ...")
            if logger: logger.end()
            return 

    if logger: logger.end()     


if __name__ == "__main__":

    args = parse_args()

    if args.use_comet:
         import dotenv
         import comet_ml
         dotenv.load_dotenv()
    
    ewc_train(args)

