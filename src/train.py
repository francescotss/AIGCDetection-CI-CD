import argparse, configparser
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR

from utils.data_loader import create_dataloader
from utils.model_loader import load_models
from utils.loss import loss_fn_kd, loss_clampping
from utils.test_model import test_model
from utils.train_utils import ReduceWeightOnPlateau
from utils.save_model import save_checkpoint

def parse_args():
    parser = argparse.ArgumentParser("train")
    parser.add_argument("-c", "--config_file", type=str, help='Config file')
    parser.add_argument("--train_data", type=str, help="Path to train dataset")
    parser.add_argument("--savepath", type=str, help="Output model save path")
    
    
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





def train(args):

    # Init
    device = 'cuda' if args.num_gpu else 'cpu'
    lr = args.lr
    alpha_kd = args.KD_alpha
    num_class = args.num_class


    # Load datasets and models
    train_loaders, val_loaders = create_dataloader(args)
    print("Dataset available in train_loaders: ", " / ".join([n for n in train_loaders]))
    print("Dataset available in val_loaders: ", " / ".join([n for n in val_loaders]))
    teacher_model, student_model = load_models(args.weight, args.network, num_gpu = args.num_gpu)
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = optim.SGD(student_model.parameters(), lr=lr, momentum=0.1)

    # Learning rate scheduler
    if args.lr_schedule == "cosine":
        print("Apply Cosine learning rate schedule")
        lr_scheduler = CosineAnnealingLR(optimizer=optimizer,
                                        T_max=10,
                                        eta_min=1e-5,
                                        verbose=True)
    else:
        print(f"Input: {args.lr_schedule}, No learning rate schedule applied ... ")
        


    best_acc = 0
    cur_patience = 0 # Early stop and saving
    print(f"Start training in {args.epochs} epochs")


    # ------- START TRAINING ------- #
    for epoch in range(args.epochs):
        correct,total = 0,0
        teacher_model.eval()
        student_model.train()
        disp = {}

        for batch_idx, (inputs, targets) in enumerate(train_loaders['train']):
            # Load data
            step = (batch_idx+1) * (epoch+1)
            inputs = inputs.to(device).to(torch.float32)
            targets = targets.to(device).to(torch.long)
            if torch.isnan(inputs).any() or torch.isnan(targets).any():
                raise ValueError("There is Nan values in input or target")

            # Forward
            teacher_outputs = teacher_model(inputs)
            penul_ft, outputs = student_model(inputs, True)

            # KD loss
            loss_main = criterion(outputs, targets)
            loss_kd = loss_fn_kd(outputs, targets, teacher_outputs)
            loss_kd = loss_clampping(loss_kd, 0, 1800)

            # Total loss
            loss = loss_main  + alpha_kd*loss_kd
            
            # Log and display
            disp["CE"] = loss_main.item()
            disp["KD"] = loss_kd.item() if loss_kd > 0 else 0.0
            call = ' | '.join(["{}: {:.4f}".format(k, v) for k, v in disp.items()])
            print("Train Epoch: {e:03d} Batch: {batch:05d}/{size:05d} | Loss: {loss:.4f} | {call}"
                            .format(e=epoch+1, batch=batch_idx+1, size=len(train_loaders['train']), loss=loss.item(), call=call))

            # Learn!
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Predictions
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == targets).sum().item()
            total += len(targets)

        # Learning rate scheduler step
        lr_scheduler.step()


        # ----- Epoch Validation ------ #

        # Current task
        _, test_acc = test_model(train_loaders['val'], student_model, criterion, device=device, source_name=args.name_target)
        total_acc = test_acc
        print("[VAL Acc] Target: {:.2f}%".format(test_acc))

        # Past tasks
        cnt = 1
        for source_name in val_loaders:
                _, source_acc = test_model(val_loaders[source_name], student_model, criterion, device=device, source_name=source_name)
                total_acc += source_acc
                print("[VAL Acc] Source {}: {:.2f}%".format(source_name, source_acc))
                cnt += 1
        print("[VAL Acc] Avg {:.2f}%".format(total_acc / cnt))

        # Early stop
        is_best_acc = total_acc > best_acc
        if is_best_acc:
                print("VAL Acc improve from {:.2f}% to {:.2f}%".format(best_acc/cnt, total_acc/cnt))
                cur_patience = 0
        else:
            cur_patience += 1
        if args.lr_schedule == "cosine" and (cur_patience > 0 and cur_patience % 4 == 0):
                alpha_kd = ReduceWeightOnPlateau(alpha_kd, args.decay_factor)

        # Save 
        best_acc = max(total_acc,best_acc)
        if  is_best_acc:
            save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': student_model.state_dict(),
                'best_acc': best_acc,
                'optimizer': optimizer.state_dict()},
            path=args.checkpoint_path
            )
            print('Save best model')


        if args.early_stop and (cur_patience == args.patience):
            print("Early stopping ...")
            return 
        
    return 
        

        

if __name__ == "__main__":

    args = parse_args()
    
    if args.savepath and not os.path.isdir(args.savepath):
        os.makedirs(args.savepath)


