import argparse, configparser


def parse_args():
    parser = argparse.ArgumentParser("train")
    parser.add_argument("-c", "--config_file", default="model_config.conf", type=str, help='Config file')
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





def train(args):
    # We need to do the imports AFTER the logger init
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

    # Init
    device = 'cuda' if args.num_gpu else 'cpu'
    lr = float(args.lr)
    alpha_kd = float(args.kd_alpha)
    epochs = int(args.epochs)
    early_stop = int(args.early_stop)
    decay_factor = float(args.decay_factor)

    # Logger
    logger = None
    if args.use_comet:
        comet_ml.init()
        logger = comet_ml.Experiment()
        logger.set_name(args.comet_name)
        logger.log_parameters(parameters=vars(args))

    # Load datasets and models
    train_loaders, val_loaders = create_dataloader(args, train=True)
    print("Dataset available in train_loaders: ", " / ".join([n for n in train_loaders]))
    print("Dataset available in val_loaders: ", " / ".join([n for n in val_loaders]))
    teacher_model, student_model = load_models(args.input_model, args.network, num_gpu = args.num_gpu)
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
        lr_scheduler = None
        print(f"Input: {args.lr_schedule}, No learning rate schedule applied ... ")
        

    # Pre-evaluation
    _, test_acc = test_model(train_loaders['val'], student_model, criterion, device=device) 
    if logger: logger.log_metric('start_acc',test_acc,0)
    print("Start Target Validation ACC: {:.2f}%".format(test_acc))


    # ------- START TRAINING ------- #
    best_acc = 0
    cur_patience = 0 # Early stop and saving
    print(f"Start training in {epochs} epochs")

    for epoch in range(epochs):
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
            outputs = student_model(inputs)

            # KD loss
            loss_main = criterion(outputs, targets)
            loss_kd = loss_fn_kd(outputs, targets, teacher_outputs)
            loss_kd = loss_clampping(loss_kd, 0, 1800)

            # Total loss
            loss = loss_main  + alpha_kd*loss_kd
            
            # Log and display
            if logger:
                 logger.log_metric('losses/loss', loss.item(), step)
                 logger.log_metric('losses/loss_main', loss_main.item(), step)
                 logger.log_metric('losses/loss_kd', loss_kd.item(), step)
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
        if lr_scheduler: lr_scheduler.step()


        # ----- Epoch Validation ------ #

        # Current task
        _, test_acc = test_model(train_loaders['val'], student_model, criterion, device=device)
        total_acc = test_acc
        print("[VAL Acc] Target: {:.2f}%".format(test_acc))
        if logger: logger.log_metric('acc/target_val_acc', test_acc, step=step, epoch=epoch)

        # Past tasks
        cnt = 1
        for source_name in val_loaders:
                _, source_acc = test_model(val_loaders[source_name], student_model, criterion, device=device, source_name=source_name)
                total_acc += source_acc
                print("[VAL Acc] Source {}: {:.2f}%".format(source_name, source_acc))
                if logger: logger.log_metric(f'acc/{source_name}_val_acc', source_acc, step=step, epoch=epoch)
                cnt += 1
        print("[VAL Acc] Avg {:.2f}%".format(total_acc / cnt))
        if logger: logger.log_metric('acc/val_acc', total_acc/cnt, step=step, epoch=epoch)

        
        is_best_acc = total_acc > best_acc
        if is_best_acc:
            print("VAL Acc improve from {:.2f}% to {:.2f}%".format(best_acc/cnt, total_acc/cnt))
            cur_patience = 0
        else:
            cur_patience += 1
        if args.lr_schedule == "cosine" and (cur_patience > 0 and cur_patience % 4 == 0):
                alpha_kd = ReduceWeightOnPlateau(alpha_kd, decay_factor)

        # Save 
        best_acc = max(total_acc,best_acc)
        if  is_best_acc or epoch==0:
            if args.network=="ViT":
                save_checkpoint(model=student_model, path=args.output_dir)
            else:   
                save_checkpoint({
                    'epoch': epoch + 1,
                    'state_dict': student_model.state_dict(),
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
    return 
        

        

if __name__ == "__main__":

    args = parse_args()

    if args.use_comet:
         import dotenv
         import comet_ml
         dotenv.load_dotenv()
    
    train(args)


