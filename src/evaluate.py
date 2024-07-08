import argparse, configparser


def parse_args():
    parser = argparse.ArgumentParser("evaluate")
    parser.add_argument("--network", help="Supported Networks: ResNet, ResNet18, Xception, MobileNet2, ViT")
    parser.add_argument("--input_model")
    parser.add_argument("--source_datasets", help="comma-separated list of directories. Example: /datasets/ds1,/datasets/ds2")
    parser.add_argument("--use_comet", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--comet_name", help="Experiment name for comet logging")
    parser.add_argument("--use_cpu", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--batch_size", default=128)
    parser.add_argument("--resolution", default=128)


    args = parser.parse_args()
    return args


def evaluate(args, global_writer=None):

    
    import torch
    import torch.nn as nn
    import numpy as np

    from sklearn.metrics import classification_report, roc_auc_score, accuracy_score, average_precision_score
    from tqdm import tqdm

    from utils.data_loader import create_dataloader
    from utils.model_loader import load_models
    from utils.train_utils import AverageMeter

    # Init
    device = 'cpu' if args.use_cpu else 'cuda'
    args.num_gpu = '0' if device=='cuda' else ''

    # Logger
    logger = None
    if args.use_comet:
        comet_ml.init()
        logger = comet_ml.Experiment()
        logger.set_name(args.comet_name)
        logger.log_parameters(parameters=vars(args))


    # Load datasets and models
    dataloaders = create_dataloader(args, train=False)
    _, model = load_models(args.input_model, args.network, num_gpu = args.num_gpu)
    criterion = nn.CrossEntropyLoss().cuda()


    tot_avg_acc, real_avg_acc, fake_avg_acc = 0.0, 0.0 ,0.0
    for ds_name in dataloaders:

        # Init
        frindly_name = ds_name.split("/")[-1]
        print(frindly_name)
        global best_acc
        correct, total =0,0
        losses = AverageMeter()
        acc_real = AverageMeter()
        acc_fake = AverageMeter()
        target=[]
        output = []
        y_true=np.zeros((0,2),dtype=np.int8)
        y_pred=np.zeros((0,2),dtype=np.int8)


        with torch.no_grad():
          model.eval()
          model.cuda()

          for inputs, targets in tqdm(dataloaders[ds_name], ncols=50):
              # Predict
              inputs, targets = inputs.to(device), targets.to(device)
              outputs = model(inputs)
              
              loss = criterion(outputs, targets)
              _, predicted = torch.max(outputs, 1)
              correct = (predicted == targets).squeeze()
              total += len(targets)
              losses.update(loss.data.tolist(), inputs.size(0))
              _y_pred = outputs.cpu().detach()
              _y_gt = targets.cpu().detach().numpy()
              acc = [0, 0]
              class_total = [0, 0]
              for i in range(len(targets)):
                  label = targets[i]
                  acc[label] += 1 if correct[i].item() == True else 0
                  class_total[label] += 1

              losses.update(loss.data.tolist(), inputs.size(0))
              if (class_total[0] != 0):
                  acc_real.update(acc[0] / class_total[0])
              if (class_total[1] != 0):
                  acc_fake.update(acc[1] / class_total[1])

              target.append(_y_gt)
              output.append(_y_pred.numpy()[:,1])
              _y_true = np.array(torch.zeros(targets.shape[0],2), dtype=np.int8)
              _y_gt = _y_gt.astype(int)
              for _ in range(len(targets)):
                  _y_true[_][_y_gt[_]] = 1
              y_true = np.concatenate((y_true,_y_true))
              a = _y_pred.argmax(1)
              _y_pred = np.array(torch.zeros(_y_pred.shape).scatter(1, a.unsqueeze(1), 1),dtype=np.int8)
              y_pred = np.concatenate((y_pred,_y_pred))

          n_real_samples = np.count_nonzero(y_true, axis=0)[0]
          n_fake_samples = np.count_nonzero(y_true, axis=0)[1]
          acc = accuracy_score(y_true, y_pred)
          ap = average_precision_score(y_true, y_pred)

          pre_rec_f1 = classification_report(y_true, y_pred, output_dict=True)


          print(f"\nLoss:{losses.avg:.4f} | Acc:{acc:.4f} | Acc Real:{acc_real.avg:.4f} | Acc Fake:{acc_fake.avg:.4f} | Ap:{ap:.4f}")
          print(f'Num reals: {n_real_samples}, Num fakes: {n_fake_samples}')


          tot_avg_acc += acc
          real_avg_acc += acc_real.avg
          fake_avg_acc += acc_fake.avg

          if logger:
            logger.log_metrics(
                  {
                      "real_acc": acc_real.avg*100.,
                      "fake_acc": acc_fake.avg*100.,
                      "tot_acc": acc*100.,
                      "ap": ap*100.,
                  },
                  prefix="test_"+str(frindly_name)
              )
            logger.log_metrics(
                  {
                      "num_reals": n_real_samples,
                      "num_fakes": n_fake_samples,
                  },
                  prefix="test_"+str(frindly_name)
              )
            logger.log_metrics(
                pre_rec_f1["0"],
                prefix="test_"+str(frindly_name)+"_real_"
            )
            logger.log_metrics(
                pre_rec_f1["1"],
                prefix="test_"+str(frindly_name)+"_fake_"
            )


    total_ds = len(dataloaders)
    if logger is not None:
              logger.log_metrics(
                    {
                        "real_acc": (real_avg_acc/total_ds)*100.,
                        "fake_acc": (fake_avg_acc/total_ds)*100.,
                        "tot_acc": (tot_avg_acc/total_ds)*100.,
                    },
                    prefix='test_'+"avg_acc"
                )

    print(f"Avg: | Acc:{tot_avg_acc/total_ds:.4f} | Acc Real:{real_avg_acc/total_ds:.4f} | Acc Fake:{fake_avg_acc/total_ds:.4f}")
    if logger: logger.end()



if __name__ == "__main__":

    args = parse_args()

    if args.use_comet:
         import dotenv
         import comet_ml
         dotenv.load_dotenv()
    
    evaluate(args)
