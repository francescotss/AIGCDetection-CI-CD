import os, random, copy
import numpy as np
from PIL import Image
from collections import OrderedDict

import torch
from torchvision import transforms, datasets
from torch.utils.data import Dataset, DataLoader

from utils.train_utils import set_seeds


class CustomDataset(Dataset):
    def __init__(self, data, target, transform=None):
        self.data = data
        self.target = target
        self.transform = transform
    
    def __len__(self):
        return len(self.target)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        path = self.data[idx]
        img = Image.open(path)
        img = img.convert('RGB')

        if self.transform:
            img = self.transform(img)
        return img, int(self.target[idx])
    



def create_dataloader(args, train:bool):
    print('\n\n\n------ Creating Loaders ------\nGPU num is' , args.num_gpu)
    os.environ['CUDA_VISIBLE_DEVICES'] =str(args.num_gpu)
    set_seeds()

    source_datasets = args.source_datasets
    target_dataset = args.target_dataset
    batch_size = int(args.batch_size)

    

    #train & valid
    train_aug, val_aug = _get_augs(args)
    #if not args.train: # Test
    #    print(f"\n===> Starting Test data loader from {path_data}")
    #    print("Source:", dict_source['source'])
    #    print("Target:", name_target)
    #   dicLoader,dicCoReD = _make_test_dataloader(path_data, dict_source['source'],
    #                                               name_target, train_aug=train_aug, val_aug=val_aug,
    #                                                batch_size=args.batch_size,
    #                                                TRAIN_MODE=args.train, MODE_BALANCED_DATA=False)
    if train:
        print('\n===> Making Loader for Continual Learning..')
        training_loaders,testing_loaders = _make_train_dataloader(target_dataset, 
                                                    ds_source_dirs=source_datasets,
                                                    train_aug=train_aug,
                                                    val_aug=val_aug,
                                                    batch_size=batch_size)
    return training_loaders, testing_loaders



def _get_augs(args):
    resize_func = transforms.RandomCrop(int(args.resolution), pad_if_needed=True)

    if bool(args.flip):
        flip_func = transforms.RandomHorizontalFlip(p=0.5)
    else:
        flip_func = transforms.Lambda(lambda img: img)

    if args.network == "ViT":
        normalize_func = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    else:
        normalize_func = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

        
    train_aug = transforms.Compose([
        resize_func,
        flip_func,
        transforms.ToTensor(),
        normalize_func,
    ])

    val_aug = transforms.Compose([
        resize_func,
        transforms.ToTensor(),
        normalize_func,
    ])

    return train_aug, val_aug



#TODO
def _make_test_dataloader(dir,
                    name_source,
                    name_target,
                    name_mixed_folder='',
                    train_aug=None,
                    val_aug=None,
                    batch_size=128,
                    TRAIN_MODE=True,
                    MODE_BALANCED_DATA = False
                    ):
    
    dic_CoReD = None
    train_target_dataset = datasets.ImageFolder(dir,transform=None)
    new_samples = train_target_dataset.samples

    train_target_dataset = CustomDataset(np.array(new_samples)[:,0], np.array(new_samples)[:,1], val_aug)
    train_target_loader = DataLoader(train_target_dataset,
                                    batch_size=batch_size,
                                    shuffle=True,
                                    num_workers=2,
                                    pin_memory=True
                                    )
    dic = {'test_dataset':train_target_loader}

    return dic, dic_CoReD



def _make_train_dataloader(ds_target_dir,
                            ds_source_dirs,
                            train_aug=None,
                            val_aug=None,
                            batch_size=128,
                            ):
    
    NUM_WORKERS = 2 #TODO: Implement in config
    
    train_dir = os.path.join(ds_target_dir, 'train')
    val_target_dir = os.path.join(ds_target_dir, 'val')

    assert os.path.exists(train_dir) and os.path.exists(val_target_dir), 'Training Dataset does not exist'

    #For Validataion
    validation_loaders = OrderedDict()
    for name in ds_source_dirs.split(','):
            assert os.path.exists(name), f"Validation Dataset {name} does not exist"

            print('===> Making Loader :', name)
            path = os.path.join(name, "val")
            _loader = DataLoader(datasets.ImageFolder(path, val_aug),
                                            batch_size=batch_size,
                                            shuffle=False,
                                            num_workers=NUM_WORKERS,
                                            pin_memory=True
                                            )
            validation_loaders[name] = copy.deepcopy(_loader)



    print("DATASET PATHS")
    print('val_source_dir ' ,ds_source_dirs)
    print('val_target_dir ' ,val_target_dir)
    print('train_dir ' ,train_dir)

    
    train_target_dataset = datasets.ImageFolder(train_dir,transform=None)
    train_target_dataset = CustomDataset(np.array(train_target_dataset.samples)[:,0],np.array(train_target_dataset.targets),train_aug)
    
    train_target_loader = DataLoader(train_target_dataset,
                                    batch_size=batch_size,
                                    shuffle=True,
                                    num_workers=NUM_WORKERS,
                                    pin_memory=True
                                    )

    val_target_loader = DataLoader(datasets.ImageFolder(val_target_dir, val_aug),
                                batch_size=batch_size,
                                shuffle=False,
                                num_workers=NUM_WORKERS,
                                pin_memory=True
                                )

    
    train_loaders = OrderedDict()
    train_loaders['train'] = train_target_loader
    train_loaders['val'] = val_target_loader

    return train_loaders, validation_loaders