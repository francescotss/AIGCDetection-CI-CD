import os, random, copy
import numpy as np
from PIL import Image
from collections import OrderedDict

import torch
from torchvision import transforms, datasets
from torchvision.transforms import v2
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
    batch_size = int(args.batch_size)


    

    if train:
        print('\n===> Making Loader for Continual Learning..')
        target_dataset = args.target_dataset
        train_aug, val_aug = _get_augs(args, True)
        training_loaders,testing_loaders = _make_train_dataloader(target_dataset, 
                                                    ds_source_dirs=source_datasets,
                                                    train_aug=train_aug,
                                                    val_aug=val_aug,
                                                    batch_size=batch_size)
        return training_loaders, testing_loaders
    else:
        print('\n===> Making Loader Testing')
        _, val_aug = _get_augs(args, False)
        return _make_test_dataloader(source_datasets, val_aug=val_aug,batch_size=batch_size)

        
    
    



def _get_augs(args, train):
    resize_func = transforms.RandomCrop(int(args.resolution), pad_if_needed=True)

    if args.network == "ViT":
        normalize_func = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    else:
        normalize_func = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    # Validation transforms for robustness test
    val_transforms = []
    if 'val_transforms' in args and args.val_transforms!='':
        for transform in args.val_transforms.split(","):
            if transform=="contrast":
                val_transforms.append(transforms.ColorJitter(contrast=(0.5,2)))
            elif transform=="brightness":
                val_transforms.append(transforms.ColorJitter(brightness=(0.5,2)))
            elif transform=="jpeg":
                val_transforms.append(v2.JPEG((10,90)))
            else:
                error = f'{transform} transform not implemented'
                raise NotImplementedError(error)




    train_aug = None
    if train:
        if bool(args.flip):
            flip_func = transforms.RandomHorizontalFlip(p=0.5)
        else:
            flip_func = transforms.Lambda(lambda img: img)

        train_aug = transforms.Compose([
            resize_func,
            flip_func,
            transforms.ToTensor(),
            normalize_func,
        ])

    val_aug = transforms.Compose(
        [resize_func]+
        val_transforms+
        [transforms.ToTensor(),
        normalize_func,
    ])

    return train_aug, val_aug




def _make_test_dataloader(paths, val_aug=None, batch_size=128):

    NUM_WORKERS = 2 #TODO: Implement in config

    #For Validataion
    validation_loaders = OrderedDict()
    for name in paths.split(','):
            path = os.path.join(name, "test")
            assert os.path.exists(path), f"Validation Dataset {path} does not exist"

            print('===> Making Loader :', path)
            _loader = DataLoader(datasets.ImageFolder(path, val_aug),
                                            batch_size=batch_size,
                                            shuffle=False,
                                            num_workers=NUM_WORKERS,
                                            pin_memory=True
                                            )
            validation_loaders[name] = copy.deepcopy(_loader)

    
    return validation_loaders



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