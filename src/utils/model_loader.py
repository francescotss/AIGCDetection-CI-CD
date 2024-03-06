import os
import torch
import torch.nn as nn

from networks.mobilenetv2 import mobilenet_v2
from networks.resnet import resnet18, resnet50
from networks.xceptionnet import xception

def load_models(weight, nameNet='ResNet', num_gpu='', TrainMode=True):
    print("\n\n\n ------ Loading models 2 ------")
    teacher_model, student_model = None,None
    device = 'cuda' if num_gpu else 'cpu'
    checkpoint = None
    download_weights = False
    
    assert weight == "" or os.path.isfile(weight), "Pretrained weights not found"
    
    if weight != "":
        print(f"Loading {nameNet} from {weight}")
        checkpoint = torch.load(weight)  
    else:
        download_weights = True
        

    #model load
    if nameNet=='Xception':
        teacher_model = xception(num_classes=2, pretrained='imagenet' if download_weights else '')
        student_model = xception(num_classes=2, pretrained='imagenet' if download_weights else '')
    elif nameNet=='ResNet':
        teacher_model = resnet50(pretrained=download_weights, num_classes=1)
        student_model = resnet50(pretrained=download_weights, num_classes=1)
    elif nameNet=='ResNet18':
        teacher_model = resnet18(pretrained=download_weights, num_classes=1)
        student_model = resnet18(pretrained=download_weights, num_classes=1)
    elif nameNet=='MobileNet2':
        teacher_model = mobilenet_v2(pretrained=download_weights, num_classes=2)
        student_model = mobilenet_v2(pretrained=download_weights, num_classes=2)
    else:
        raise NotImplementedError(f"{nameNet} not implemented")
                
    if ',' in num_gpu :
        teacher_model = nn.DataParallel(teacher_model)
        student_model = nn.DataParallel(student_model)

    #weight load
    if checkpoint:
        if nameNet=='ResNet' or nameNet=='ResNet18':
            if 'model' in checkpoint:
                teacher_model.load_state_dict(checkpoint['model'])
                student_model.load_state_dict(checkpoint['model'])
                teacher_model.compability_layer = nn.Linear(1, 2, bias=False)
                student_model.compability_layer = nn.Linear(1, 2, bias=False)
            else:
                teacher_model.compability_layer = nn.Linear(1, 2, bias=False)
                student_model.compability_layer = nn.Linear(1, 2, bias=False)
                teacher_model.load_state_dict(checkpoint['state_dict'])
                student_model.load_state_dict(checkpoint['state_dict'])
                
            teacher_model.compability_layer.weight = nn.Parameter(torch.tensor([[0], [1.]]), requires_grad=False)
            student_model.compability_layer.weight = nn.Parameter(torch.tensor([[0], [1.]]), requires_grad=False)   
            print("Loaded")
        else:
            teacher_model.load_state_dict(checkpoint['state_dict'])
            student_model.load_state_dict(checkpoint['state_dict'])
    elif nameNet=='ResNet18' or nameNet=='ResNet':
        teacher_model.compability_layer = nn.Linear(1, 2, bias=False)
        student_model.compability_layer = nn.Linear(1, 2, bias=False)
        teacher_model.compability_layer.weight = nn.Parameter(torch.tensor([[0], [1.]]), requires_grad=False)
        student_model.compability_layer.weight = nn.Parameter(torch.tensor([[0], [1.]]), requires_grad=False)
        
    if TrainMode:  
        student_model.train()
    else:
        student_model.eval()
    teacher_model.eval()
    student_model.to(device)
    teacher_model.to(device)

    
    return teacher_model, student_model