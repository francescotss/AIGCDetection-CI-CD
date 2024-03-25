import os, logging, json, base64
import argparse
from io import BytesIO
from PIL import Image

import torch
from torchvision.transforms.functional import pil_to_tensor, normalize, crop

def parse_args():
    parser = argparse.ArgumentParser("score")
    parser.add_argument("--input_model", default='')
    parser.add_argument("--raw_data")

    args = parser.parse_args()

    os.environ["AZUREML_MODEL_DIR"] = args.input_model
    
    print(args)
    return args



def init():
    """
    This function is called when the container is initialized/started, typically after create/update of the deployment.
    You can write the logic here to perform init operations like caching the model in memory
    """
    global model
    # AZUREML_MODEL_DIR is an environment variable created during deployment.
    # It is the path to the model folder (./azureml-models/$MODEL_NAME/$VERSION)
    # Please provide your model's folder name if there is one
    model_path = os.path.join(
        os.getenv("AZUREML_MODEL_DIR"), "model_output/model.pth"
    )
    
    if os.path.isdir(model_path):
        model_path = os.path.join(model_path, 'model.pth')
        
    assert os.path.isfile(model_path), f"Pretrained {model_path} not found"

    checkpoint = torch.load(model_path, map_location='cpu')
    model = resnet50(pretrained=False, num_classes=1)
    model.compability_layer = nn.Linear(1, 2, bias=False)
    model.load_state_dict(checkpoint['state_dict'])
    model.compability_layer.weight = nn.Parameter(torch.tensor([[0], [1.]]), requires_grad=False)
    model.eval()
    logging.info("Init complete")


def run(raw_data):
    """
    This function is called for every invocation of the endpoint to perform the actual scoring/prediction.
    In the example we extract the data from the json input and call the scikit-learn model's predict()
    method and return the result back
    """
    logging.info("request received")
    image = json.loads(raw_data)["image"]
    image = Image.open(BytesIO(base64.b64decode(image))).convert('RGB')
    image = pil_to_tensor(image).float()
    image = crop(image, top=0, left=0, height=128, width=128)
    image = normalize(image, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    image = image.unsqueeze(0)
    
    with torch.no_grad():
        outputs = model(image)
        _, predicted = torch.max(outputs, 1)

    print(outputs, predicted)
    logging.info("Request processed")
    return predicted.item()



# RESNET

import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo


__all__ = ['ResNet', 'resnet18', 'resnet34', 'resnet50', 'resnet101',
           'resnet152']


model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = conv1x1(inplanes, planes)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes, stride)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = conv1x1(planes, planes * self.expansion)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=1000, zero_init_residual=False):
        super(ResNet, self).__init__()
        self.inplanes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)
        self.compability_layer = None


        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

        
    def features(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        return x
        
        
    def forward(self, x, include_features=False):
        ft = self.features(x)

        x = self.avgpool(ft)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        x = self.compability_layer(x)

        if include_features:
            return ft, x
        return x


def resnet18(pretrained=False, num_classes=1000):
    """Constructs a ResNet-18 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    if pretrained:
        model = ResNet(BasicBlock, [2, 2, 2, 2], num_classes=1000)
        model.load_state_dict(model_zoo.load_url(model_urls['resnet18']))
        model.fc = nn.Linear(512 * BasicBlock.expansion, num_classes)
    else: 
        model = ResNet(BasicBlock, [2, 2, 2, 2], num_classes)
    return model


def resnet34(pretrained=False, **kwargs):
    """Constructs a ResNet-34 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [3, 4, 6, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet34']))
    return model


def resnet50(pretrained=False, num_classes=1000):
    """Constructs a ResNet-50 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    if pretrained:
            
        model = ResNet(Bottleneck, [3, 4, 6, 3], num_classes=1000)
        model.load_state_dict(model_zoo.load_url(model_urls['resnet50']))
        model.fc = nn.Linear(512 * Bottleneck.expansion, num_classes)
    else: 
        model = ResNet(Bottleneck, [3, 4, 6, 3], num_classes)
    return model


def resnet101(pretrained=False, **kwargs):
    """Constructs a ResNet-101 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 23, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet101']))
    return model


def resnet152(pretrained=False, **kwargs):
    """Constructs a ResNet-152 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 8, 36, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet152']))
    return model



if __name__ == "__main__":

    
    args = parse_args()
    init()
    cnt = 0
    for img in os.listdir("/home/fra/AIGCDetection-CI-CD/datasets/gaugan/val/1_fake/"):   
        data = {}
        with open(f"/home/fra/AIGCDetection-CI-CD/datasets/gaugan/val/1_fake/{img}", mode='rb') as file:
            img = file.read()
        data['image'] = base64.encodebytes(img).decode('utf-8')

        raw_data = json.dumps(data)
        print(raw_data)
        run(raw_data)
        if cnt > 10:
            break
        cnt += 1
        #run(args.raw_data)
