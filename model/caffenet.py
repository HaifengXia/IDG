from collections import OrderedDict
from model.Discriminator import *
import torch
from torch import nn as nn
from model.Generator import Generator
from model.Gdiscriminator import GDiscriminator
from model.Discriminator import Discriminator


class Id(nn.Module):
    def __init__(self):
        super(Id, self).__init__()

    def forward(self, x):
        return x

class AlexNetCaffe(nn.Module):
    def __init__(self, num_classes=100, dropout=True):
        super(AlexNetCaffe, self).__init__()
        print("Using Caffe AlexNet")
        self.features = nn.Sequential(OrderedDict([
            ("conv1", nn.Conv2d(3, 96, kernel_size=11, stride=4)),
            ("relu1", nn.ReLU(inplace=True)),
            ("pool1", nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True)),
            ("norm1", nn.LocalResponseNorm(5, 1.e-4, 0.75)),
            ("conv2", nn.Conv2d(96, 256, kernel_size=5, padding=2, groups=2)),
            ("relu2", nn.ReLU(inplace=True)),
            ("pool2", nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True)),
            ("norm2", nn.LocalResponseNorm(5, 1.e-4, 0.75)),
            ("conv3", nn.Conv2d(256, 384, kernel_size=3, padding=1)),
            ("relu3", nn.ReLU(inplace=True)),
            ("conv4", nn.Conv2d(384, 384, kernel_size=3, padding=1, groups=2)),
            ("relu4", nn.ReLU(inplace=True)),
            ("conv5", nn.Conv2d(384, 256, kernel_size=3, padding=1, groups=2)),
            ("relu5", nn.ReLU(inplace=True)),
            ("pool5", nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True)),
        ]))
        self.classifier = nn.Sequential(OrderedDict([
            ("fc6", nn.Linear(256 * 6 * 6, 4096)),
            ("relu6", nn.ReLU(inplace=True)),
            ("drop6", nn.Dropout() if dropout else Id()),
            ("fc7", nn.Linear(4096, 4096)),
            ("relu7", nn.ReLU(inplace=True)),
            ("drop7", nn.Dropout() if dropout else Id())]))

        self.class_classifier = nn.Linear(4096, num_classes)


    def forward(self, x, lambda_val=0):
        x = self.features(x*57.6)
        #57.6 is the magic number needed to bring torch data back to the range of caffe data, based on used std
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return self.class_classifier(x)

def caffenet(num_classes, num_domains=None, pretrained=True):
    model = AlexNetCaffe(num_classes)
    for m in model.modules():
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight, .1)
            nn.init.constant_(m.bias, 0.)

    if pretrained:
        state_dict = torch.load("C:/Users/haifxia/Desktop/Transfer_Code/torch_ICLR/DG/EISNet-master/pretrained/alexnet_caffe.pth.tar")
        del state_dict["classifier.fc8.weight"]
        del state_dict["classifier.fc8.bias"]
        model.load_state_dict(state_dict, strict=False)
    return model

class DGcaffenet(nn.Module):
    def __init__(self, opts, pretrained=True, grl=True):
        super(DGcaffenet, self).__init__()
        self.num_domains = opts['num_domains']
        self.num_classes = opts['num_classes']
        # self.num_domains = num_domains
        self.base_model = caffenet(self.num_classes, pretrained=pretrained)
        self.Generator = Generator(opts)
        self.Gdiscriminator = GDiscriminator([4096, 256, 2], grl=grl, reverse=True)
        self.discriminator = Discriminator([4096, 1024, 1024, self.num_domains], grl=grl, reverse=True)
        
    def forward(self, x, training=True):
        x = self.base_model.features(x*57.6)
        x = x.view(x.size(0), 256 * 6 * 6)
        x = self.base_model.classifier(x)

        if training:
            G_x = self.Generator(x)
            concat_x = torch.cat((x, G_x), dim=0)
            concat_class = self.base_model.class_classifier(concat_x)
            concat_dis = self.Gdiscriminator(concat_x)
            concat_domain = self.discriminator(concat_x)
            output_class = concat_class[:x.size(0), :]
            G_class = concat_class[x.size(0):, :]
            output_domain = concat_domain[:x.size(0), :]
            G_domain = concat_domain[:x.size(0), :]
            return output_class, output_domain, x, G_x, G_class, G_domain, concat_dis
        else:
            output_class = self.base_model.class_classifier(x)
            output_domain = self.discriminator(x)
            return x, output_class