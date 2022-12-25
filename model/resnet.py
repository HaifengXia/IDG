from torchvision.models import resnet18
from model.Discriminator import Discriminator
from model.Generator import Generator
from model.Gdiscriminator import GDiscriminator
import torch.nn as nn
import torch
import torch.nn.init as init

def resnet(num_classes, pretrained=True):
    model = resnet18(pretrained=pretrained)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)
    nn.init.xavier_uniform_(model.fc.weight, .1)
    nn.init.constant_(model.fc.bias, 0.)
    return model

class DGresnet(nn.Module):
    def __init__(self, opts, pretrained=True, grl=True):
        super(DGresnet, self).__init__()
        self.num_domains = opts['num_domains']
        self.num_classes = opts['num_classes']
        self.base_model = resnet(num_classes=self.num_classes, pretrained=pretrained)
        self.Generator = Generator(opts)
        self.Gdiscriminator = GDiscriminator([512, 256, 2], grl=grl, reverse=True)
        self.discriminator = Discriminator([512, 1024, 1024, self.num_domains], grl=grl, reverse=True)

        
    def forward(self, x, training=True):
        x = self.base_model.conv1(x)
        x = self.base_model.bn1(x)
        x = self.base_model.relu(x)
        x = self.base_model.maxpool(x)
        
        x = self.base_model.layer1(x)
        x = self.base_model.layer2(x)
        x = self.base_model.layer3(x)
        x = self.base_model.layer4(x)
        
        x = self.base_model.avgpool(x)
        x = x.view(x.size(0), -1)


        if training:
            G_x = self.Generator(x)
            concat_x = torch.cat((x, G_x), dim=0)
            concat_class = self.base_model.fc(concat_x)
            concat_dis = self.Gdiscriminator(concat_x)
            concat_domain = self.discriminator(concat_x)
            output_class = concat_class[:x.size(0),:]
            G_class = concat_class[x.size(0):, :]
            output_domain = concat_domain[:x.size(0),:]
            G_domain = concat_domain[:x.size(0),:]
            return output_class, output_domain, x, G_x, G_class, G_domain, concat_dis
        else:
            output_class = self.base_model.fc(x)
            output_domain = self.discriminator(x)
            return x, output_class