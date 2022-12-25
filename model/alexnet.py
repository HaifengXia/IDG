import torch.nn as nn
import torch
import torch.nn.init as init
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo
from model.Generator import Generator
from model.Gdiscriminator import GDiscriminator
from model.Discriminator import Discriminator
from torchvision.models import AlexNet

__all__ = ['AlexNet', 'alexnet']

model_urls = {
    'alexnet': 'https://download.pytorch.org/models/alexnet-owt-4df8aa71.pth',
}


def alexnet(num_classes, pretrained=True):
    """AlexNet model architecture from the
    `"One weird trick..." <https://arxiv.org/abs/1404.5997>`_ paper.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = AlexNet()
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['alexnet']))
        print('Load pre trained model')
    num_ftrs = model.classifier[-1].in_features
    model.classifier[-1] = nn.Linear(num_ftrs, num_classes)
    nn.init.xavier_uniform_(model.classifier[-1].weight, .1)
    nn.init.constant_(model.classifier[-1].bias, 0.)
    return model


class DGalexnet(nn.Module):
    def __init__(self, opts, pretrained=True, grl=True):
        super(DGalexnet, self).__init__()
        self.num_domains = opts['num_domains']
        self.num_classes = opts['num_classes']
        self.num_domains = self.num_domains
        self.base_model = alexnet(self.num_classes, pretrained=pretrained)
        self.Generator = Generator(opts)
        self.Gdiscriminator = GDiscriminator([4096, 256, 2], grl=grl, reverse=True)
        self.discriminator = Discriminator([4096, 1024, 1024, self.num_domains], grl=grl, reverse=True)
        self.feature_layers = nn.Sequential(*list(self.base_model.classifier.children())[:-1])
        self.fc = list(self.base_model.classifier.children())[-1]

    def forward(self, x, training=True):
        x = self.base_model.features(x)
        x = x.view(x.size(0), 256 * 6 * 6)
        x = self.feature_layers(x)

        if training:
            G_x = self.Generator(x)
            concat_x = torch.cat((x, G_x), dim=0)
            concat_class = self.fc(concat_x)
            concat_dis = self.Gdiscriminator(concat_x)
            concat_domain = self.discriminator(concat_x)
            output_class = concat_class[:x.size(0), :]
            G_class = concat_class[x.size(0):, :]
            output_domain = concat_domain[:x.size(0), :]
            G_domain = concat_domain[:x.size(0), :]
            return output_class, output_domain, x, G_x, G_class, G_domain, concat_dis
        else:
            output_class = self.fc(x)
            output_domain = self.discriminator(x)
            return output_class