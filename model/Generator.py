import torch.nn as nn
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo
import torch

class Generator(nn.Module):
    def __init__(self, opts):
        super(Generator, self).__init__()
        self.gpu = opts['gpu']
        self.num_domain = opts['num_domains']
        self.sample_size = opts['sample_size']
        self.zdim = opts['zdim']
        self.dedim = opts['dedim']
        self.Gdim = opts['Gdim']
        # self.encoder = nn.Linear(512, 1024)
        self.mean = nn.Linear(self.Gdim, self.zdim)
        self.log_sigma = nn.Linear(self.Gdim, self.zdim)
        self.decoder = nn.Sequential(
            # nn.Linear(self.zdim, self.dedim),
            nn.ReLU(),
            nn.Linear(self.zdim, self.Gdim),
            #nn.BatchNorm1d(512),
            # nn.ReLU(),
            # nn.Sigmoid()
            )
        # self.encoder_relu = nn.ReLU()
        # self.encoder_BN = nn.BatchNorm1d(1024)
        

    def spilt(self, x):

        batch_size = x.size(0) / self.num_domain

        for i in range(self.num_domain):
            temp = x[int(i*batch_size):int((i+1) * batch_size), :]
            if i == 0:
                x_dom = temp
            else:
                x_dom = x_dom + temp

        return x_dom / 3.0

    def statistic(self, x):
        
        mu = self.mean(x)
        log_sig = self.log_sigma(x)

        return mu, log_sig

    def get_encoder(self, mu, log_sig):

        device = torch.device("cuda:" + str(self.gpu))
        eps = torch.randn(mu.size(0), self.zdim).to(device)
        z = mu + eps * torch.sqrt(1e-8 + torch.exp(log_sig))

        return z


    def forward(self, x):
        
        #spilt the features into multi-source domains
        x = self.spilt(x)
        #learn the statistics of features
        mu, log_sig = self.statistic(x)
        
        mu = mu.repeat(self.sample_size, 1)
        log_sig = log_sig.repeat(self.sample_size, 1)

        z = self.get_encoder(mu, log_sig)

        G_x = self.decoder(z)

        return G_x