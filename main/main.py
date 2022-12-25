import sys
sys.path.append('../')

from torch.utils.data import DataLoader
import torch
import argparse
import os
from util.util import *
from data.prepare_data import prepare_data
from train.eval import *
import configs

if __name__ == '__main__':

    for run_times in range(10):
        print('************************************')
        print(run_time)
        print('************************************')
    
        opts = getattr(configs, 'config')

        dataloader = prepare_data(opts)

        path = opts['save_root']
        if not os.path.isdir(path):
            os.makedirs(path)

        device = torch.device("cuda:" + str(opts['gpu']) if torch.cuda.is_available() else "cpu")

        num_epoch = opts['num_epoch']
        lr_step = opts['lr_step']

        # disc_dim = get_disc_dim(args.train, args.clustering, len(source_domain), args.num_clustering) num_domain

        model = get_model(opts['model'], opts['train'])(opts=opts, pretrained=True)

        model = model.to(device)
        model_lr = get_model_lr(opts['model'], opts['train'], model, fc_weight=opts['fc_weight'], disc_weight=opts['disc_weight'], gene_weight=opts['gene_weight'])
        optimizers = [get_optimizer(model_part, opts['lr'] * alpha, opts['momentum'], opts['weight_decay']) for model_part, alpha in model_lr]

        if opts['scheduler'] == 'inv':
            schedulers = [get_scheduler(opts['scheduler'])(optimizer=opt, alpha=10, beta=0.75, total_epoch=num_epoch)
                         for opt in optimizers]
        elif opts['scheduler'] == 'step':
            schedulers = [get_scheduler(opts['scheduler'])(optimizer=opt, step_size=lr_step, gamma=opts['lr_decay_gamma'])
                         for opt in optimizers]
        else:
            raise ValueError('Name of scheduler unknown %s' %opts['scheduler'])

        best_acc = 0.0
        test_acc = 0.0
        best_epoch = 0

        for epoch in range(num_epoch):

            print('Epoch: {}/{}, Lr: {:.6f}'.format(epoch, num_epoch-1, optimizers[0].param_groups[0]['lr']))
            print('Temporary Best Accuracy is {:.4f} ({:.4f} at Epoch {})'.format(test_acc, best_acc, best_epoch))

            hist = [1, 2, 3]
            weight = 1. / np.histogram(hist, bins=opts['num_domains'])[0]
            weight = weight / weight.sum() * model.num_domains
            weight = torch.from_numpy(weight).float().to(device)

            model, optimizers = get_train(opts['train'])(
                model=model, dataloader=dataloader, optimizers=optimizers, device=device,
                epoch=epoch, num_epoch=num_epoch, filename=path+'/source_train.txt', entropy=opts['entropy'],
                disc_weight=weight, entropy_weight=opts['entropy_weight'], grl_weight=opts['grl_weight'], opts=opts, model_select=opts['model_select'])

            if epoch % opts['eval_step'] == 0:
                target_test = iter(dataloader['test'])
                acc_ = eval_model(model, target_test, epoch, path+'/target_test.txt')


            test_acc = acc_
            if acc_ >= best_acc:
                best_acc = acc_
                best_epoch = epoch

            for scheduler in schedulers:
                scheduler.step()