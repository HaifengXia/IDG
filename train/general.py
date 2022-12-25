from torch import nn
from util.util import split_domain
import torch
from numpy.random import *
import numpy as np
from loss.EntropyLoss import HLoss
from loss.MaximumSquareLoss import MaximumSquareLoss
from utils.utils import to_cuda, to_onehot
from utils.utils import to_cuda, accuracy

def get_samples(data_loader, data_iterator):
    try:
        sample = next(data_iterator)
    except StopIteration:
        data_iterator = iter(data_loader)
        sample = next(data_iterator)

    return sample, data_iterator

def CAS(data_loader, data_iterator):
    samples, data_iterator = get_samples(data_loader, data_iterator)

    inputs_1st = samples['Img_source_1']
    paths_1st = samples['Path_source_1']
    nums_1st = [len(paths) for paths in paths_1st]
    labels_1st = samples['Label_source_1']

    inputs_2nd = samples['Img_source_2']
    paths_2nd = samples['Path_source_2']
    nums_2nd = [len(paths) for paths in paths_2nd]
    labels_2nd = samples['Label_source_2']

    inputs_3rd = samples['Img_source_3']
    paths_3rd = samples['Path_source_3']
    nums_3rd = [len(paths) for paths in paths_3rd]
    labels_3rd = samples['Label_source_3']

    return inputs_1st, labels_1st, \
            inputs_2nd, labels_2nd, \
            inputs_3rd, labels_3rd, data_iterator

def reconstruction_loss(opts, real, reconstr):
    if opts['cost'] == 'l2':
        # c(x,y) = ||x - y||_2
        loss = torch.sum(torch.pow(real - reconstr, 2), axis=1)
        loss = 1.0 * torch.mean(torch.sqrt(1e-08 + loss))
    elif opts['cost'] == 'l2sq':
        # c(x,y) = ||x - y||_2^2
        loss = torch.sum(torch.pow(real - reconstr, 2), axis=1)
        loss = 1.0 * torch.mean(loss)
    elif opts['cost'] == 'l1':
        # c(x,y) = ||x - y||_1
        loss = torch.sum(torch.pow(real - reconstr, 2), axis=1)
        loss = 1.0 * torch.mean(loss)
    elif opts['cost'] == 'norm':
        loss = torch.norm((real-reconstr).abs(), 2, 1).sum() / float(real.size(0))
    else:
        assert False, 'Unknown cost function %s' % opts['cost']
    return loss

def train(model, dataloader, optimizers, device, epoch, num_epoch, filename, entropy, opts, disc_weight=None, entropy_weight=1.0, grl_weight=1.0, model_select='alexnet'):
    dataloader['categorical'].construct()
    train_data = iter(dataloader['categorical'])
    class_criterion = nn.CrossEntropyLoss()
    domain_criterion = nn.CrossEntropyLoss(weight=disc_weight)
    dis_criterion = nn.CrossEntropyLoss()
    mse_criterion = nn.MSELoss()
    if entropy == 'default':
        entropy_criterion = HLoss()
    else:
        entropy_criterion = MaximumSquareLoss()
    p = epoch / num_epoch
    alpha = (2. / (1. + np.exp(-10 * p)) -1) * grl_weight
    beta = (2. / (1. + np.exp(-10 * p)) -1) * entropy_weight
    gamma = (2. / (1. + np.exp(-10 * p)) -1)
    model.discriminator.set_lambd(alpha)
    model.Gdiscriminator.set_lambd(alpha)
    model.train()
    running_loss_class = 0.0
    running_correct_class = 0
    running_loss_domain = 0.0
    running_correct_domain = 0.0
    # Iterate over data.
    for iteration in range(opts['iter']):
        # print(iteration)

        inputs_1st, labels_1st, \
        inputs_2nd, labels_2nd, \
        inputs_3rd, labels_3rd, new_data = CAS(dataloader['categorical'], train_data)
        train_data = new_data

        inputs_1st = torch.cat([to_cuda(samples)
                                       for samples in inputs_1st], dim=0)
        inputs_2nd = torch.cat([to_cuda(samples)
                                        for samples in inputs_2nd], dim=0)
        inputs_3rd = torch.cat([to_cuda(samples)
                                for samples in inputs_3rd], dim=0)

        labels_1st = torch.cat([to_cuda(samples)
                                for samples in labels_1st], dim=0)
        labels_2nd = torch.cat([to_cuda(samples)
                                for samples in labels_2nd], dim=0)
        labels_3rd = torch.cat([to_cuda(samples)
                                for samples in labels_3rd], dim=0)

        domains_1st = to_cuda(torch.tensor([0] * len(labels_1st)))
        domains_2nd = to_cuda(torch.tensor([1] * len(labels_1st)))
        domains_3rd = to_cuda(torch.tensor([2] * len(labels_1st)))

        label_dis_R = to_cuda(torch.tensor([1] * len(labels_1st) * 3))
        label_dis_F = to_cuda(torch.tensor([0] * len(labels_1st) * 6))
        label_dis = torch.cat((label_dis_R, label_dis_F), dim=0)


        inputs = torch.cat((inputs_1st, inputs_2nd, inputs_3rd), dim=0)
        labels = torch.cat((labels_1st, labels_2nd, labels_3rd), dim=0)
        domains = torch.cat((domains_1st, domains_2nd, domains_3rd), dim=0)


        #forward
        output_class, output_domain, x, G_x, G_class, G_domain, concat_dis = model(inputs, training=True)

        total_batch_size = output_class.size(0)
        repetition = int(G_x.size(0) / total_batch_size)
        Greal_labels = labels.repeat(repetition, 1).reshape(G_x.size(0))
        features = x.repeat(repetition, 1)

        loss_class = class_criterion(output_class, labels)
        loss_class_G = 1.0 * class_criterion(G_class, Greal_labels)


        loss_domain = domain_criterion(output_domain, domains)
        loss_dis = dis_criterion(concat_dis, label_dis)

        loss_recon = 1.0 * reconstruction_loss(opts, features, G_x)

        _, pred_class = torch.max(output_class, 1)
        _, pred_domain = torch.max(output_domain, 1)

        total_loss = loss_class + loss_domain


        if model_select == 'resnet_18':
        ##################################### renset-18 parameters ###########################
            for optimizer in optimizers:
                optimizer.zero_grad()
            loss_dis.backward(retain_graph=True)
            temp_grad = []
            count = 0
            for param in model.parameters():
                if count > 61 and count < 72: # count >  59
                    temp_grad.append(param.grad.data.clone())
                else:
                    temp_grad.append([])
                count += 1
            grad_for_Gdis_loss = temp_grad

            for optimizer in optimizers:
                optimizer.zero_grad()
            loss_class_G.backward(retain_graph=True)
            temp_grad = []
            count = 0
            for param in model.parameters():
                if count < 68: # count >  59
                    temp_grad.append(param.grad.data.clone())
                else:
                    temp_grad.append([])
                count += 1
            grad_for_Gclass_loss = temp_grad

            for optimizer in optimizers:
                optimizer.zero_grad()
            loss_recon.backward(retain_graph=True)
            temp_grad = []
            count = 0
            for param in model.parameters():
                if count > 61 and count < 68:
                    temp_grad.append(param.grad.data.clone())
                else:
                    temp_grad.append([])
                count += 1

            grad_for_recon_loss = temp_grad

            for optimizer in optimizers:
                optimizer.zero_grad()
            total_loss.backward()
            temp_grad = []
            count = 0
            for param in model.parameters():
                if count < 62 or count > 71:
                    temp_grad.append(param.grad.data.clone())
                else:
                    temp_grad.append([])
                count += 1
            grad_for_total_loss = temp_grad

            count = 0
            for param in model.parameters():
                temp_grad = param.grad.data.clone()
                temp_grad.zero_()
                if epoch < 0:
                    if count < 62 or count > 71:
                        temp_grad = temp_grad + grad_for_total_loss[count]
                    elif count > 61 and count < 68:
                        temp_grad = temp_grad + 1.0 * grad_for_recon_loss[count] + 1.0 * grad_for_Gclass_loss[count]
                else:
                    if count < 62 or count > 71:
                        if count < 62:
                            temp_grad = temp_grad + grad_for_total_loss[count] + 1.0 * grad_for_Gclass_loss[count]
                        else:
                            temp_grad = temp_grad + grad_for_total_loss[count]
                    else:
                        if count > 61 and count < 68:
                            temp_grad = temp_grad + 1.0 * grad_for_Gdis_loss[count] + 1.0 * grad_for_Gclass_loss[count]
                        else:
                            temp_grad = temp_grad + 1.0 * grad_for_Gdis_loss[count]

                temp_grad = temp_grad
                param.grad.data = temp_grad
                count = count + 1

        elif model_select == 'resnet_50':
        #####################################Resnet-50######################################
            for optimizer in optimizers:
                optimizer.zero_grad()
            loss_dis.backward(retain_graph=True)
            temp_grad = []
            count = 0
            for param in model.parameters():
                if count > 160 and count < 171:
                    temp_grad.append(param.grad.data.clone())
                else:
                    temp_grad.append([])
                count += 1
            grad_for_Gdis_loss = temp_grad

            for optimizer in optimizers:
                optimizer.zero_grad()
            loss_class_G.backward(retain_graph=True)
            temp_grad = []
            count = 0
            for param in model.parameters():
                if count < 167:
                    temp_grad.append(param.grad.data.clone())
                else:
                    temp_grad.append([])
                count += 1
            grad_for_Gclass_loss = temp_grad

            for optimizer in optimizers:
                optimizer.zero_grad()
            loss_recon.backward(retain_graph=True)
            temp_grad = []
            count = 0
            for param in model.parameters():
                if count > 160 and count < 167:
                    temp_grad.append(param.grad.data.clone())
                else:
                    temp_grad.append([])
                count += 1

            grad_for_recon_loss = temp_grad

            for optimizer in optimizers:
                optimizer.zero_grad()
            total_loss.backward()
            temp_grad = []
            count = 0
            for param in model.parameters():
                if count < 161 or count > 170:
                    temp_grad.append(param.grad.data.clone())
                else:
                    temp_grad.append([])
                count += 1
            grad_for_total_loss = temp_grad

            count = 0
            for param in model.parameters():
                temp_grad = param.grad.data.clone()
                temp_grad.zero_()
                if epoch < 0:
                    if count < 161 or count > 170:
                        temp_grad = temp_grad + grad_for_total_loss[count]
                    elif count > 160 and count < 167:
                        temp_grad = temp_grad + 1.0 * grad_for_recon_loss[count] + 1.0 * grad_for_Gclass_loss[count]
                else:
                    if count < 161 or count > 170:
                        if count < 161:  # count == 60 or count == 61:
                            temp_grad = temp_grad + grad_for_total_loss[count] + 1.0 * grad_for_Gclass_loss[count]
                        else:
                            temp_grad = temp_grad + grad_for_total_loss[count]
                    else:
                        if count > 160 and count < 167:
                            temp_grad = temp_grad + 1.0 * grad_for_Gdis_loss[count] + 1.0 * grad_for_Gclass_loss[
                                count]
                        else:
                            temp_grad = temp_grad + 1.0 * grad_for_Gdis_loss[count]
                    temp_grad = temp_grad
                    param.grad.data = temp_grad
                    count = count + 1

        else:
        ####################################### Alexnet ##########################################
            for optimizer in optimizers:
                optimizer.zero_grad()
            loss_dis.backward(retain_graph=True)
            temp_grad = []
            count = 0
            for param in model.parameters():
                if count > 15 and count < 26:
                    temp_grad.append(param.grad.data.clone())
                else:
                    temp_grad.append([])
                count += 1
            grad_for_Gdis_loss = temp_grad

            for optimizer in optimizers:
                optimizer.zero_grad()
            loss_class_G.backward(retain_graph=True)
            temp_grad = []
            count = 0
            for param in model.parameters():
                if count < 22:
                    temp_grad.append(param.grad.data.clone())
                else:
                    temp_grad.append([])
                count += 1
            grad_for_Gclass_loss = temp_grad

            for optimizer in optimizers:
                optimizer.zero_grad()
            loss_recon.backward(retain_graph=True)
            temp_grad = []
            count = 0
            for param in model.parameters():
                if count > 15 and count < 22:
                    temp_grad.append(param.grad.data.clone())
                else:
                    temp_grad.append([])
                count += 1

            grad_for_recon_loss = temp_grad

            for optimizer in optimizers:
                optimizer.zero_grad()
            total_loss.backward()
            temp_grad = []
            count = 0
            for param in model.parameters():
                if count < 16 or count > 25:
                    temp_grad.append(param.grad.data.clone())
                else:
                    temp_grad.append([])
                count += 1
            grad_for_total_loss = temp_grad

            count = 0
            for param in model.parameters():
                temp_grad = param.grad.data.clone()
                temp_grad.zero_()
                if epoch < 0:
                    if count < 16 or count > 25:
                        temp_grad = temp_grad + grad_for_total_loss[count]
                    elif count > 15 and count < 22:
                        temp_grad = temp_grad + 1.0 * grad_for_recon_loss[count] + 1.0 * grad_for_Gclass_loss[count]
                else:
                    if count < 16 or count > 25:
                        if count < 16:
                            temp_grad = temp_grad + grad_for_total_loss[count] + 1.0 * grad_for_Gclass_loss[count]
                        else:
                            temp_grad = temp_grad + grad_for_total_loss[count]
                    else:
                        if count > 15 and count < 22:
                            temp_grad = temp_grad + 1.0 * grad_for_Gdis_loss[count] + 1.0 * grad_for_Gclass_loss[
                                count]
                        else:
                            temp_grad = temp_grad + 1.0 * grad_for_Gdis_loss[count]


                    temp_grad = temp_grad
                    param.grad.data = temp_grad
                    count = count + 1

        for optimizer in optimizers:
            optimizer.step()

        running_loss_class += loss_class.item() * inputs.size(0)
        running_correct_class += torch.sum(pred_class == labels.data)
        running_loss_domain += loss_domain.item() * inputs.size(0)
        running_correct_domain += torch.sum(pred_domain == domains.data)


    epoch_loss_class = running_loss_class / opts['iter']
    epoch_acc_class = running_correct_class.double() / opts['iter']
    epoch_loss_domain = running_loss_domain / opts['iter']
    epoch_acc_domain = running_correct_domain.double() / opts['iter']
    
    log = 'Train: Epoch: {} Alpha: {:.4f} Loss Class: {:.4f} Acc Class: {:.4f}, Loss Domain: {:.4f} Acc Domain: {:.4f}'.format(epoch, alpha, epoch_loss_class, epoch_acc_class, epoch_loss_domain, epoch_acc_domain)
    with open(filename, 'a') as f: 
        f.write(log + '\n') 
    return model, optimizers