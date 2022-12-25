from torch import nn
import torch
from utils.utils import to_cuda, accuracy
import numpy as np
import pandas as pd

def eval_model(model, eval_data, epoch, filename):
    softmax = nn.Softmax(dim=1)
    model.eval()
    preds = []
    gts = []
    feats = []
    for sample in iter(eval_data):
        with torch.no_grad():
            data, gt = to_cuda(sample['Img']), to_cuda(sample['Label'])
            feat, logits = model(data, training=False)
            logits = softmax(logits)
            preds += [logits]
            gts += [gt]
            feats += [feat]

    preds = torch.cat(preds, dim=0)
    gts = torch.cat(gts, dim=0)
    feats = torch.cat(feats, dim=0)

    res = accuracy(preds, gts)
    log = 'Eval: Epoch: {} Acc: {:.4f}'.format(epoch, res)
    print(log)
    with open(filename, 'a') as f:
        if epoch == 0:
            f.write('**********************************************'+'\n')
        f.write(log + '\n')

    return res