import os
from .image_folder import make_dataset_with_labels, make_dataset_classwise
from PIL import Image
from torch.utils.data import Dataset
import random
from math import ceil
import torch

class CategoricalSTDataset(Dataset):
    def __init__(self):
        super(CategoricalSTDataset, self).__init__()

    def initialize(self, source_root_1, source_root_2, source_root_3,
                  classnames, class_set, 
                  source_batch_size, seed=None, 
                  transform=None, **kwargs):

        self.source_root_1 = source_root_1
        self.source_root_2 = source_root_2
        self.source_root_3 = source_root_3
        # self.target_paths = target_paths

        self.transform = transform
        self.class_set = class_set
        
        self.data_paths = {}
        self.data_paths['source_1'] = {}
        cid = 0
        for c in self.class_set:
            self.data_paths['source_1'][cid] = make_dataset_classwise(self.source_root_1, c)
            cid += 1

        self.data_paths['source_2'] = {}
        cid = 0
        for c in self.class_set:
            self.data_paths['source_2'][cid] = make_dataset_classwise(self.source_root_2, c)
            cid += 1

        self.data_paths['source_3'] = {}
        cid = 0
        for c in self.class_set:
            self.data_paths['source_3'][cid] = make_dataset_classwise(self.source_root_3, c)
            cid += 1

        self.seed = seed
        self.classnames = classnames

        self.batch_sizes = {}
        for d in ['source_1', 'source_2', 'source_3']:
            self.batch_sizes[d] = {}
            cid = 0
            for c in self.class_set:
                batch_size = source_batch_size
                self.batch_sizes[d][cid] = min(batch_size, len(self.data_paths[d][cid]))
                cid += 1


    def __getitem__(self, index):
        data = {}
        for d in ['source_1', 'source_2', 'source_3']:
            cur_paths = self.data_paths[d]
            if self.seed is not None:
                random.seed(self.seed)

            inds = random.sample(range(len(cur_paths[index])), \
                                 self.batch_sizes[d][index])

            path = [cur_paths[index][ind] for ind in inds]
            data['Path_'+d] = path
            assert(len(path) > 0)
            for p in path:
                img = Image.open(p).convert('RGB')
                if self.transform is not None:
                    img = self.transform(img)

                if 'Img_'+d not in data:
                    data['Img_'+d] = [img]
                else:
                    data['Img_'+d] += [img]

            data['Label_'+d] = [self.classnames.index(self.class_set[index])] * len(data['Img_'+d])
            # if d == 'source_1':
            #     data['Domain_'+d] = [0] * len(data['Img_'+d])
            # elif d == 'source_2':
            #     data['Domain_'+d] = [1] * len(data['Img_'+d])
            # elif d == 'source_3':
            #     data['Domain_'+d] = [2] * len(data['Img_'+d])

            data['Img_'+d] = torch.stack(data['Img_'+d], dim=0)




        # print('--------------------------------------------')
        # print(data['Domain_source_1'])
        return data

    def __len__(self):
        return len(self.class_set)

    def name(self):
        return 'CategoricalSTDataset'

