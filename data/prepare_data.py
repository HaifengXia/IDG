import os
import data.utils as data_utils
from data.custom_dataset_dataloader import CustomDatasetDataLoader
from data.class_aware_dataset_dataloader import ClassAwareDataLoader

def prepare_data(opts):
    dataloaders = {}
    train_transform = data_utils.get_transform(train=True)
    test_transform = data_utils.get_transform(False)

    source_1 = opts['source_name_1'] #cfg.DATASET.SOURCE_NAME
    source_2 = opts['source_name_2']
    source_3 = opts['source_name_3']
    target = opts['target_name']
    #target = cfg.DATASET.TARGET_NAME
    dataroot_S1 = os.path.join(opts['data_root'], source_1)
    dataroot_S2 = os.path.join(opts['data_root'], source_2)
    dataroot_S3 = os.path.join(opts['data_root'], source_3)
    dataroot_T = os.path.join(opts['data_root'], target)

    with open(os.path.join(opts['data_root'], 'category.txt'), 'r') as f:
        classes = f.readlines()
        classes = [c.strip() for c in classes]
    assert(len(classes) == opts['num_classes'])
    set_class = [classes[c] for c in range(opts['num_classes'])]

    
    # initialize the categorical dataloader
    dataset_type = 'CategoricalSTDataset'
    source_batch_size = opts['source_batch_size']
    # target_batch_size = cfg.TRAIN.TARGET_CLASS_BATCH_SIZE
    print('Building categorical dataloader...')
    dataloaders['categorical'] = ClassAwareDataLoader(
                dataset_type=dataset_type, 
                source_batch_size=source_batch_size, 
                source_dataset_root_1=dataroot_S1,
                source_dataset_root_2=dataroot_S2,
                source_dataset_root_3=dataroot_S3,
                transform=train_transform, 
                classnames=classes, class_set=set_class, num_selected_classes=opts['num_selected_classes'],
                drop_last=True, sampler='RandomSampler')

    batch_size = opts['test_batch_size']
    dataset_type = 'SingleDataset'
    test_domain = target
    dataroot_test = os.path.join(opts['data_root'], test_domain)
    dataloaders['test'] = CustomDatasetDataLoader(
                    dataset_root=dataroot_test, dataset_type=dataset_type,
                    batch_size=batch_size, transform=test_transform,
                    train=False, classnames=classes)

    return dataloaders