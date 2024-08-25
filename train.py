# Standard libraries
import argparse
from collections import defaultdict
import os

# Third-party libraries
import numpy as np
import pandas as pd
import torch
from monai.data import DataLoader, CacheDataset
import tqdm
import pickle
from sklearn.model_selection import RepeatedStratifiedKFold

# Third-party libraries
from src.logger import logger
from src.utils import make_data_dict, froze_encoder, clean_state_dict, set_seed
from src.model import ResNet10Wrapper
from src.optimization import optimization_loop

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='data.csv')
    parser.add_argument('--n_gpus', type=int, default=1)
    parser.add_argument('--n_cpus', type=int, default=30)
    parser.add_argument('--froze', type=bool, default=False)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--tumor', type=bool, default=True)
    parser.add_argument('--entity', type=str, default=None)
    parser.add_argument('--path_to_save', type=str, default='./logs')
    parser.add_argument('--project', type=str, default='gliomol')
    parser.add_argument('--seed', type=int, default=2024)
    args = parser.parse_args()


    if args.tumor:
        logger.info('Using tumor centered data')
        from src.augmentations import transforms_tumor_train as transforms_train
        from src.augmentations import transforms_tumor_val as transforms_val
        pretrained_path = 'pretrained_encoders/tumor_encoder.pth'
    else:
        logger.info('Using whole brain data')
        from src.augmentations import train_transforms_regular as transforms_train
        from src.augmentations import val_transforms_regular as transforms_val
        pretrained_path = 'pretrained_encoders/brain_encoder.pth'
    
    logger.info('Fixing seed!')
    set_seed(args.seed)

    if not os.path.exists(args.path_to_save):
        os.makedirs(args.path_to_save)
    
    meta_metrics = defaultdict(list)
    logger.info('Loading data')
    dataframe = pd.read_csv(args.data_path)
    kfold = RepeatedStratifiedKFold(n_splits=3, n_repeats=100, random_state=args.seed)
    for i, (train_index, test_index) in tqdm.tqdm(enumerate(kfold.split(dataframe, dataframe.label)), total=(kfold.get_n_splits(dataframe, dataframe.label))):
        name = f'ResNet_tumor_{args.tumor}'
        data_train, data_test = dataframe.iloc[train_index], dataframe.iloc[test_index]
        data_dict_train = make_data_dict(data_train, tumor=args.tumor)
        data_dict_test = make_data_dict(data_test, tumor=args.tumor)
        train_ds = CacheDataset(data=data_dict_train, transform=transforms_train, cache_rate=1., runtime_cache="processes", copy_cache=False)
        val_ds = CacheDataset(data=data_dict_test, transform=transforms_val, cache_rate=1., runtime_cache="processes", copy_cache=False)

        train_loader = DataLoader(train_ds, batch_size=6, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=1, shuffle=False)

        model = ResNet10Wrapper()
        if pretrained_path is not None:
            model.load_state_dict(clean_state_dict(torch.load(pretrained_path)), strict=False)
        if args.froze:
            model = froze_encoder(model)
        if args.n_gpus > 1:
            model = torch.nn.DataParallel(model, device_ids=list(range(args.n_gpus)))
        
        logger.info(f"number of trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")
        device = torch.device(f'cuda' if torch.cuda.is_available() else 'cpu')
        model = model.to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)
        criterion = torch.nn.BCEWithLogitsLoss()
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.1, patience=15)
        config = vars(args)
        config.update({'pretrained_path': pretrained_path})
        model, metrics = optimization_loop(model=model, train_dl=train_loader, val_dl=val_loader, optimizer=optimizer, criterion=criterion, scheduler=scheduler, epochs=args.epochs, device=device, entity=args.entity, project=args.project, name=name, config=config, is_lopo=False)
        for key, value in metrics.items():
            meta_metrics[key].append(value[-1])
    
    with open(os.path.join(args.path_to_save, f'meta_metrics_tumor-{str(args.tumor)}.pkl'), 'wb') as fp:
        pickle.dump(meta_metrics, fp)


