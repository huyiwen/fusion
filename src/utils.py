import torch
import os
from src.dataset import Multimodal_Datasets

def get_data(args, dataset, split='train'):
    data_path = os.path.join(args.data_path, dataset) + f'_{split}.dt'
    if not os.path.exists(data_path):
        print(f"  - Creating new {split} data")
        data = Multimodal_Datasets(args.data_path, dataset, split, args.aligned)
        torch.save(data, data_path)
    else:
        print(f"  - Found cached {split} data")
        data = torch.load(data_path)
    return data


def save_model(args, model, name=''):
    torch.save(model, f'results/{name}.pt')


def load_model(args, name=''):
    # name = save_load_name(args, name)
    model = torch.load(f'results/{name}.pt')
    return model
