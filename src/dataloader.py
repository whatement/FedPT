import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


class DatasetSplit(Dataset):
    """
    An abstract Dataset class wrapped around Pytorch Dataset class.
    """
    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = [int(i) for i in idxs]

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]
        if isinstance(image, torch.Tensor):
            image = image.clone().detach()
        else:
            image = image
        if isinstance(label, torch.Tensor):
            label = label.clone().detach()
        else:
            label = torch.tensor(label)
        return image, label


# 返回 Client Dataloader dict, 以及每个节点的数据总量 (List)
def client_dataloader(dataset, user_groups, num_clients, batch_size):
    client_loaders_dict = {}
    for i in range(num_clients):
        split = DatasetSplit(dataset, user_groups[i])
        if len(split) > 0:
            client_loaders_dict[i] = DataLoader(split,
                                                batch_size=batch_size, shuffle=True,
                                                drop_last=True)
        else:
            client_loaders_dict[i] = 0
    return client_loaders_dict


# 返回测试数据集的 Dataloader
def server_dataloader(dataset, batch_size):
    server_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    return server_loader