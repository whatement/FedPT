import copy
import random
import sys

import numpy as np

import torch
import torch.nn as nn
import torchvision
from numpy import mean
from torch.utils.data import DataLoader
import wandb
from tqdm import tqdm

from src.dataloader import client_dataloader, server_dataloader
from src.optimizer import optimizer_opt
from src.utils import Averager, matrix_score
from count import time_function

# Finished
class FedAvg():
    def __init__(
            self, train_dataset, test_dataset, train_user_groups, test_user_groups, data_distribution, model, options
    ):
        self.options = options
        self.device = options.gpu

        self.model = model.to(self.device)
        self.local_model = [copy.deepcopy(self.model) for _ in range(options.num_clients)]

        self.weight_data_np = np.array(data_distribution)

        # client dataloaders
        self.train_loaders = client_dataloader(train_dataset, train_user_groups, options.num_clients, options.batch_size)
        self.test_loaders = client_dataloader(test_dataset, test_user_groups, options.num_clients, options.batch_size)
        # server dataloader
        self.server_test_loader = server_dataloader(test_dataset, options.batch_size)

        if options.test_all:
            for i in range(options.num_clients):
                self.test_loaders[i] = self.server_test_loader

    def trainer(self):
        w_num = self.weight_data_np.sum(axis=1)
        client_list = [i for i in range(self.options.num_clients)]

        matrix_local, matrix_global = [], []
        for r in range(1, self.options.num_rounds + 1):
            print("Round {}:".format(r))
            matrix_local_list, pml_accuracy_list, pmv_accuracy_list, client_label_list, client_loss_list, client_val_loss_list = [], [], [], [], [], []

            model_dict = self.model.state_dict()
            client_select = random.sample(client_list, int(self.options.num_clients * self.options.radio))
            sum_num = 0
            for c in client_select:
                sum_num += w_num[c]
            w = np.true_divide(w_num, sum_num)

            # Update
            for client in client_select:
                self.local_model[client].load_state_dict(model_dict)
                client_loss = self.client_update(model=self.local_model[client], train_loader=self.train_loaders[client])
                client_loss_list.append(client_loss)

            self.server_update(client_models=self.local_model, global_model=self.model, select=client_select, w=w)

            if r >= self.options.num_rounds - self.options.test_rounds:
                # Test
                for client in client_select:
                    matrix, pml_acc, pmv_acc, label_accuracy, test_loss = self.test(model=self.local_model[client], loader=self.test_loaders[client], weight_data=self.weight_data_np[client])
                    matrix_local_list.append(matrix)
                    pml_accuracy_list.append(pml_acc)
                    pmv_accuracy_list.append(pmv_acc)
                    client_label_list.append(label_accuracy)
                    client_val_loss_list.append(test_loss)
                matrix_local.append(matrix_local_list)
                matrix, server_accuracy, b, server_label, val_loss = self.test(model=self.model, loader=self.server_test_loader, weight_data=self.weight_data_np.sum(axis=0))
                matrix_global.append(matrix)

                print(" Server Acc: {:.4f}    PML Acc: {:.4f}    PMV Acc: {:.4f}".format(server_accuracy, mean(pml_accuracy_list),  mean(pmv_accuracy_list)))

                
    def client_update(self, model, train_loader):
        model.train()
        optimizer = optimizer_opt(model, self.options)
        criterion = nn.CrossEntropyLoss()
        model = model.to(self.device)
        train_loss = 0.0
        for epoch in range(self.options.num_epochs):
            average_loss = Averager()
            for i, (x, y) in enumerate(train_loader):
                x = x.to(self.device)
                y = y.to(self.device)
                optimizer.zero_grad()
                features, logits = model(x)
                loss = criterion(logits, y)
                loss.backward()
                optimizer.step()
                average_loss.add(loss.item())
            train_loss = average_loss.item()
        return train_loss

    def server_update(self, client_models, global_model, select, w):
        mean_weight_dict = {}
        for name, param in self.model.state_dict().items():
            weight = []
            for index in select:
                weight.append(client_models[index].state_dict()[name] * w[index])
            weight = torch.stack(weight, dim=0)
            mean_weight_dict[name] = weight.sum(dim=0)
        global_model.load_state_dict(mean_weight_dict, strict=False)

    def test(self, model, loader, weight_data):
        model.eval()
        criterion = nn.CrossEntropyLoss()
        predict_y = []
        img_y = []
        model.to(self.device)
        test_loss = Averager()
        with torch.no_grad():
            for i, (x, y) in enumerate(loader):
                x = x.to(self.device)
                y = y.to(self.device)
                features, logits = model(x)
                loss = criterion(logits, y)
                predicted = logits.argmax(dim=1)
                predict_y.extend(predicted.cpu())
                img_y.extend(y.cpu())
                test_loss.add(loss.item())

        matrix, pml_accuracy, pmv_accuracy, label_accuracy = matrix_score(predict_y, img_y, self.options.num_classes, weight_data)

        return matrix, pml_accuracy, pmv_accuracy, label_accuracy, test_loss.item()

