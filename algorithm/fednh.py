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
import torch.nn.functional as F

from src.dataloader import client_dataloader, server_dataloader
from src.optimizer import optimizer_opt
from src.utils import Averager, matrix_score
from count import time_function

# 2023 AAAI
class FedNH():
    def __init__(
            self, train_dataset, test_dataset, train_user_groups, test_user_groups, data_distribution, model, options
    ):
        self.options = options
        self.device = options.gpu

        self.model = model
        self.local_model = [copy.deepcopy(self.model) for _ in range(options.num_clients)]
        self.proto = F.normalize(torch.randn(options.num_classes, model.fdim), dim=1).to(self.device)
        self.local_proto = torch.zeros(options.num_clients, options.num_classes, model.fdim).to(self.device)

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
        # w1 = np.true_divide(self.weight_data_np, self.weight_data_np.sum(axis=0))
        # w2 = np.true_divide(self.weight_data_np.sum(axis=1), self.weight_data_np.sum())
        w1_num = self.weight_data_np
        w2_num = self.weight_data_np.sum(axis=1)
        client_list = [i for i in range(self.options.num_clients)]

        matrix_local, matrix_global = [], []
        for r in range(1, self.options.num_rounds + 1):
            print("Round {}:".format(r))
            matrix_local_list, pml_accuracy_list, pmv_accuracy_list, client_label_list, client_loss_list, client_val_loss_list = [], [], [], [], [], []

            # select clients
            model_dict = self.model.state_dict()
            client_select = random.sample(client_list, int(self.options.num_clients * self.options.radio))
            w1_sum_num = np.zeros(self.options.num_classes)
            w2_sum_num = 0
            for c in client_select:
                w1_sum_num += w1_num[c]
                w2_sum_num += w2_num[c]   
            w1_sum_num[w1_sum_num == 0.0] = 1e-12
            w1 = np.true_divide(w1_num, w1_sum_num)
            w2 = np.true_divide(w2_num, w2_sum_num)

            # Update
            for client in client_select:
                self.local_model[client].load_state_dict(model_dict)
                client_loss = self.client_update(model=self.local_model[client], train_loader=self.train_loaders[client], weight_data=self.weight_data_np[client], client=client)
                client_loss_list.append(client_loss)

            # Test
            if r >= self.options.num_rounds - self.options.test_rounds:
                for client in client_select:
                    matrix, pml_acc, pmv_acc, label_accuracy, test_loss = self.test(model=self.local_model[client], loader=self.test_loaders[client], weight_data=self.weight_data_np[client])
                    matrix_local_list.append(matrix)
                    pml_accuracy_list.append(pml_acc)
                    pmv_accuracy_list.append(pmv_acc)
                    client_label_list.append(label_accuracy)
                    client_val_loss_list.append(test_loss)
                matrix_local.append(matrix_local_list)

                self.server_update1(select=client_select, w=w1)
                self.server_update2(client_models=self.local_model, global_model=self.model, select=client_select, w=w2)

                matrix, server_accuracy, _, server_label, val_loss = self.test(model=self.model, loader=self.server_test_loader, weight_data=self.weight_data_np.sum(axis=0))
                matrix_global.append(matrix)

                print(" Server Acc: {:.4f}    PML Acc: {:.4f}    PMV Acc: {:.4f}".format(server_accuracy, mean(pml_accuracy_list),  mean(pmv_accuracy_list)))

            else:
                self.server_update1(select=client_select, w=w1)
                self.server_update2(client_models=self.local_model, global_model=self.model, select=client_select, w=w2)
            
    
    def client_update(self, model, train_loader, weight_data, client):
        model.train()
        optimizer = optimizer_opt(model, self.options)
        criterion = nn.CrossEntropyLoss()
        model = model.to(self.device)
        train_loss = 0.0
        for epoch in range(self.options.num_epochs):
            average_loss = Averager()
            self.local_proto[client] = self.local_proto[client] * 0
            for i, (x, y) in enumerate(train_loader):
                x = x.to(self.device)
                y = y.to(self.device)
                optimizer.zero_grad()
                features, logits = model(x, self.proto)
                for index, label in enumerate(y):
                    f = torch.div(features[index].detach(), weight_data[label])
                    self.local_proto[client][label] = self.local_proto[client][label] + f
                loss = criterion(self.options.fednh_s * logits, y)
                loss.backward()
                optimizer.step()
                average_loss.add(loss.item())

            train_loss = average_loss.item()

        return train_loss

    # Prototype Mountain
    def server_update1(self, select, w):
        self.proto = self.proto * 0.5
        for client in select:
            w_matrix = torch.tensor(w[client].reshape(-1, 1).repeat(self.model.fdim, axis=1))
            self.proto = self.proto + self.local_proto[client] * w_matrix.to(self.device) * 0.5
        self.proto = F.normalize(self.proto, dim=1).to(torch.float32)

    # Model AVG
    def server_update2(self, client_models, global_model, select, w):
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
                features, logits = model(x, self.proto)
                loss = criterion(logits, y)
                predicted = logits.argmax(dim=1)
                predict_y.extend(predicted.cpu())
                img_y.extend(y.cpu())
                test_loss.add(loss.item())

        matrix, pml_accuracy, pmv_accuracy, label_accuracy = matrix_score(predict_y, img_y,self.options.num_classes, weight_data)

        return matrix, pml_accuracy, pmv_accuracy, label_accuracy, test_loss.item()

    def save_matrix(self, matrix_local, matrix_global, select):
        metrix_artifact = wandb.Artifact('Metrix', type='Metrix')
        print("Local Matrix:")
        for i in select:
            table = wandb.Table(columns=[i for i in range(self.options.num_classes)], data=matrix_local[-1][i])
            metrix_artifact.add(table, "Metrix Local {}".format(i))
            print(matrix_local[-1][i])
        print("Global Matrix:")
        table = wandb.Table(columns=[i for i in range(self.options.num_classes)], data=matrix_global[-1])
        metrix_artifact.add(table, "Metrix Global")
        print(matrix_global[-1])
        self.wandb.log_artifact(metrix_artifact)



