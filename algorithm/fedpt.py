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
from src.utils import Averager, generate_equiangular_tight, matrix_score
from count import time_function


class FedPT():
    def __init__(
            self, train_dataset, test_dataset, train_user_groups, test_user_groups, data_distribution, model, options
    ):
        self.options = options
        self.device = options.gpu

        self.model = model
        self.local_model = [copy.deepcopy(self.model) for _ in range(options.num_clients)]
        # self.proto = F.normalize(torch.randn(options.num_classes, model.fdim), dim=1).to(self.device)
        self.proto = F.normalize(generate_equiangular_tight(options.num_classes, model.fdim), dim=1).to(self.device)
        self.local_proto = torch.zeros(options.num_clients, options.num_classes, model.fdim).to(self.device)

        self.weight_data_np = np.array(data_distribution)
        self.temperature = options.fedpt_temp

        self.lamb = 1.0
        self.gam = 0.01

        # client dataloaders
        self.train_loaders = client_dataloader(train_dataset, train_user_groups, options.num_clients, options.batch_size)
        self.test_loaders = client_dataloader(test_dataset, test_user_groups, options.num_clients, options.batch_size)
        # server dataloader
        self.server_test_loader = server_dataloader(test_dataset, options.batch_size)

        if options.test_all:
            for i in range(options.num_clients):
                self.test_loaders[i] = self.server_test_loader


    def trainer(self):
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

            self.server_update1(select=client_select, w=w1)
            self.server_update2(client_models=self.local_model, global_model=self.model, select=client_select, w=w2)
            
            if r >= self.options.num_rounds - self.options.test_rounds:
                # Test
                for client in client_select:
                    matrix, pml_acc, pmv_acc, label_accuracy, test_loss = \
                        self.test(model=self.local_model[client], loader=self.test_loaders[client], weight_data=self.weight_data_np[client], local_proto=self.local_proto[client])
                    matrix_local_list.append(matrix)
                    pml_accuracy_list.append(pml_acc)
                    pmv_accuracy_list.append(pmv_acc)
                    client_label_list.append(label_accuracy)
                    client_val_loss_list.append(test_loss)
                matrix_local.append(matrix_local_list)
                matrix, server_accuracy, _, server_label, val_loss = self.test(model=self.model, loader=self.server_test_loader, weight_data=self.weight_data_np.sum(axis=0))
                matrix_global.append(matrix)

                print(" Server Acc: {:.4f}    PML Acc: {:.4f}    PMV Acc: {:.4f}".format(server_accuracy, mean(pml_accuracy_list),  mean(pmv_accuracy_list)))

    def client_update(self, model, train_loader, weight_data, client):
        model.train()
        optimizer = optimizer_opt(model, self.options)
        criterion_bscl = BalSupConLoss(self.options.num_classes, self.device)
        criterion_pcl = nn.CrossEntropyLoss()
        model = model.to(self.device)
        train_loss = 0.0
        proto = self.proto.clone().detach()
        for epoch in range(self.options.num_epochs):
            average_loss = Averager()
            proto = proto.to(self.device)
            self.local_proto[client] = proto.clone().detach() * self.options.fedpt_rate
            for i, (x, y) in enumerate(train_loader):
                x = x.to(self.device)
                y = y.to(self.device)
                optimizer.zero_grad()
                features, logits = model(x, proto)
                for index, label in enumerate(y):
                    f = torch.div(features[index].detach(), weight_data[label])
                    self.local_proto[client][label] = self.local_proto[client][label] + f * (1 - self.options.fedpt_rate)

                loss = self.lamb * criterion_pcl(logits / self.temperature, y) + self.gam * criterion_bscl(proto, features, y, self.temperature)

                loss.backward()
                optimizer.step()
                average_loss.add(loss.item())

            for label in range(self.options.num_classes):
                self.local_proto[client] = F.normalize(self.local_proto[client], dim=1)
                if weight_data[label] > 0:
                    proto[label] = self.local_proto[client][label].clone().detach()
   
            train_loss = average_loss.item()

        return train_loss
    
    def server_update1(self, select, w):
        self.proto = self.proto * 0
        for client in select:
            w_matrix = torch.tensor(w[client].reshape(-1, 1).repeat(self.model.fdim, axis=1))
            self.proto = self.proto + self.local_proto[client] * w_matrix.to(self.device)
        self.proto = F.normalize(self.proto, dim=1).to(torch.float32)

    def server_update2(self, client_models, global_model, select, w):
        mean_weight_dict = {}
        for name, param in self.model.state_dict().items():
            weight = []
            for index in select:
                weight.append(client_models[index].state_dict()[name] * w[index])
            weight = torch.stack(weight, dim=0)
            mean_weight_dict[name] = weight.sum(dim=0)
        global_model.load_state_dict(mean_weight_dict, strict=False)

    def test(self, model, loader, weight_data, local_proto=None):
        model.eval()
        criterion = nn.CrossEntropyLoss()
        predict_y = []
        img_y = []
        model.to(self.device)
        test_loss = Averager()
        if local_proto is None:
            proto = self.proto.clone().detach()
        else:
            proto = local_proto
        with torch.no_grad():
            for i, (x, y) in enumerate(loader):
                x = x.to(self.device)
                y = y.to(self.device)
                features, logits = model(x, proto)
                loss = criterion(logits, y)
                predicted = logits.argmax(dim=1)
                predict_y.extend(predicted.cpu())
                img_y.extend(y.cpu())
                test_loss.add(loss.item())

        matrix, pml_accuracy, pmv_accuracy, label_accuracy = matrix_score(predict_y, img_y,self.options.num_classes, weight_data)

        return matrix, pml_accuracy, pmv_accuracy, label_accuracy, test_loss.item()



class BalSCL(nn.Module):
    def __init__(self, cls_num=None):
        super(BalSCL, self).__init__()
        self.cls_num = cls_num

    def forward(self, proto, features, targets, temperature, device):

        batch_size = features.shape[0]
        targets = targets.contiguous().view(-1, 1)
        targets_centers = torch.arange(self.cls_num, device=device).view(-1, 1)
        targets = torch.cat([targets, targets_centers], dim=0)
        batch_cls_count = torch.eye(self.cls_num)[targets].sum(dim=0).squeeze()

        mask = torch.eq(targets[:batch_size], targets.T).float().to(device)
        logits_mask = torch.scatter(torch.ones_like(mask), 1, torch.arange(batch_size).view(-1, 1).to(device), 0)
        mask = mask * logits_mask

        features = torch.cat([features, proto.to(device)], dim=0)
        logits = features[:batch_size].mm(features.T)
        logits = torch.div(logits, temperature)
        logits_max, _ = torch.max(logits, dim=1, keepdim=True)
        logits = logits - logits_max.detach()
        exp_logits = torch.exp(logits) * logits_mask
        per_ins_weight = torch.tensor([batch_cls_count[i] for i in targets], device=device).view(1, -1).expand(
            batch_size, batch_size + self.cls_num) - mask
        exp_logits_sum = exp_logits.div(per_ins_weight).sum(dim=1, keepdim=True)
        log_prob = logits - torch.log(exp_logits_sum)
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

        loss = - mean_log_prob_pos.mean()
        return loss
