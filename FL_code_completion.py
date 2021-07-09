from utils.task_utils import TaskFold, TaskIndexFold
from torch._C import device
from models import GGNN
import argparse
from enum import Enum
import time
import os.path as osp
from os import getpid
import json
from typing import Any, Dict, Optional, Tuple, List, Iterable, Union
import torch
import torch.nn.functional as F
from torch_geometric.data import DataLoader
#from dataProcessing import CGraphDataset, PYGraphDataset, LRGraphDataset
import numpy as np
from utils import cal_metrics, top_5, cal_early_stopping_metric, pretty_print_epoch_task_metrics, average_weights
from multiprocessing import cpu_count
from FL_client import SingleClient, get_client_params, name_to_task_id
from random import shuffle
import copy
from models import name_to_dataset_class, name_to_model_class


class DataFold(Enum):
    TRAIN = 0
    VALIDATION = 1
    TEST = 2


class FL_Completion:
    @classmethod
    def default_args(cls):
        parser = argparse.ArgumentParser(description="Federated_DGAP")
        # 不包含naming的为varmisuse数据集。包含naming的为变量预测的数据集。
        parser.add_argument('--dataset_name_task',
                            type=str,
                            default="matrixrandom",
                            help='the name of the dataset1. optional:[python, learning]')

        parser.add_argument('--backbone_model',
                            type=str,
                            default="ggnn",
                            help='the backbone model of features extract')

        # 目前output model 参数没有用。
        parser.add_argument('--task_load_model_file',
                            type=str,
                            default="trained_models/model_save/ggnn_varmisuse.pt",
                            help='')

        parser.add_argument('--optimizer', type=str, default="Adam", help='')
        parser.add_argument('--lr', type=float, default=0.001, help='')
        parser.add_argument('--lr_deduce_per_epoch', type=int, default=10, help='')
        parser.add_argument('--max_epochs', type=int, default=1500, help='')
        parser.add_argument('--cur_epoch', type=int, default=1, help='')
        parser.add_argument('--max_variable_candidates', type=int, default=5, help='')
        parser.add_argument('--batch_size', type=int, default=64, help='')
        parser.add_argument('--result_dir', type=str, default="trained_models/", help='')
        parser.add_argument('--dropout_rate', type=float, default=0., help='keep_prob = 1-dropout_rate')
        parser.add_argument('--h_features', type=int, default=400, help='')
        parser.add_argument('--rows', type=int, default=200, help='')
        parser.add_argument('--columns', type=int, default=400, help='')
        parser.add_argument('--data_per_column_rate', type=float, default=0.5, help='keep_prob = 1-dropout_rate')
        parser.add_argument('--target_per_data_rate', type=float, default=0.5, help='keep_prob = 1-dropout_rate')
        parser.add_argument('--device', type=str, default="cuda", help='')


        parser.add_argument('--dataset_train_data_dir_task', type=str, default="data/random", help='')
        parser.add_argument('--dataset_validate_data_dir_task', type=str, default="data/random", help='')
        #--Federated setting
        parser.add_argument('--client_fraction',
                            type=float,
                            default=1,
                            help='Fraction of clients to be used for federated updates. Default is 0.1.')
        parser.add_argument('--client_nums', type=int, default=2, help='Number of clients. Default is 100.')
        parser.add_argument('--client_max_epochs', type=int, default=1, help='client epochs. Default is 10.')

        return parser

    @staticmethod
    def name() -> str:
        return "Yang-Federated-Model"

    def __init__(self, args):
        self.args = args
        self.run_id = "_".join([self.name(), time.strftime("%Y-%m-%d-%H-%M-%S"), str(getpid())])
        self._loaded_datasets_task = {}
        self.load_data()
        self.__make_model()
        self.__make_federated_client()

    @property
    def log_file(self):
        return osp.join(self.args.result_dir, "%s.log" % self.run_id)

    def log_line(self, msg):
        with open(self.log_file, 'a') as log_fh:
            log_fh.write(msg + '\n')
        print(msg)

    @property
    def best_model_file(self):
        return osp.join(self.args.result_dir, osp.join("model_save", "%s_best_model.pt" % self.run_id))

    def freeze_model_params(self, filter: Union[str, List[str]], reverse=False):
        for name, parameter in self.model.named_parameters():
            parameter.requires_grad = not reverse
            for f in filter:
                if f.lower() in name.lower():
                    parameter.requires_grad = reverse

        self.log_line("freeze params in %s requires_grad be %s" % (filter, reverse))
        for name, parameter in self.model.named_parameters():
            if parameter.requires_grad == True:
                self.log_line(name)
                #print(parameter.shape)
                #print(parameter)

    def save_model(self, path: str) -> None:
        save_dict = {
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "params": vars(self.args),
        }
        torch.save(save_dict, path)

    def load_model(self, path) -> None:
        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint['model_state_dict'], strict=True)
        self.args.cur_epoch = checkpoint['params']['cur_epoch']
        #self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    def __make_model(self) -> None:
        # 构造模型。


        num_edge_types_task = self.make_task_input_task(None, get_num_edge_types=True)
        
        modelCls, appendix_args = name_to_model_class(self.args.backbone_model, self.args)
        self.log_line("model_args:" + json.dumps(appendix_args))

        self.model = modelCls(
            num_edge_types=num_edge_types_task,
            in_features=self.args.columns,
            out_features=self.args.h_features,
            dropout=self.args.dropout_rate,
            device=self.args.device,
            **appendix_args,
        )

        self.model.to(self.args.device)
       
        self.__make_train_step()

    def __make_train_step(self):
        # use optimizer
        lr = self.args.lr
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)

        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer,
                                                         int(self.args.lr_deduce_per_epoch / self.args.client_fraction),
                                                         gamma=0.6,
                                                         last_epoch=-1)
        self.args.num_model_params = sum(p.numel() for p in list(self.model.parameters()))  # numel()


    def __make_federated_client(self):
        
        client_nums = self.args.client_nums  # 2

        self.client = []
        # 平均分配
        # task1 client
        task_client_nums = client_nums

        dataset_task_data_nums = len(self._loaded_datasets_task[DataFold.TRAIN])
        #self.task_id_1 = name_to_task_id(self.args.task1)
        client_data_params_task1 = get_client_params(task_client_nums, dataset_task_data_nums)

        for cid in range(task_client_nums):
            client = SingleClient(
                client_id=len(self.client),
                data=self._loaded_datasets_task[DataFold.TRAIN][client_data_params_task1[cid]],
                log_file=self.log_file,
                client_max_epochs=self.args.client_max_epochs,
                client_batch_size=self.args.batch_size,
                make_task_input=self.make_task_input_task,
                #dataLoader_numworkers=int(cpu_count() / 2),
                dataLoader_numworkers=0,
                device=self.args.device,
            )
            self.client.append(client)


    def load_data(self) -> None:
       
        train_path_task = self.args.dataset_train_data_dir_task
        validate_path_task = self.args.dataset_validate_data_dir_task

        # if dataset dir not exists
        if not osp.exists(train_path_task) or not osp.exists(validate_path_task):
            raise Exception("train data or validate data dir not exists error")

        datasetCls_task, appendix_args_task, self.make_task_input_task = name_to_dataset_class(
            self.args.dataset_name_task, self.args)

        self._loaded_datasets_task[DataFold.TRAIN] = datasetCls_task(
            train_path_task,
            rows=self.args.rows,
            columns=self.args.columns,
            data_per_column_rate=self.args.data_per_column_rate,
            target_per_data_rate=self.args.target_per_data_rate,
            **appendix_args_task)
        self._loaded_datasets_task[DataFold.VALIDATION] = datasetCls_task(
            validate_path_task,
            rows=self.args.rows,
            columns=self.args.columns,
            data_per_column_rate=self.args.data_per_column_rate,
            target_per_data_rate=self.args.target_per_data_rate,
            **appendix_args_task)


    def criterion(self, y_score, y_true, criterion=torch.nn.MSELoss()):
        loss = criterion(y_score, y_true)
        metrics = cal_metrics(y_score, y_true)
        return loss, metrics

    def __run_epoch(
        self,
        epoch_name: str,
        data: Iterable[Any],
        data_fold: DataFold,
        batch_size: int,
        make_task_input: object,
        quiet: Optional[bool] = False,
    ) -> Tuple[float]:

        batch_iterator = DataLoader(
            data,
            batch_size=batch_size,
            shuffle=True if data_fold == DataFold.TRAIN else False,
            #num_workers=int(cpu_count()/2))
            num_workers=0)

        start_time = time.time()
        processed_graphs, processed_nodes, processed_batch = 0, 0, 0
        epoch_loss = 0.0
        task_metric_results = []

        for batch_data in batch_iterator:
            batch_data = batch_data.to(self.args.device)

            processed_graphs += batch_data.num_graphs
            processed_nodes += batch_data.num_nodes
            processed_batch += 1

            # train in server only have validation
            if data_fold == DataFold.VALIDATION:
                self.model.eval()
                with torch.no_grad():
                    task_batch_data = make_task_input(batch_data)
                    logits = self.model(x=batch_data.x_validate,edge_list=[batch_data.edge_index], slot_id=batch_data.slot_id_validate)
                    loss, metrics = self.criterion(logits, batch_data.label_validate)
                    epoch_loss += loss.item()
                    task_metric_results.append(metrics)

            if not quiet:
                print("Runing %s, batch %i (has %i graphs). Loss so far: %.4f" %
                      (epoch_name, processed_batch, batch_data.num_graphs, epoch_loss / processed_batch),
                      end="\r")

        epoch_time = time.time() - start_time
        per_graph_loss = epoch_loss / processed_batch
        graphs_per_sec = processed_graphs / epoch_time
        nodes_per_sec = processed_nodes / epoch_time

        return per_graph_loss, task_metric_results, processed_graphs, processed_batch, graphs_per_sec, nodes_per_sec, processed_graphs, processed_nodes

    def train(self, quiet=False):
        """训练函数。调用train_epoch训练每一个epoch，获取输出后进行输出。

        Args:
            quiet (bool, optional): [description]. Defaults to False.
        """
        self.log_line(json.dumps(vars(self.args), indent=4))
        total_time_start = time.time()

        clients_fraction = self.args.client_fraction
        clients_nums = self.args.client_nums
        clients_each_epoch = max(int(clients_fraction * clients_nums), 1)

        (best_valid_metric_task, best_val_metric_epoch, best_val_metric_descr) = (float("+inf"), 0, "")

        client_ids = list(range(len(self.client)))
        #self.bn_dict = {}
        for epoch in range(self.args.cur_epoch, self.args.max_epochs + 1):
            self.log_line("== Server Epoch %i" % epoch)

            shuffle(client_ids)
            cur_client_ids = client_ids[:clients_each_epoch]

            # Federated train
            client_weight_list = []
            client_loss_list = []
            for cur_client_id in cur_client_ids:
                cur_client = self.client[cur_client_id]
                client_weight, client_loss, bn_params = cur_client.update_weights(
                    copy.deepcopy(self.model), torch.optim.Adam,
                    self.scheduler.get_last_lr()[0])#, self.bn_dict.get(cur_client.task_id, None))
                
                client_weight_list.append(copy.deepcopy(client_weight))
                client_loss_list.append(copy.deepcopy(client_loss))

            global_weights = average_weights(client_weight_list)
           
            self.model.load_state_dict(global_weights)
            self.log_line("==Server Epoch %i Train: loss: %.5f" %
                          (epoch, sum(client_loss_list) / len(client_loss_list)))

            # Fed Validate
            # --validate task1
            valid_loss, valid_task_metrics, valid_num_graphs, valid_num_batchs, valid_graphs_p_s, valid_nodes_p_s, test_graphs, test_nodes = self.__run_epoch(
                "epoch %i (validation)" % epoch,
                self._loaded_datasets_task[DataFold.VALIDATION],
                DataFold.VALIDATION,
                self.args.batch_size,
                make_task_input=self.make_task_input_task,
                quiet=quiet)

            early_stopping_metric_task = cal_early_stopping_metric(valid_task_metrics)
            valid_metric_descr_task = pretty_print_epoch_task_metrics(valid_task_metrics, valid_num_graphs,valid_num_batchs)
            if not quiet:
                print("\r\x1b[K", end='')
            self.log_line(
                "valid: loss: %.5f || %s || graphs/sec: %.2f | nodes/sec: %.0f | graphs: %.0f | nodes: %.0f | lr: %0.6f"
                % (valid_loss, valid_metric_descr_task, valid_graphs_p_s, valid_nodes_p_s, test_graphs,
                   test_nodes, self.scheduler.get_last_lr()[0]))
            if early_stopping_metric_task < best_valid_metric_task:
                self.args.cur_epoch = epoch + 1  #TODO: model save 检查这个参数
                self.save_model(self.best_model_file)
                self.log_line("  (Best epoch so far, target metric decreased to %.5f from %.5f. Saving to '%s')" %
                              (early_stopping_metric_task, best_valid_metric_task, self.best_model_file))
                best_valid_metric_task = early_stopping_metric_task
                best_val_metric_epoch = epoch
                best_val_metric_descr = valid_metric_descr_task

    def test(self):
        pass
