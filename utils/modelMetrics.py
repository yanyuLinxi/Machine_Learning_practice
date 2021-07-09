import torch
import numpy as np
from typing import Any, Dict, Optional, Tuple, List, Iterable
import copy


def cal_metrics(y_score, y_true):
    with torch.no_grad():
        return torch.sqrt(torch.mean(y_score-y_true)**2).cpu().item()


def top_5(y_score, y_true):
    #print(y_score)
    #print(y_score.size())
    #print(y_true)
    #print(y_true.size())
    top_5 = [0, 0, 0, 0, 0]

    _, maxk = torch.topk(y_score, 5, dim=-1)
    total = y_true.size(0)
    test_labels = y_true.view(-1, 1)  # reshape labels from [n] to [n,1] to compare [n,k]

    for i in range(5):
        top_5[i] += (test_labels == maxk[:, 0:i + 1]).sum().item()

    return {("top" + str(i + 1)): top_5[i] / total for i in range(5)}


def cal_early_stopping_metric(task_metric_results: List[Dict[str, np.ndarray]], ) -> float:
    # Early stopping based on accuracy; as we are trying to minimize, negate it:
    acc = sum(task_metric_results) / float(len(task_metric_results))
    return -acc


def pretty_print_epoch_task_metrics(task_metric_results: List[Dict[str, np.ndarray]], num_graphs: int,
                                    num_batchs: int) -> str:
    top_str = "RMSE:%f" % (sum(task_metric_results)/ float(len(task_metric_results)))
    return top_str


def average_weights(model_weights: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    w_avg = copy.deepcopy(model_weights[0])
    for key in w_avg.keys():
        for i in range(1, len(model_weights)):
            w_avg[key] += model_weights[i][key]
        w_avg[key] = torch.div(w_avg[key], len(model_weights))
    return w_avg