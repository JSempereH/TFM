from typing import List, Dict
import torch.nn as nn
from abc import ABC
import torch


class Strategy(ABC):
    pass


class FedAvg(Strategy):
    def __init__(self) -> None:
        pass

    def aggregation(
        self,
        global_model: nn.Module,
        list_local_models: List[nn.Module],
        list_len_datasets: List[int],
    ):
        global_params = global_model.state_dict()
        num_clients = len(list_local_models)

        # Initialize the aggregated parameters with zeros
        aggregated_params = {k: torch.zeros_like(v) for k, v in global_params.items()}

        # Aggregate parameters from local models
        for i in range(num_clients):
            local_params = list_local_models[i].state_dict()
            weight = list_len_datasets[i] / sum(
                list_len_datasets
            )  # Compute the weight based on the number of samples
            for key in aggregated_params.keys():
                aggregated_params[key] += weight * local_params[key]

        # Update the global model with the aggregated parameters
        global_model.load_state_dict(aggregated_params)


class FedProx(
    Strategy
):  # TODO: Mejor que herede de una clase base donde se defina la agregación
    def __init__(self, mu) -> None:
        self.mu = mu

    def aggregation(
        self,
        global_model: nn.Module,
        list_local_models: List[nn.Module],
        list_len_datasets: List[nn.Module],
    ):
        global_params = global_model.state_dict()
        num_clients = len(list_local_models)

        # Initialize the aggregated parameters with zeros
        aggregated_params = {k: torch.zeros_like(v) for k, v in global_params.items()}

        # Aggregate parameters from local models
        for i in range(num_clients):
            local_params = list_local_models[i].state_dict()
            weight = list_len_datasets[i] / sum(
                list_len_datasets
            )  # Compute the weight based on the number of samples
            for key in aggregated_params.keys():
                aggregated_params[key] += weight * local_params[key]

        # Update the global model with the aggregated parameters
        global_model.load_state_dict(aggregated_params)

class FedNova(Strategy):
    def __init__(self, mu) -> None:
        self.mu = mu # Compatible with FedProx (Regularization term for loss)

    def aggregation(
        self,
        global_model: nn.Module,
        list_local_models: List[nn.Module],
        list_len_datasets: List[int],
        local_steps: List[int],  # Número de pasos locales de entrenamiento para cada cliente
    ):
        """Assumes Vanilla SGD"""
        global_params = global_model.state_dict()
        num_clients = len(list_local_models)
        total_data_points = sum(list_len_datasets)

        # Initialize the aggregated parameters with zeros
        aggregated_params = {k: torch.zeros_like(v) for k, v in global_params.items()}

        # Compute the normalized aggregation
        for i in range(num_clients):
            local_params = list_local_models[i].state_dict()
            weight = list_len_datasets[i] / (total_data_points*local_steps[i])
            for key in aggregated_params.keys():
                aggregated_params[key] += weight * (local_params[key] - global_params[key]) 

        # Normalize the aggregated updates
        for key in aggregated_params.keys():
            aggregated_params[key] *= sum(x*y for x, y in zip(list_len_datasets, local_steps)) / total_data_points

        # Update the global model with the aggregated parameters
        new_global_params = {key: global_params[key] + aggregated_params[key] for key in global_params.keys()}
        global_model.load_state_dict(new_global_params)


class SCAFFOLD(Strategy):
    def __init__(self, global_lr: float = 1) -> None:
        """By default, global_lr is 1 for SCAFFOLD"""
        self.global_lr = global_lr

    def aggregation(
        self,
        global_model: nn.Module,
        list_local_models: List[nn.Module],
        list_len_datasets: List[int],
        y_delta_list: List[List[torch.Tensor]],
        c_delta_list: List[List[torch.Tensor]],
        c_global: List[torch.Tensor],
    ):
        global_params = global_model.state_dict()
        num_clients = len(list_local_models)
        weights = torch.ones(num_clients) / num_clients

        # Aggregate y_delta and update global model parameters
        for param, y_delta in zip(global_params.values(), zip(*y_delta_list)):
            param.data += self.global_lr * torch.sum(
                torch.stack(y_delta, dim=-1) * weights, dim=-1
            )

        # Update global control variates
        for c_g, c_delta in zip(c_global, zip(*c_delta_list)):
            c_g.data += torch.stack(c_delta, dim=-1).sum(dim=-1) / num_clients

        global_model.load_state_dict(global_params)