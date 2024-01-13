from typing import List
import torch.nn as nn
import torch


class FedAvg:
    def __init__(self) -> None:
        pass
    
    def aggregation(self,
                    global_model: nn.Module,
                    list_local_models: List[nn.Module],
                    list_len_datasets: List[nn.Module]):
        global_params = global_model.state_dict()
        num_clients = len(list_local_models)
        
        # Initialize the aggregated parameters with zeros
        aggregated_params = {k: torch.zeros_like(v) for k, v in global_params.items()}

        # Aggregate parameters from local models
        for i in range(num_clients):
            local_params = list_local_models[i].state_dict()
            weight = list_len_datasets[i] / sum(list_len_datasets)  # Compute the weight based on the number of samples
            for key in aggregated_params.keys():
                aggregated_params[key] += weight * local_params[key]

        # Update the global model with the aggregated parameters
        global_model.load_state_dict(aggregated_params)


class FedProx(): # Mejor que herede de una clase base donde se defina la agregación
    def __init__(self, global_model, mu) -> None:
        self.global_model = global_model
        self.mu = mu

    def aggregation(self,
                    global_model: nn.Module,
                    list_local_models: List[nn.Module],
                    list_len_datasets: List[nn.Module]):
        global_params = global_model.state_dict()
        num_clients = len(list_local_models)
        
        # Initialize the aggregated parameters with zeros
        aggregated_params = {k: torch.zeros_like(v) for k, v in global_params.items()}

        # Aggregate parameters from local models
        for i in range(num_clients):
            local_params = list_local_models[i].state_dict()
            weight = list_len_datasets[i] / sum(list_len_datasets)  # Compute the weight based on the number of samples
            for key in aggregated_params.keys():
                aggregated_params[key] += weight * local_params[key]

        # Update the global model with the aggregated parameters
        global_model.load_state_dict(aggregated_params)