import torch
from collections import OrderedDict
from models import train, test, train_fedprox
from strategies import FedAvg, FedProx

class Client:
    def __init__(self,
                 client_name,
                 net, 
                 local_train_dataloader, 
                 local_val_dataloader,
                 n_epochs):
        self.client_name = client_name
        self.net = net
        self.local_train_dataloader = local_train_dataloader
        self.local_val_dataloader = local_val_dataloader
        self.n_epochs = n_epochs
    
    def get_parameters(self):
        """Return the parameters of the current net."""
        return [val.cpu().numpy() for _, val in self.net.state_dict().items()]

    def set_parameters(self, parameters) -> None:
        """Change the parameters of the model using the given ones."""
        params_dict = zip(self.net.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params_dict})
        self.net.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, strategy):
        print(f"{self.client_name} fit")
        self.set_parameters(parameters)
        if isinstance(strategy, FedAvg):
            train(self.net, self.local_train_dataloader, self.n_epochs, server_name=self.client_name)
        elif isinstance(strategy, FedProx):
            train_fedprox(self.net, self.local_train_dataloader, self.n_epochs, server_name=self.client_name,
                          mu = strategy.mu, global_model=strategy.global_model)
        #return self.get_parameters(self.net), len(self.local_train_dataloader)

    def evaluate(self, parameters, config):
        print(f"[{self.client_name} evaluate")
        self.set_parameters( parameters)
        loss, accuracy = test(self.net, self.valloader)
        return float(loss), len(self.valloader), {"accuracy": float(accuracy)}

    

