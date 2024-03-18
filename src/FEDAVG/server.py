import torch
from collections import OrderedDict
from models import train, test, train_fedprox
from strategies import FedAvg, FedProx


class Client:
    def __init__(
        self, client_name, net, local_train_dataloader, local_val_dataloader, n_epochs
    ):
        self.client_name = client_name
        self.net = net
        self.local_train_dataloader = local_train_dataloader
        self.local_val_dataloader = local_val_dataloader
        self.n_epochs = n_epochs
        self.metrics = {"loss": [], "accuracy": []}

    def get_parameters(self):
        """Return the parameters of the current net."""
        return [val.cpu().numpy() for _, val in self.net.state_dict().items()]

    def set_parameters(self, parameters) -> None:
        """Change the parameters of the model using the given ones."""
        params_dict = zip(self.net.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params_dict})
        self.net.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, strategy, global_net):
        print(f"{self.client_name} fit")
        self.set_parameters(parameters)
        if isinstance(strategy, FedAvg):
            train(
                self.net,
                self.local_train_dataloader,
                self.n_epochs,
                server_name=self.client_name,
            )
        elif isinstance(strategy, FedProx):
            train_fedprox(
                self.net,
                self.local_train_dataloader,
                self.n_epochs,
                server_name=self.client_name,
                mu=strategy.mu,
                global_model=global_net,
            )
        # return self.get_parameters(self.net), len(self.local_train_dataloader)

    def evaluate(self):
        """Use current net to test on local_val_dataloader"""
        loss, accuracy = test(self.net, self.local_val_dataloader)
        self.metrics["loss"].append(loss)
        self.metrics["accuracy"].append(accuracy)


class Central_Server:
    def __init__(self, global_net, test_dl):
        self.global_net = global_net
        self.test_dl = test_dl
        self.metrics = {"loss": [], "accuracy": []}

    def get_parameters(self):
        """Return the parameters of the gloabl net."""
        return [val.cpu().numpy() for _, val in self.global_net.state_dict().items()]

    def test(self):
        loss, accuracy = test(self.global_net, self.test_dl)
        self.metrics["loss"].append(loss)
        self.metrics["accuracy"].append(accuracy)
