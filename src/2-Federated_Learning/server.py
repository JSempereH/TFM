import torch
from collections import OrderedDict
from models import train, test, train_fedprox, train_scaffold
from strategies import FedAvg, FedProx, FedNova, SCAFFOLD
from tqdm import tqdm

class Client:
    def __init__(
        self, client_name, net, local_train_dataloader, local_val_dataloader, n_epochs, device=torch.device("cpu")
    ):
        self.client_name = client_name
        self.net = net
        self.local_train_dataloader = local_train_dataloader
        self.local_val_dataloader = local_val_dataloader
        self.n_epochs = n_epochs
        self.device = device

        # SCAFFOLD specific attributes
        self.iter_trainloader = iter(self.local_train_dataloader)
        self.c_local = [torch.zeros_like(param) for param in self.net.parameters()]
        self.c_global = [torch.zeros_like(param) for param in self.net.parameters()]
        self.y_delta = []
        self.c_delta = []

        self.metrics = {
            "train_loss": [],
            "train_accuracy": [],
            "val_loss": [],
            "val_accuracy": [],
            "test_loss": [],
            "test_accuracy": [],
        }

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
        elif isinstance(strategy, FedNova): #Use also FedProx when possible
            train_fedprox(
                self.net,
                self.local_train_dataloader,
                self.n_epochs,
                server_name=self.client_name,
                mu=strategy.mu,
                global_model=global_net,
            )
        elif isinstance(strategy, SCAFFOLD):
            self.y_delta, self.c_delta, self.c_local = train_scaffold(
                self.net,
                self.local_train_dataloader,
                self.n_epochs,
                self.client_name,
                global_net,
                self.c_global,
                self.c_local
            )
        # return self.get_parameters(self.net), len(self.local_train_dataloader)

    

    def evaluate(self):
        """Use current net to test on local_val_dataloader."""
        loss, accuracy = test(self.net, self.local_val_dataloader, self.client_name)
        self.metrics["test_loss"].append(loss)
        self.metrics["test_accuracy"].append(accuracy)
    


class Central_Server:
    def __init__(self, global_net, test_dl):
        self.global_net = global_net
        self.test_dl = test_dl

        # SCAFFOLD specific attributes
        self.c_global = [torch.zeros_like(param) for param in self.global_net.parameters()]
        self.metrics = {"global_loss": [], "global_accuracy": []}

    def aggregate(self, list_clients, strategy, list_of_epochs):
        if isinstance(strategy, SCAFFOLD):
            clients_package = {i: client for i, client in enumerate(list_clients)}
            c_delta_list = [client.c_delta for client in list_clients]
            y_delta_list = [client.y_delta for client in list_clients]
            strategy.aggregation(
                self.global_net,
                [client.net for client in list_clients],
                [client.local_train_dataloader.dataset.__len__() for client in list_clients],
                y_delta_list,
                c_delta_list,
                self.c_global
            )
        elif isinstance(strategy, FedNova):
            strategy.aggregation(
                global_model=self.global_net,
                list_local_models=[client.net for client in list_clients],
                list_len_datasets=[
                    client.local_train_dataloader.dataset.__len__()
                    for client in list_clients
                ],
                local_steps=list_of_epochs
            )
        else:
            # FedAvg, FedProx, FedNova
            strategy.aggregation(
                self.global_net,
                [client.net for client in list_clients],
                [client.local_train_dataloader.dataset.__len__() for client in list_clients],
            )

    def get_parameters(self):
        """Return the parameters of the gloabl net."""
        return [val.cpu().numpy() for _, val in self.global_net.state_dict().items()]

    def test(self):
        loss, accuracy = test(self.global_net, self.test_dl, "central_server")
        self.metrics["global_loss"].append(loss)
        self.metrics["global_accuracy"].append(accuracy)
