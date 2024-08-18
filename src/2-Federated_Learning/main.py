from datasets import *
from server import Client, Central_Server
from models import Net, Net_MNIST
from utils import (
    are_models_equal,
    print_block,
    print_experiment_config,
    get_dataloaders,
    select_fl_strategy,
    check_config_parameters,
)
from strategies import SCAFFOLD
import copy
import hydra
from omegaconf import DictConfig
import os
import logging

log = logging.getLogger(__name__)


@hydra.main(version_base="1.3", config_path=os.getcwd(), config_name="config")
def main(cfg: DictConfig) -> None:
    check_config_parameters(cfg)

    working_dir = os.getcwd()
    print(f"The current working directory is {working_dir} \n")
    print_experiment_config(cfg)
    train_dl, val_dl, test_dl = get_dataloaders(cfg)

    list_clients = []
    if cfg.dataset == "MNIST":
        global_net = Net_MNIST()
    else:
        global_net = Net()
    central_server = Central_Server(global_net, test_dl)
    for i in range(cfg.num_clients):
        if cfg.same_initial_weights:
            client_net = copy.deepcopy(global_net)  # All clients same initial weights
        else:
            if cfg.dataset == "MNIST":
                client_net = Net_MNIST()
            else:
                client_net = Net()  # Each client has different initial weights
        client = Client(
            client_name=f"Client {i+1}",
            net=client_net,
            local_train_dataloader=train_dl[i],
            local_val_dataloader=val_dl[i],
            n_epochs=cfg.list_of_epochs[i],
        )
        list_clients.append(client)

    print(
        f"\nAre models equal at initialization?: {are_models_equal([client.net for client in list_clients]) and are_models_equal([list_clients[0].net, central_server.global_net])}"
    )

    strategy = select_fl_strategy(cfg)

    for round in range(cfg.n_rounds):
        list_of_epochs = cfg.list_of_epochs
        # TODO: Make possible variable local steps
        # Here, we would change the local epochs of each client
        print_block(f"Round {round+1}" + "/" + f"{cfg.n_rounds}")
        for client in list_clients:
            client.fit(
                client.get_parameters(), strategy, global_net=central_server.global_net
            )

        central_server.aggregate(list_clients, strategy, list_of_epochs)

        for client in list_clients:
            client.set_parameters(central_server.get_parameters())
        print(
            f"\nAre models equal?: {are_models_equal([client.net for client in list_clients]) and are_models_equal([list_clients[0].net, central_server.global_net])}"
        )

        # Use global net for test_dl
        central_server.test()
        # Each client test its own val_dl with their own local net, which is now the global net
        for client in list_clients:
            client.evaluate()


if __name__ == "__main__":
    main()
