from datasets import *
from server import Client, Central_Server
from models import Net
from utils import (
    are_models_equal,
    print_block,
    print_experiment_config,
    get_dataloaders,
    select_fl_strategy,
)
from strategies import FedAvg, FedProx
import copy
import hydra
from omegaconf import DictConfig
import os


@hydra.main(version_base="1.3", config_path=os.getcwd(), config_name="config")
def main(cfg: DictConfig) -> None:
    working_dir = os.getcwd()
    print(f"The current working directory is {working_dir} \n")
    print_experiment_config(cfg)
    # To access elements of the config

    # TODO: Make an utils' function to check that the config parameters are valid
    if len(cfg.list_of_epochs) != cfg.num_clients:
        raise RuntimeError(f"Number of clients != len(list_of_epochs).")

    train_dl, val_dl, test_dl = get_dataloaders(cfg)

    list_clients = []
    global_net = Net()
    central_server = Central_Server(global_net, test_dl)
    for i in range(cfg.num_clients):
        if cfg.same_initial_weights:
            client_net = copy.deepcopy(global_net)  # All clients same initial weights
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
        f"\nAre models equal?: {are_models_equal([client.net for client in list_clients]) and are_models_equal([list_clients[0].net, central_server.global_net])}"
    )

    strategy = select_fl_strategy(cfg)

    for round in range(cfg.n_rounds):
        print_block(f"Round {round+1}" + "/" + f"{cfg.n_rounds}")
        for client in list_clients:
            client.fit(
                client.get_parameters(), strategy, global_net=central_server.global_net
            )

        # Aggregation strategy
        first_weight_tensor = None

        for name, param in central_server.global_net.named_parameters():
            first_weight_tensor = param
            break
        # print(f"Before FedAvg: {first_weight_tensor}")

        strategy.aggregation(
            global_model=central_server.global_net,
            list_local_models=[client.net for client in list_clients],
            list_len_datasets=[
                client.local_train_dataloader.dataset.__len__()
                for client in list_clients
            ],
        )

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

# NUM_CLIENTS = 3
# LIST_OF_EPOCHS = [2, 2, 2]
# SAME_INITIAL_WEIGHTS = True
# N_ROUNDS = 2
# MU = 0.01
# DATASET_NAME = "CIFAR10"
# IID = "Non_IID_Noise"

# BETA = 0.5  # beta argument for Dirichlet Distribution, needed for IID = 'Non_IID_Dirichlet' or 'Non_IID_Qunatity_Skew'
# LABELS_PER_PARTY = [
#     2,
#     2,
#     6,
# ]  # Which labels belong to the each client, for IID = 'Non_IID_Quantity_Based'
# NOISE_LEVEL = 0.1  # Sigma argument in Gaussian distribution, for IID = 'Non_IID_Noise'
