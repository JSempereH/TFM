from datasets import *
from server import Client, Central_Server
from models import Net
from utils import are_models_equal, print_block
from strategies import FedAvg, FedProx
import copy

NUM_CLIENTS = 3
LIST_OF_EPOCHS = [2, 2, 2]
SAME_INITIAL_WEIGHTS = True
N_ROUNDS = 2
MU = 0.01
DATASET_NAME = "CIFAR10"
IID = "Non_IID_Noise"

BETA = 0.5  # beta argument for Dirichlet Distribution, needed for IID = 'Non_IID_Dirichlet' or 'Non_IID_Qunatity_Skew'
LABELS_PER_PARTY = [
    2,
    2,
    6,
]  # Which labels belong to the each client, for IID = 'Non_IID_Quantity_Based'
NOISE_LEVEL = 0.1  # Sigma argument in Gaussian distribution, for IID = 'Non_IID_Noise'

if len(LIST_OF_EPOCHS) != NUM_CLIENTS:
    raise RuntimeError(f"NUM_CLIENTS != len(LIST_OF_EPOCHS).")


# Select partition strategy
if IID == "IID":
    train_dl, val_dl, test_dl = load_IID_datasets(
        num_clients=NUM_CLIENTS, dataset_name=DATASET_NAME
    )
elif IID == "Non_IID_Dirichlet":
    if BETA is None:
        raise NameError(
            "Beta argument for Dirichlet Non-IID partition strategy not defined."
        )
    train_dl, val_dl, test_dl = load_non_iid_dataloaders_Dirichlet(
        NUM_CLIENTS, DATASET_NAME, BETA
    )
elif IID == "Non_IID_Quantity_Based":
    if LABELS_PER_PARTY is None:
        raise NameError(
            "Labels_per_party argument for quantity-based Non-IID partition strategy not defined."
        )
    if len(LABELS_PER_PARTY) != NUM_CLIENTS:
        raise RuntimeError("NUM_CLIENTS != len(LABELS_PER_PARTY).")
    train_dl, val_dl, test_dl = load_non_iid_dataloaders_quantity_based(
        LABELS_PER_PARTY, DATASET_NAME
    )
elif IID == "Non_IID_Qunatity_Skew":
    if BETA is None:
        raise NameError(
            "Beta argument for Dirichlet Non-IID partition strategy not defined."
        )
    train_dl, val_dl, test_dl = load_non_iid_dataloaders_QuantitySkew(
        NUM_CLIENTS, DATASET_NAME, BETA
    )
elif IID == "Non_IID_Noise":
    train_dl, val_dl, test_dl = load_non_iid_dataloaders_Noise(
        NUM_CLIENTS, NOISE_LEVEL, DATASET_NAME
    )


list_clients = []
global_net = Net()
central_server = Central_Server(global_net, test_dl)
for i in range(NUM_CLIENTS):
    if SAME_INITIAL_WEIGHTS:
        client_net = copy.deepcopy(global_net)  # All clients same initial weights
    else:
        client_net = Net()  # Each client has different initial weights
    client = Client(
        client_name=f"Client {i+1}",
        net=client_net,
        local_train_dataloader=train_dl[i],
        local_val_dataloader=val_dl[i],
        n_epochs=LIST_OF_EPOCHS[i],
    )
    list_clients.append(client)


print(
    f"\nAre models equal?: {are_models_equal([client.net for client in list_clients]) and are_models_equal([list_clients[0].net, central_server.global_net])}"
)


# strategy = FedAvg()
strategy = FedProx(global_model=central_server.global_net, mu=MU)

for round in range(N_ROUNDS):
    print_block(f"Round {round+1}" + "/" + f"{N_ROUNDS}")
    for client in list_clients:
        client.fit(client.get_parameters(), strategy)

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
            client.local_train_dataloader.dataset.__len__() for client in list_clients
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
