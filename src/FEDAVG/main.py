from datasets import *
from server import Client
from models import Net, are_models_equal
from strategies import FedAvg, FedProx
import copy

NUM_CLIENTS = 3
LIST_OF_EPOCHS = [2, 2, 2]
SAME_INITIAL_WEIGHTS = True
N_ROUNDS = 2


if len(LIST_OF_EPOCHS) != NUM_CLIENTS:
    raise ValueError(f"NUM_CLIENTS != len(LIST_OF_EPOCHS).")

train_dl, val_dl, test_dl = load_IID_datasets(
    num_clients=NUM_CLIENTS, dataset_name="CIFAR10"
)

client_1 = Client(
    client_name="Cliente 1",
    net=Net(),
    local_train_dataloader=train_dl[0],
    local_val_dataloader=val_dl[0],
    n_epochs=4,
)

list_clients = []
global_model = Net()
for i in range(NUM_CLIENTS):
    if SAME_INITIAL_WEIGHTS:
        client_net = copy.deepcopy(global_model)  # All clients same initial weights
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
    f"\nAre models equal?: {are_models_equal([client.net for client in list_clients])}"
)

print("\nNow we train client 1:\n")
client_1 = list_clients[0]
# client_1.fit(client_1.get_parameters())

# print(f"\nAre models equal?: {are_models_equal([client.net for client in list_clients])}")


def print_block(sentence):
    # Calculate the length of the sentence for formatting
    sentence_length = len(sentence)

    # Print the top border
    print("#" * (sentence_length + 4))

    # Print the sentence with borders
    print("# " + sentence + " #")

    # Print the bottom border
    print("#" * (sentence_length + 4))


# strategy = FedAvg()
strategy = FedProx(global_model=global_model, mu=0.01)

for round in range(N_ROUNDS):
    print_block(f"Round {round+1}" + "/" + f"{N_ROUNDS}")
    for client in list_clients:
        client.fit(client.get_parameters(), strategy)

    # Aggregation strategy
    first_weight_tensor = None

    for name, param in global_model.named_parameters():
        first_weight_tensor = param
        break
    # print(f"Before FedAvg: {first_weight_tensor}")

    strategy.aggregation(
        global_model=global_model,
        list_local_models=[client.net for client in list_clients],
        list_len_datasets=[
            client.local_train_dataloader.dataset.__len__() for client in list_clients
        ],
    )

    for name, param in global_model.named_parameters():
        first_weight_tensor = param
        break
    # print(f"After FedAvg {first_weight_tensor}")

    # TODO: HACER UN client.set_parameters con los parametros del modelo agregado
