from typing import List, Tuple
from omegaconf import DictConfig
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
from datasets import *
from strategies import *


class bcolors:
    HEADER = "\033[95m"
    OKBLUE = "\033[94m"
    OKCYAN = "\033[96m"
    OKGREEN = "\033[92m"
    WARNING = "\033[93m"
    FAIL = "\033[91m"
    ENDC = "\033[0m"
    BOLD = "\033[1m"
    UNDERLINE = "\033[4m"


def are_models_equal(model_list: List[nn.Module]):
    # Get the state_dicts of the models in the list
    state_dicts_list = [model.state_dict() for model in model_list]

    # Check if the keys of state_dicts are the same for all models
    if any(
        set(state_dict.keys()) != set(state_dicts_list[0].keys())
        for state_dict in state_dicts_list[1:]
    ):
        return False

    # Check if the values (weights) of the corresponding keys are the same for all models
    for key in state_dicts_list[0].keys():
        if any(
            not torch.equal(state_dict[key], state_dicts_list[0][key])
            for state_dict in state_dicts_list[1:]
        ):
            return False

    # If all checks pass, the models are equal
    return True


def print_block(sentence):
    # Calculate the length of the sentence for formatting
    sentence_length = len(sentence)

    # Print the top border
    print("#" * (sentence_length + 4))

    # Print the sentence with borders
    print("# " + sentence + " #")

    # Print the bottom border
    print("#" * (sentence_length + 4))


def print_experiment_config(cfg: DictConfig, n_blocks=30) -> None:
    print(n_blocks * "-")
    print(f"The dataset is {bcolors.BOLD}{cfg.dataset}{bcolors.ENDC}")
    print(f"The paritition strategy is: {bcolors.BOLD}{cfg.iid}{bcolors.ENDC}")
    print(f"The FL strategy is: {bcolors.BOLD}{cfg.fl_strategy}{bcolors.ENDC}")
    print(f"The number of clients is: {bcolors.BOLD}{cfg.num_clients}{bcolors.ENDC}")
    print(
        f"The local epochs (per client) list is: {bcolors.BOLD}{cfg.list_of_epochs}{bcolors.ENDC}"
    )
    if (cfg.labels_per_party is not None) and (cfg.iid == "Non_IID_Quantity_Based"):
        print(
            f"Labels per party is: {bcolors.BOLD}{cfg.labels_per_party}{bcolors.ENDC}"
        )
    if (cfg.beta is not None) and (
        cfg.iid == "Non_IID_Dirichlet" or cfg.iid == "Non_IID_Qunatity_Skew"
    ):
        print(f"Beta value is: {bcolors.BOLD}{cfg.beta}{bcolors.ENDC}")
    print(
        f"Do all the models start with the same weights? {bcolors.BOLD}{cfg.same_initial_weights}{bcolors.ENDC}"
    )
    print(f"Number of rounds: {bcolors.BOLD}{cfg.n_rounds}{bcolors.ENDC}")
    if (cfg.noise_level is not None) and (cfg.iid == "Non_IID_Noise"):
        print(f"Noise level: {bcolors.BOLD}{cfg.noise_level}{bcolors.ENDC}")
    print(f"The batch size is {bcolors.BOLD}{cfg.batch_size}{bcolors.ENDC}")
    print(f"The learning rate is {bcolors.BOLD}{cfg.lr}{bcolors.ENDC}")
    print(n_blocks * "-" + "\n")


def get_dataloaders(
    cfg: DictConfig,
) -> Tuple[List[DataLoader], List[DataLoader], DataLoader]:
    # Select partition strategy
    if cfg.iid == "IID":
        train_dl, val_dl, test_dl = load_IID_datasets(
            num_clients=cfg.num_clients, dataset_name=cfg.dataset
        )
    elif cfg.iid == "Non_IID_Dirichlet":
        if cfg.beta is None:
            raise NameError(
                "Beta argument for Dirichlet Non-IID partition strategy not defined."
            )
        train_dl, val_dl, test_dl = load_non_iid_dataloaders_Dirichlet(
            cfg.num_clients, cfg.dataset, cfg.beta
        )
    elif cfg.iid == "Non_IID_Quantity_Based":
        if cfg.labels_per_party is None:
            raise NameError(
                "Labels_per_party argument for quantity-based Non-IID partition strategy not defined."
            )
        if len(cfg.labels_per_party) != cfg.num_clients:
            raise RuntimeError("NUM_CLIENTS != len(LABELS_PER_PARTY).")
        train_dl, val_dl, test_dl = load_non_iid_dataloaders_quantity_based(
            cfg.labels_per_party, cfg.dataset
        )
    elif cfg.iid == "Non_IID_Qunatity_Skew":
        if cfg.beta is None:
            raise NameError(
                "Beta argument for Dirichlet Non-IID partition strategy not defined."
            )
        train_dl, val_dl, test_dl = load_non_iid_dataloaders_QuantitySkew(
            cfg.num_clients, cfg.dataset, cfg.beta
        )
    elif cfg.iid == "Non_IID_Noise":
        train_dl, val_dl, test_dl = load_non_iid_dataloaders_Noise(
            cfg.num_clients, cfg.noise_level, cfg.dataset
        )
    return train_dl, val_dl, test_dl


def select_fl_strategy(cfg: DictConfig) -> Strategy:
    if cfg.fl_strategy == "FedAvg":
        return FedAvg()
    elif cfg.fl_strategy == "FedProx":
        return FedProx(cfg.mu)
