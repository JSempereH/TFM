from typing import List, Tuple
from omegaconf import DictConfig
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
from datasets import *
from strategies import *
import csv
import hydra
import logging
import sys


# Configure logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    handlers=[
                        logging.StreamHandler(sys.stdout),  # Log to console
                        logging.FileHandler("main.log")  # Log to file
                    ])

logger = logging.getLogger(__name__)

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
    sentence_length = len(sentence)
    print("#" * (sentence_length + 4))
    print("# " + sentence + " #")
    print("#" * (sentence_length + 4))


def print_experiment_config(cfg: DictConfig, n_blocks=30) -> None:
    separator = n_blocks * "-"
    logger.info(separator)
    logger.info(f"The dataset is {cfg.dataset}")
    logger.info(f"The partition strategy is: {cfg.iid}")
    logger.info(f"The FL strategy is: {cfg.fl_strategy}")
    logger.info(f"The number of clients is: {cfg.num_clients}")
    logger.info(f"The local epochs (per client) list is: {cfg.list_of_epochs}")
    if (cfg.mu is not None) and (cfg.fl_strategy == "FedProx" or cfg.fl_strategy == "FedNova"):
        logger.info(f"Mu is: {cfg.mu}")
    if (cfg.labels_per_party is not None) and (cfg.iid == "Non_IID_Quantity_Based"):
        logger.info(f"Labels per party is: {cfg.labels_per_party}")
    if (cfg.beta is not None) and (cfg.iid == "Non_IID_Dirichlet" or cfg.iid == "Non_IID_Quantity_Skew"):
        logger.info(f"Beta value is: {cfg.beta}")
    logger.info(f"Do all the models start with the same weights? {cfg.same_initial_weights}")
    logger.info(f"Number of rounds: {cfg.n_rounds}")
    if (cfg.noise_level is not None) and (cfg.iid == "Non_IID_Noise"):
        logger.info(f"Noise level: {cfg.noise_level}")
    logger.info(f"The batch size is {cfg.batch_size}")
    logger.info(f"The learning rate is {cfg.lr}")
    logger.info(separator + "\n")


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
    elif cfg.iid == "Non_IID_Quantity_Skew":
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
    elif cfg.fl_strategy == "FedNova":
        return FedNova(cfg.mu)
    elif cfg.fl_strategy == "SCAFFOLD":
        return SCAFFOLD(cfg.global_lr)


def save_train_metrics_to_csv(epoch, train_loss, train_accuracy, server_name: str):
    hydra_cfg = hydra.core.hydra_config.HydraConfig.get()
    filepath = (
        hydra_cfg["runtime"]["output_dir"] + "/" + server_name + "_train_metrics.csv"
    )
    with open(filepath, mode="a", newline="") as file:
        writer = csv.writer(file)
        # If it's an empty csv file, write header
        if file.tell() == 0:
            writer.writerow(["epoch", "train_loss", "train_accuracy"])
        # Escribir las métricas
        writer.writerow([epoch, train_loss, train_accuracy])


def save_test_metrics_to_csv(test_loss, test_accuracy, server_name: str):
    hydra_cfg = hydra.core.hydra_config.HydraConfig.get()
    filepath = (
        hydra_cfg["runtime"]["output_dir"] + "/" + server_name + "_test_metrics.csv"
    )
    with open(filepath, mode="a", newline="") as file:
        writer = csv.writer(file)
        # If it's an empty csv file, write header
        if file.tell() == 0:
            writer.writerow(["test_loss", "test_accuracy"])
        # Escribir las métricas
        writer.writerow([test_loss, test_accuracy])


def check_config_parameters(cfg: DictConfig) -> None:
    if len(cfg.list_of_epochs) != cfg.num_clients:
        raise RuntimeError(
            f"{bcolors.FAIL}Number of clients != len(list_of_epochs).{bcolors.ENDC}"
        )
    if cfg.dataset not in ["CIFAR10", "MNIST", "FashionMNIST"]:
        raise RuntimeError(
            f"{bcolors.FAIL}Wrong Dataset name, the possible datasets are: CIFAR10, MNIST or FashionMNIST.{bcolors.ENDC}"
        )
    if cfg.iid == "IID" and any(
        param is not None for param in [cfg.beta, cfg.labels_per_party, cfg.noise_level]
    ):
        raise RuntimeError(
            f"{bcolors.FAIL}Partition strategy IID is not compatible with arguments beta, labels_per_party nor noise_level.{bcolors.ENDC}"
        )
    if cfg.iid == "Non_IID_Dirichlet" and cfg.beta is None:
        raise RuntimeError(
            f"{bcolors.FAIL}Partition strategy Non_IID_Dirichlet needs a beta argument.{bcolors.ENDC}"
        )
    if cfg.iid == "Non_IID_Quantity_Skew" and cfg.beta is None:
        raise RuntimeError(
            f"{bcolors.FAIL}Partition strategy Non_IID_Quantity_Skew needs a beta argument.{bcolors.ENDC}"
        )
    if cfg.iid == "Non_IID_Noise" and cfg.noise_level is None:
        raise RuntimeError(
            f"{bcolors.FAIL}Partition strategy Non_IID_Noise needs a noise_level argument.{bcolors.ENDC}"
        )
    if cfg.iid == "Non_IID_Dirichlet" and any(
        param is not None for param in [cfg.labels_per_party, cfg.noise_level]
    ):
        pass  # TODO: Finish the checks
