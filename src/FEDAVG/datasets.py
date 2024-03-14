import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split, Subset
from torchvision.datasets import CIFAR10
import torch
import matplotlib
import matplotlib.pyplot as plt
from typing import List
import numpy as np
import os


def download_datasets(dataset_name: str):
    if dataset_name == "CIFAR10":
        transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )
        trainset = CIFAR10("./dataset", train=True, download=True, transform=transform)
        testset = CIFAR10("./dataset", train=False, download=True, transform=transform)

    #   TODO: Add more datasets: MNIST, EMNIST, etc

    return trainset, testset


def load_IID_datasets(num_clients: int, dataset_name: str = "CIFAR10"):
    # Download and transform CIFAR-10 (train and test)

    trainset, testset = download_datasets(dataset_name)

    # Split training set into `num_clients` partitions to simulate different local datasets
    partition_size = len(trainset) // num_clients
    remainder = len(trainset) % num_clients  # We dont want to loose any data
    lengths = [partition_size] * num_clients
    for i in range(remainder):
        lengths[i] += 1

    datasets = random_split(trainset, lengths, torch.Generator().manual_seed(42))

    # Split each partition into train/val and create DataLoader
    trainloaders = []
    valloaders = []
    for ds in datasets:
        len_val = len(ds) // 10  # 10 % validation set
        len_train = len(ds) - len_val
        lengths = [len_train, len_val]
        ds_train, ds_val = random_split(ds, lengths, torch.Generator().manual_seed(42))
        trainloaders.append(DataLoader(ds_train, batch_size=32, shuffle=True))
        valloaders.append(DataLoader(ds_val, batch_size=32))
    testloader = DataLoader(testset, batch_size=32)
    return trainloaders, valloaders, testloader


def load_non_iid_dataloaders_Dirichlet(num_clients, dataset_name="CIFAR10", beta=0.5):
    """
    Dirichlet Distribution-based label imbalance. Different P(y_i) among clients.
    """
    trainset, testset = download_datasets(dataset_name)
    num_classes = len(trainset.classes)
    proportions = np.random.dirichlet([beta] * num_classes, size=num_clients)
    trainloaders = []
    valloaders = []

    for client_proportions in proportions:
        client_indices = []
        for class_idx, class_proportion in enumerate(client_proportions):
            class_samples = np.where(np.array(trainset.targets) == class_idx)[0]
            num_samples = int(len(class_samples) * class_proportion)
            sampled_indices = np.random.choice(
                class_samples, num_samples, replace=False
            )
            client_indices.extend(sampled_indices.tolist())

        train_data = torch.utils.data.Subset(trainset, client_indices)
        val_size = int(len(train_data) * 0.1)  # 10% for validation
        train_data, val_data = torch.utils.data.random_split(
            train_data, [len(train_data) - val_size, val_size]
        )

        trainloader = DataLoader(train_data, batch_size=32, shuffle=True)
        valloader = DataLoader(val_data, batch_size=32)

        trainloaders.append(trainloader)
        valloaders.append(valloader)

    testloader = DataLoader(testset, batch_size=32)
    return trainloaders, valloaders, testloader


def load_non_iid_dataloaders_quantity_based(
    labels_per_party: list, dataset_name: str = "CIFAR10"
):
    """
    Quantity-based label imbalance partition. Different P(y_i) among clients.
    len(labels_per_party) already gives the number of clients

    """
    trainset, testset = download_datasets(dataset_name)
    num_classes = len(trainset.classes)
    total_labels = sum(labels_per_party)

    assert (
        num_classes == total_labels
    ), "The sum of labels per party must be equal to the total number of classes"

    party_data_indices = []
    start_idx = 0

    for num_labels in labels_per_party:
        end_idx = start_idx + num_labels
        label_indices = []

        for label_id in range(start_idx, end_idx):
            label_indices.extend(
                [idx for idx, (_, label) in enumerate(trainset) if label == label_id]
            )

        party_data_indices.append(label_indices)
        start_idx = end_idx

    trainloaders = []
    valloaders = []
    val_split = 0.1  # 10% for validation set
    for data_indices in party_data_indices:
        train_data = Subset(trainset, data_indices)
        num_train = len(train_data)
        indices = list(range(num_train))
        split = int(np.floor(val_split * num_train))

        np.random.shuffle(indices)
        train_idx, val_idx = indices[split:], indices[:split]

        trainloader = DataLoader(
            Subset(train_data, train_idx), batch_size=32, shuffle=True
        )
        valloader = DataLoader(Subset(train_data, val_idx), batch_size=32)

        trainloaders.append(trainloader)
        valloaders.append(valloader)

    testloader = testloader = DataLoader(testset, batch_size=32)
    return trainloaders, valloaders, testloader


#################################3


def plot_label_bars_multi(
    train_loaders: List[DataLoader],
    val_loaders: List[DataLoader],
    dataset_name: str = "CIFAR10",
):
    """
    Plotea subgráficos de barras de las etiquetas en listas de DataLoaders de CIFAR10.

    Parameters:
    - train_loaders: Lista de DataLoaders de entrenamiento de CIFAR10.
    - val_loaders: Lista de DataLoaders de validación de CIFAR10.
    """
    plt.style.use("ggplot")
    num_loaders = len(train_loaders)
    bar_width = 0.2  # Ancho de las barras

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    all_labels_train, all_labels_val = [], []
    for i, (train_loader, val_loader) in enumerate(zip(train_loaders, val_loaders)):
        labels_train, labels_val = [], []
        for _, labels in train_loader:
            labels_train.extend(labels.numpy())
        for _, labels in val_loader:
            labels_val.extend(labels.numpy())

        all_labels_train.append(labels_train)
        all_labels_val.append(labels_val)

        # Calcula la posición de las barras para cada conjunto de DataLoaders
        positions_train = np.arange(10) + i * bar_width
        positions_val = positions_train + bar_width

        # Plotea los subgráficos de barras de las etiquetas
        axes[0].bar(
            positions_train,
            np.bincount(labels_train, minlength=10),
            width=bar_width,
            label=f"Client {i + 1}",
        )
        axes[1].bar(
            positions_val,
            np.bincount(labels_val, minlength=10),
            width=bar_width,
            label=f"Client {i + 1}",
        )

    for ax, title in zip(axes, ["Train", "Validation"]):
        ax.set_xticks(np.arange(10) + (bar_width * (num_loaders - 1)) / 2)
        ax.set_xticklabels(range(10))
        ax.set_title(f"Bar plot of {dataset_name} Labels - {title}")
        ax.set_xlabel("Labels")
        ax.set_ylabel("Frequency")
        ax.legend()

    plt.tight_layout()
    plot_title = (
        "plots/Train-Val-BarPlot-"
        + str(num_loaders)
        + "-clients-"
        + dataset_name
        + ".png"
    )
    plt.savefig(plot_title)
