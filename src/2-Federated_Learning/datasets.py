import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader, random_split, Subset
from torchvision.datasets import CIFAR10, MNIST, FashionMNIST
import torch
import matplotlib.pyplot as plt
from typing import List, Tuple
import numpy as np

torch.manual_seed(42)
np.random.seed(42)


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


def download_datasets(dataset_name: str) -> Tuple[Dataset, Dataset]:
    """
    Downloads and returns the specified dataset.

    Args:
        dataset_name (str): Name of the dataset.

    Returns:
        tuple: A tuple containing the training and test datasets.
    """
    if dataset_name == "CIFAR10":
        transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )
        trainset = CIFAR10("./dataset", train=True, download=True, transform=transform)
        testset = CIFAR10("./dataset", train=False, download=True, transform=transform)

    elif dataset_name == "MNIST":
        transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
        )
        trainset = MNIST("./dataset", train=True, download=True, transform=transform)
        testset = MNIST("./dataset", train=False, download=True, transform=transform)

    elif dataset_name == "FashionMNIST":
        transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
        )
        trainset = FashionMNIST(
            root="./dataset", train=True, download=True, transform=transform
        )
        testset = FashionMNIST(
            root="./dataset", train=False, download=True, transform=transform
        )

    else:
        raise ValueError(
            "Dataset not implemented! Check the name."
        )

    return trainset, testset


def split_dataset_randomly(
    dataset: Dataset, num_clients: int, seed: int = 42
) -> List[Dataset]:
    """
    Split the dataset randomly into `num_clients` partitions.

    Args:
        dataset (Dataset): The dataset to be split.
        num_clients (int): The number of partitions (clients).
        seed (int): Random seed for reproducibility. Default is 42.

    Returns:
        list: A list containing `num_clients` partitions of the dataset.
    """
    # Calculate partition sizes
    partition_size = len(dataset) // num_clients
    remainder = len(dataset) % num_clients
    lengths = [partition_size] * num_clients
    for i in range(remainder):
        lengths[i] += 1

    # Split dataset into partitions
    partitions = random_split(
        dataset, lengths, generator=torch.Generator().manual_seed(seed)
    )

    return partitions


def create_train_val_loaders(
    partitions: List[Dataset],
    batch_size: int = 32,
    val_split: float = 0.1,
    shuffle: bool = True,
    seed: int = 42,
) -> Tuple[List[DataLoader], List[DataLoader]]:
    """
    Create DataLoader instances for training and validation sets for each partition.

    Args:
        partitions (list[Dataset]): A list containing partitions of the dataset.
        batch_size (int): Batch size for DataLoader. Default is 32.
        val_split (float): Fraction of data to include in the validation set. Default is 0.1.
        shuffle (bool): Whether to shuffle the data. Default is True.
        seed (int): Random seed for reproducibility. Default is 42.

    Returns:
        tuple: A tuple containing lists of DataLoader instances for training and validation sets.
    """
    if not 0 < val_split < 1:
        raise ValueError("val_split must be between 0 and 1.")

    trainloaders = []
    valloaders = []

    for ds in partitions:
        len_val = int(len(ds) * val_split)
        len_train = len(ds) - len_val
        lengths = [len_train, len_val]

        ds_train, ds_val = random_split(
            ds, lengths, generator=torch.Generator().manual_seed(seed)
        )

        trainloader = DataLoader(ds_train, batch_size=batch_size, shuffle=shuffle)
        valloader = DataLoader(ds_val, batch_size=batch_size)

        trainloaders.append(trainloader)
        valloaders.append(valloader)

    return trainloaders, valloaders


def load_IID_datasets(
    num_clients: int, dataset_name: str = "CIFAR10"
) -> Tuple[List[DataLoader], List[DataLoader], DataLoader]:
    """
    Load IID (independently and identically distributed) datasets.

    Args:
        num_clients (int): Number of clients.
        dataset_name (str): Name of the dataset. Default is "CIFAR10".

    Returns:
        tuple: A tuple containing lists of DataLoader instances for training and validation sets, and a DataLoader instance for the test set.
    """
    trainset, testset = download_datasets(dataset_name)
    partitions = split_dataset_randomly(trainset, num_clients)

    # Split each partition into train/val and create DataLoader
    trainloaders, valloaders = create_train_val_loaders(partitions)
    testloader = DataLoader(testset, batch_size=32)
    return trainloaders, valloaders, testloader


def load_non_iid_dataloaders_Dirichlet(
    num_clients, dataset_name="CIFAR10", beta=0.5
) -> Tuple[List[DataLoader], List[DataLoader], DataLoader]:
    """
    Dirichlet Distribution-based label imbalance. Different P(y_i) among clients.

    Args:
        num_clients (int): Number of clients.
        dataset_name (str): Name of the dataset. Default is "CIFAR10".
        beta (float): Concentration parameter for the Dirichlet distribution. Default is 0.5.

    Returns:
        tuple: A tuple containing lists of DataLoader instances for training and validation sets, and a DataLoader instance for the test set.
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
) -> Tuple[List[DataLoader], List[DataLoader], DataLoader]:
    """
    Quantity-based label imbalance partition. Different P(y_i) among clients.
    len(labels_per_party) already gives the number of clients.
    For example: [3,3,4] for CIFAR10 would give all the first 3 labels to client 1,
    the next 3 labels to client 2 and the last 4 labels to client 3.

    Args:
        labels_per_party (list): A list specifying the number of labels each party/client possesses.
        dataset_name (str): Name of the dataset. Default is "CIFAR10".

    Returns:
        tuple: A tuple containing lists of DataLoader instances for training and validation sets, and a DataLoader instance for the test set.
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


def load_non_iid_dataloaders_Noise(
    num_clients: int, noise_level: float, dataset_name="CIFAR10"
):
    """
    Non-IID partitioning strategy with Gaussian noise. Adds increasing levels of Gaussian noise to each client's data.

    Args:
        num_clients (int): Number of parties (clients).
        noise_level (float): Base level of Gaussian noise to be added.
        dataset_name (str): Name of the dataset. Default is "CIFAR10".

    Returns:
        tuple: A tuple containing lists of DataLoader instances for training and validation sets, and a DataLoader instance for the test set.
    """
    def add_gaussian_noise(dataset, noise_level, party_idx, num_clients):
        """
        Adds Gaussian noise to a dataset.

        Args:
            dataset (list): The dataset to which noise will be added.
            noise_level (float): Base level of Gaussian noise.
            party_idx (int): Index of the current client.
            num_clients (int): Total number of clients.

        Returns:
            list: The dataset with added Gaussian noise.
        """
        noisy_dataset = []
        for data in dataset:
            image, label = data
            noise_scale = (
                noise_level * (party_idx + 1) / num_clients
            )  # Increase noise level with party index
            noisy_image = image + torch.randn_like(image) * noise_scale
            noisy_dataset.append((noisy_image, label))
        return noisy_dataset

    trainset, testset = download_datasets(dataset_name)
    partitions = split_dataset_randomly(trainset, num_clients)
    noisy_partitions = [
        add_gaussian_noise(partition, noise_level, idx, num_clients)
        for idx, partition in enumerate(partitions)
    ]
    trainloaders, valloaders = create_train_val_loaders(noisy_partitions)
    testloader = DataLoader(testset, batch_size=32)
    return trainloaders, valloaders, testloader


def load_non_iid_dataloaders_QuantitySkew(
    num_clients: int,
    dataset_name: str = "CIFAR10",
    beta: float = 0.5,
    batch_size: int = 32,
    val_split: float = 0.1,
    shuffle: bool = True,
    seed: int = 42,
):
    """
    Quantity Skew partitioning strategy. Varying dataset sizes across parties using Dirichlet distribution.

    Args:
        num_clients (int): Number of parties (clients).
        dataset_name (str): Name of the dataset. Default is "CIFAR10".
        beta (float): Concentration parameter for Dirichlet distribution. Default is 0.5.
        batch_size (int): Batch size for DataLoader. Default is 32.
        val_split (float): Fraction of data to include in the validation set. Default is 0.1.
        shuffle (bool): Whether to shuffle the data. Default is True.
        seed (int): Random seed for reproducibility. Default is 42.

    Returns:
        tuple: A tuple containing lists of DataLoader instances for training and validation sets, and a DataLoader instance for the test set.
    """
    trainset, testset = download_datasets(dataset_name)
    proportions = np.random.dirichlet([beta] * num_clients)
    # Scale proportions to total number of samples
    quantities = (proportions * len(trainset)).astype(int)
    quantities[-1] += len(trainset) - sum(quantities)
    # Partition the dataset based on the quantities
    partitions = random_split(trainset, lengths=quantities)

    # Create train and validation loaders
    trainloaders, valloaders = create_train_val_loaders(
        partitions,
        batch_size=batch_size,
        val_split=val_split,
        shuffle=shuffle,
        seed=seed,
    )

    # Create test loader
    testloader = DataLoader(testset, batch_size=batch_size)

    return trainloaders, valloaders, testloader


#################################3


def plot_label_bars_multi(
    train_loaders: List[DataLoader],
    val_loaders: List[DataLoader],
    dataset_name: str,
    extra_name: str = "",
):
    """
    Plots bar subplots of labels in lists of CIFAR10 DataLoaders.

    Args:
        train_loaders (List[DataLoader]): List of CIFAR10 training DataLoaders.
        val_loaders (List[DataLoader]): List of CIFAR10 validation DataLoaders.
        dataset_name (str): Name of the dataset.
        extra_name (str): Additional name to append to the plot file name.
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

    classes_names = download_datasets(dataset_name)[0].classes
    for ax, title in zip(axes, ["Train", "Validation"]):
        ax.set_xticks(np.arange(10) + (bar_width * (num_loaders - 1)) / 2)
        ax.set_xticklabels(classes_names, rotation=45, ha="right")
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
        + extra_name
        + ".png"
    )
    plt.savefig(plot_title)
