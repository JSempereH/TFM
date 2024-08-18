import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from typing import List, Tuple
from utils import save_train_metrics_to_csv, save_test_metrics_to_csv

# TODO: This must come from the Hydra's config
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CRITERION = torch.nn.CrossEntropyLoss()
OPTIMIZER = "ADAM"


class Net(nn.Module):
    def __init__(self) -> None:
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class Net_MNIST(nn.Module):
    def __init__(self) -> None:
        super(Net_MNIST, self).__init__()
        # Change the input channels from 3 to 1 for grayscale images
        self.conv1 = nn.Conv2d(1, 6, 5)  
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        # Adjust the input size of the first fully connected layer
        self.fc1 = nn.Linear(16 * 4 * 4, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        # Flatten the tensor for the fully connected layer
        x = x.view(-1, 16 * 4 * 4)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


def select_optimizer(net: nn.Module):
    if OPTIMIZER == "ADAM":
        return torch.optim.Adam(filter(lambda p: p.requires_grad, net.parameters()))


def train(net, trainloader, epochs: int, server_name: str):
    """
    Train the neural network on the training set.

    Args:
        net: The neural network model to be trained.
        trainloader: DataLoader for the training set.
        epochs (int): Number of training epochs.
        server_name (str): Identifier for the server or training session.

    Returns:
        None
    """
    criterion = CRITERION
    optimizer = select_optimizer(net=net)

    pbar = tqdm(range(epochs), desc=f"Training {server_name}:", colour="green")
    net.to(DEVICE)
    net.train()
    for epoch in pbar:
        correct, total, epoch_loss = 0, 0, 0.0
        for images, labels in trainloader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            outputs = net(images)
            loss = criterion(net(images), labels)
            loss.backward()
            optimizer.step()
            # Metrics
            epoch_loss += loss
            total += labels.size(0)
            correct += (torch.max(outputs.data, 1)[1] == labels).sum().item()
        epoch_loss /= len(trainloader.dataset)
        epoch_acc = correct / total
        pbar.set_description(
            f"Epoch {epoch+1}: train loss {epoch_loss}, accuracy {epoch_acc}"
        )
        save_train_metrics_to_csv(
            epoch=epoch + 1,
            train_loss=epoch_loss.item(),
            train_accuracy=epoch_acc,
            server_name=server_name,
        )


def train_fedprox(
    net, trainloader, epochs: int, server_name: str, mu: float, global_model: nn.Module
):
    """
    Train the network on the training set using the FedProx algorithm.

    Args:
        net: The neural network model to be trained.
        trainloader: DataLoader for the training set.
        epochs (int): Number of training epochs.
        server_name (str): Identifier for the server or training session.
        mu (float): Proximal term coefficient for FedProx regularization.
        global_model (nn.Module): The global model used for FedProx regularization.

    Returns:
        None
    """
    criterion = CRITERION
    optimizer = select_optimizer(net=net)

    pbar = tqdm(range(epochs), desc=f"Training {server_name}:", colour="green")

    global_weight_collector = list(global_model.to(DEVICE).parameters())
    net.to(DEVICE)
    net.train()
    for epoch in pbar:
        correct, total, epoch_loss = 0, 0, 0.0
        for images, labels in trainloader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            outputs = net(images)
            loss = criterion(net(images), labels)
            fed_prox_reg = 0.0
            for param_index, param in enumerate(net.parameters()):
                fed_prox_reg += (mu / 2) * torch.norm(
                    (param - global_weight_collector[param_index])
                ) ** 2
            loss += fed_prox_reg
            loss.backward()
            optimizer.step()
            # Metrics
            epoch_loss += loss
            total += labels.size(0)
            correct += (torch.max(outputs.data, 1)[1] == labels).sum().item()
        epoch_loss /= len(trainloader.dataset)
        epoch_acc = correct / total
        pbar.set_description(
            f"Epoch {epoch+1}: train loss {epoch_loss}, accuracy {epoch_acc}"
        )
        save_train_metrics_to_csv(
            epoch=epoch + 1,
            train_loss=epoch_loss.item(),
            train_accuracy=epoch_acc,
            server_name=server_name,
        )


def train_scaffold(net, trainloader, epochs: int, server_name: str, global_model: nn.Module,
                   c_global: List[torch.Tensor], c_local: List[torch.Tensor]):
        """Train for SCAFFOLD."""
        optimizer = select_optimizer(net)
        model_params = net.state_dict()

        net.train()
        net.to(DEVICE)
        pbar = tqdm(range(epochs), desc=f"Training {server_name}:", colour="green")
        for epoch in pbar:
            correct, total, epoch_loss = 0, 0, 0.0
            for images, labels in trainloader:
                images, labels = images.to(DEVICE), labels.to(DEVICE)

                net.zero_grad()
                logits = net(images)
                loss = CRITERION(net(images), labels)
                loss.backward()

                for param, c_g, c_l in zip(net.parameters(), c_global, c_local):
                    if param.requires_grad:
                        param.grad.data += (c_g - c_l).to(DEVICE)

                optimizer.step()
                # Metrics
                epoch_loss += loss
                total += labels.size(0)
                correct += (torch.max(logits.data, 1)[1] == labels).sum().item()
            
            epoch_loss /= len(trainloader.dataset)
            epoch_acc = correct / total
            pbar.set_description(
                f"Epoch {epoch+1}: train loss {epoch_loss}, accuracy {epoch_acc}"
            )
            save_train_metrics_to_csv(
            epoch=epoch + 1,
            train_loss=epoch_loss.item(),
            train_accuracy=epoch_acc,
            server_name=server_name,
        )

        with torch.no_grad():
            y_delta = []
            c_delta = []
            c_plus = []

            for key in model_params.keys():
                y_delta.append(model_params[key].cpu() - global_model.state_dict()[key])

            coef = 1 / (epochs * optimizer.param_groups[0]['lr'])
            for c_g, c_l, y_d in zip(c_global, c_local, y_delta):
                c_plus.append(c_l - c_g - coef * y_d)

            for c_p, c_l in zip(c_plus, c_local):
                c_delta.append(c_p - c_l)

            c_local = c_plus
            return y_delta, c_delta, c_local

def test(net, testloader, server_name: str) -> Tuple[float, float]:
    """
    Evaluate the network on the test set (for clients, the test set is local).

    Args:
        net: The neural network model to be evaluated.
        testloader: DataLoader for the test set.

    Returns:
        tuple: A tuple containing the average test loss and accuracy.
    """
    criterion = torch.nn.CrossEntropyLoss()
    correct, total, loss = 0, 0, 0.0
    net.to(DEVICE)
    net.eval()
    with torch.no_grad():
        for images, labels in tqdm(testloader, desc=f"Test", colour="yellow"):
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = net(images)
            loss += criterion(outputs, labels).item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    loss /= len(testloader.dataset)
    accuracy = correct / total
    save_test_metrics_to_csv(loss, accuracy, server_name)
    return loss, accuracy
