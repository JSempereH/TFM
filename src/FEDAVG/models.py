import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from typing import List

DEVICE = torch.device("cpu")
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

def select_optimizer(net: nn.Module):
    if OPTIMIZER == "ADAM":
        return torch.optim.Adam(filter(lambda p: p.requires_grad, net.parameters()))

def train(net, trainloader, epochs: int, server_name: str):
    """Train the network on the training set."""
    criterion = CRITERION
    optimizer = select_optimizer(net=net)

    pbar = tqdm(range(epochs), 
                desc = f"Training {server_name}:",
                colour="green")
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
        pbar.set_description(f"Epoch {epoch+1}: train loss {epoch_loss}, accuracy {epoch_acc}")


def train_fedprox(net, trainloader, epochs: int, server_name: str, mu: float, global_model: nn.Module):
    """Train the network on the training set."""
    criterion = CRITERION
    optimizer = select_optimizer(net=net)

    pbar = tqdm(range(epochs), 
                desc = f"Training {server_name}:",
                colour="green")
    
    global_weight_collector = list(global_model.to(DEVICE).parameters())
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
                fed_prox_reg += ((mu / 2) * torch.norm((param - global_weight_collector[param_index]))**2)
            loss += fed_prox_reg
            loss.backward()
            optimizer.step()
            # Metrics
            epoch_loss += loss
            total += labels.size(0)
            correct += (torch.max(outputs.data, 1)[1] == labels).sum().item()
        epoch_loss /= len(trainloader.dataset)
        epoch_acc = correct / total
        pbar.set_description(f"Epoch {epoch+1}: train loss {epoch_loss}, accuracy {epoch_acc}")


def test(net, testloader):
    """Evaluate the network on the entire test set."""
    criterion = torch.nn.CrossEntropyLoss()
    correct, total, loss = 0, 0, 0.0
    net.eval()
    with torch.no_grad():
        for images, labels in testloader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = net(images)
            loss += criterion(outputs, labels).item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    loss /= len(testloader.dataset)
    accuracy = correct / total
    return loss, accuracy


def are_models_equal(model_list: List[nn.Module]):
    # Get the state_dicts of the models in the list
    state_dicts_list = [model.state_dict() for model in model_list]

    # Check if the keys of state_dicts are the same for all models
    if any(set(state_dict.keys()) != set(state_dicts_list[0].keys()) for state_dict in state_dicts_list[1:]):
        return False

    # Check if the values (weights) of the corresponding keys are the same for all models
    for key in state_dicts_list[0].keys():
        if any(not torch.equal(state_dict[key], state_dicts_list[0][key]) for state_dict in state_dicts_list[1:]):
            return False

    # If all checks pass, the models are equal
    return True
