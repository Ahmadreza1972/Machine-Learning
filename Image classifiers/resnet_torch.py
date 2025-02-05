import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms
from torch import nn, optim
from torch.nn import functional as F
from torch.utils.data import random_split
from tqdm import tqdm
import matplotlib.pyplot as plt

# Custom Dataset Class
class ClassDataset(Dataset):
    def __init__(self, data, labels, transform=None):
        self.data = data
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img = self.data[idx]
        label = self.labels[idx]
        img = transforms.ToPILImage()(img)
        if self.transform:
            img = self.transform(img)
        return img, label

# Load data function
def load_data(data_path):
    raw_data = torch.load(data_path)
    data = raw_data['data']
    labels = raw_data['labels']
    indices = raw_data['indices']
    return data, labels, indices

def load_class_names(filepath):
    with open(filepath, 'r') as file:
        classes = [line.strip() for line in file]
    return classes

def remap_labels(labels, class_mapping):
    return [class_mapping[label] for label in labels]



# ResNet Model Wrapper
import torch
import torch.nn as nn
from torchvision import models

import torch
import torch.nn as nn
from torchvision import models

import torch
import torch.nn as nn
from torchvision import models

import torch
import torch.nn as nn
from torchvision import models

from prettytable import PrettyTable

class DynamicResNet(nn.Module):
    def __init__(self, resnet_type, num_classes, pretrained=True):
        """
        Wrapper for dynamic ResNet usage.
        Args:
            resnet_type (str): The type of ResNet ('resnet18', 'resnet34', etc.).
            num_classes (int): Number of output classes for classification.
            pretrained (bool): Whether to use pretrained weights.
        """
        super(DynamicResNet, self).__init__()
        self.resnet =models.resnet18(pretrained=True)

        # Modify the final fully connected layer to match the number of classes
        in_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.resnet(x)


# Training function
def train_model(model, dataloader, val_loader, num_epochs, lr, device):
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    train_losses, val_losses = [], []

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0

        for inputs, labels in tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        epoch_loss = running_loss / len(dataloader)
        train_losses.append(epoch_loss)

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()

        val_loss /= len(val_loader)
        val_losses.append(val_loss)

        print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {epoch_loss:.4f}, Val Loss: {val_loss:.4f}")

    return train_losses, val_losses

# Testing function
def test_model(model, dataloader, device):
    model.to(device)
    model.eval()
    correct, total = 0, 0

    with torch.no_grad():
        for inputs, labels in tqdm(dataloader, desc="Testing"):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    print(f"Test Accuracy: {accuracy:.2f}%")
    return accuracy

# Data Loader Functions
def data_train_loader(validat_rate, batch_size):
    train_data, labels, _ = load_data('Image classifiers/data/Model1/model1_train.pth')
    unique_labels = sorted(set(labels))
    class_mapping = {label: i for i, label in enumerate(unique_labels)}
    remapped_labels = remap_labels(labels, class_mapping)

    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2470, 0.2435, 0.2616]),
    ])

    dataset = ClassDataset(train_data, remapped_labels, transform=transform)
    train_size = int((1 - validat_rate) * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader

def data_test_loader(batch_size):
    test_data, labels, _ = load_data('Image classifiers/data/Model1/model1_test.pth')
    unique_labels = sorted(set(labels))
    class_mapping = {label: i for i, label in enumerate(unique_labels)}
    remapped_labels = remap_labels(labels, class_mapping)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2470, 0.2435, 0.2616]),
    ])

    dataset = ClassDataset(test_data, remapped_labels, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    return dataloader

def get_total_params(model):
    table = PrettyTable(["Modules", "Parameters"])
    total_params = 0
    for name, parameter in model.named_parameters():
        if "layer1" in name or "layer2" in name or "layer3" in name or "layer4" in name:
            parameter.requires_grad = False
    
    for name, parameter in model.named_parameters():        
        if not parameter.requires_grad:
            continue
        params = parameter.numel()
        table.add_row([name, params])
        total_params += params
    print(table)
    print(f"Total Trainable Params: {total_params}")
    #return total_params, trainable_params

# Main script
if __name__ == '__main__':
    validat_rate = 0.2
    batch_size = 32
    num_classes = 5
    learning_rate = 0.0005
    num_epochs = 20
    resnet_type = "resnet50"  # resnet18 resnet34, resnet50, etc.
    device = "cuda" if torch.cuda.is_available() else "cpu"

    train_loader, val_loader = data_train_loader(validat_rate, batch_size)
    test_loader = data_test_loader(batch_size)

    model = DynamicResNet(resnet_type=resnet_type, num_classes=num_classes,pretrained=True)
    #total_params, trainable_params = get_total_params(model)
    get_total_params(model)
    #print(f"{total_params}-{trainable_params}")

    train_losses, val_losses = train_model(model, train_loader, val_loader, num_epochs, learning_rate, device)
    test_model(model, test_loader, device)

    plt.plot(train_losses, label="Train")
    plt.plot(val_losses, label="Validation")
    plt.legend()
    plt.show()
