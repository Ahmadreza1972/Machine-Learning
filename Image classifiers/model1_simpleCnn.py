import torch
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
from torchvision import transforms
from torch.utils.data import random_split



class ClassDataset(Dataset):
    def __init__(self, data, labels, transform=None):
        self.data = data
        self.labels = labels
        self.transform = ToTensor() if transform else None

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img = self.data[idx]
        label = self.labels[idx]
        if self.transform and not isinstance(img, torch.Tensor):
            img = self.transform(img)
        return img, label
    
def load_data(data_path):
    raw_data = torch.load(data_path)
    data = raw_data['data']
    #print(data.shape)
    labels = raw_data['labels']
    # print(labels.type())
    indices = raw_data['indices'] #indice is the idx of the data in the original dataset (CIFAR100)
    return data, labels, indices

def load_class_names(filepath):
    with open(filepath, 'r') as file:
        classes = [line.strip() for line in file]
    return classes
def remap_labels(labels, class_mapping):
    return [class_mapping[label] for label in labels]

class CNNModel(nn.Module):
    def __init__(self, input_channels, num_classes, conv_layers_config, fc_layers_config, device, reg_lambda=1e-4):
        super(CNNModel, self).__init__()

        self.device = device  # Store device info
        self.num_classes = num_classes
        self.features = nn.ModuleDict()
        self.reg_lambda = reg_lambda  # Regularization strength

        # Add convolutional layers dynamically
        in_channels = input_channels
        for i, layer_config in enumerate(conv_layers_config):
            self.features[f"conv{i+1}"] = nn.Conv2d(
                in_channels=in_channels,
                out_channels=layer_config["out_channels"],
                kernel_size=layer_config["kernel_size"],
                stride=layer_config["stride"],
                padding=layer_config["padding"],
            ).to(self.device)  # Ensure conv layer is on the correct device
            self.features[f"relu{i+1}"] = nn.ReLU().to(self.device)
            if layer_config["maxpool"]>0:
                self.features[f"pool{i+1}"] = nn.MaxPool2d(kernel_size=layer_config["maxpool"], stride=2).to(self.device)
            in_channels = layer_config["out_channels"]

        # Fully connected layer configuration
        self.flatten_size = None  # Will be dynamically determined
        self.fc_layers_config = fc_layers_config
        self.fc = nn.ModuleDict()  # Fully connected layers

        # Dropout
        self.dropout = nn.Dropout(0.5).to(self.device)  # Ensure dropout is on the correct device

    def forward(self, x):
        reg_loss = 0  # Initialize regularization loss

        # Forward pass through convolutional layers
        for layer in self.features.values():
            x = layer(x)  # No need to move x here since layers are already on the right device
            
            # Accumulate L2 regularization loss for convolutional layers
            if isinstance(layer, nn.Conv2d):  # Regularization only for weight layers
                reg_loss += self.reg_lambda * torch.sum(layer.weight ** 2)

        # Flatten the feature map
        if self.flatten_size is None:
            self.flatten_size = x.view(x.size(0), -1).shape[1]
            in_features = self.flatten_size
            for i, out_features in enumerate(self.fc_layers_config):
                self.fc[f"fc{i+1}"] = nn.Linear(in_features, out_features).to(self.device)
                in_features = out_features
            self.fc["output"] = nn.Linear(in_features, self.num_classes).to(self.device)

        x = x.view(x.size(0), -1)  # Flatten

        # Forward pass through fully connected layers
        for name, layer in self.fc.items():
            x = F.relu(layer(x)) if name != "output" else layer(x)
            # Accumulate L2 regularization loss for fully connected layers
            if hasattr(layer, "weight"):
                reg_loss += self.reg_lambda * torch.sum(layer.weight ** 2)
            if "fc" in name:  # Apply dropout only on hidden layers
                x = self.dropout(x)

        return x, reg_loss





def train_model(model, train_loader, val_loader, num_epochs, lr, device):
    """
    Train CNN model with added device checks.
    """
    print(f"Training on {device}.")
    model.to(device)  # Ensure model is on the correct device

    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    train_losses = []
    val_losses = []

    for epoch in range(num_epochs):
        # Training
        model.train()
        running_loss, total, correct = 0.0, 0, 0

        for inputs, labels in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}"):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()

            # Forward pass
            outputs, reg_loss = model(inputs)
            loss = criterion(outputs, labels) + reg_loss

            # Backward pass
            loss.backward()
            optimizer.step()

            # Tracking loss and accuracy
            running_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        # Validation
        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs, _ = model(inputs)
                val_loss += criterion(outputs, labels).item()
                _, preds = torch.max(outputs, 1)
                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)

        train_losses.append(running_loss / len(train_loader))
        val_losses.append(val_loss / len(val_loader))

        print(
            f"Epoch {epoch + 1}/{num_epochs}, "
            f"Train Loss: {train_losses[-1]:.4f}, Train Acc: {correct / total:.2%}, "
            f"Val Loss: {val_losses[-1]:.4f}, Val Acc: {val_correct / val_total:.2%}"
        )

    return train_losses, val_losses


def test_model(model, dataloader, device):
    """
    Test CNN model with added device checks.
    """
    model.to(device)
    model.eval()

    total, correct, running_loss = 0, 0, 0.0
    criterion = nn.CrossEntropyLoss()

    with torch.no_grad():
        for inputs, labels in tqdm(dataloader, desc="Testing"):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs, _ = model(inputs)
            loss = criterion(outputs, labels)

            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            running_loss += loss.item()

    print(f"Test Loss: {running_loss / len(dataloader):.4f}, Test Accuracy: {correct / total:.2%}")

    
def get_total_params(model):
    total_params = sum(p.numel() for p in model.parameters())
    return total_params

def data_train_loader(validat_rate, train_path, class_path):
    train_data, labels, _ = load_data(train_path)
    classes = load_class_names(class_path)
    unique_labels = sorted(set(labels))
    class_mapping = {label: i for i, label in enumerate(unique_labels)}
    remapped_labels = remap_labels(labels, class_mapping)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2470, 0.2435, 0.2616])
    ])
    dataset = ClassDataset(train_data, remapped_labels, transform=transform)

    train_size = int((1 - validat_rate) * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    return (
        DataLoader(train_dataset, batch_size=batch_size, shuffle=True),
        DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    )

def data_test_loader(test_path, class_path):
    test_data, labels, _ = load_data(test_path)
    classes = load_class_names(class_path)
    unique_labels = sorted(set(labels))
    class_mapping = {label: i for i, label in enumerate(unique_labels)}
    remapped_labels = remap_labels(labels, class_mapping)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2470, 0.2435, 0.2616])
    ])
    dataset = ClassDataset(test_data, remapped_labels, transform=transform)

    return DataLoader(dataset, batch_size=batch_size, shuffle=False)



if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    # Your configurations
    validat_rate = 0.1
    batch_size = 16
    pic_layers = 3
    num_classes = 5
    learning_rate = 0.001
    num_epochs = 200

    train_path = 'Image classifiers\data\Model1\model1_train.pth'
    test_path = 'Image classifiers\data\Model1\model1_test.pth'
    class_path = 'Image classifiers\data\cifar100_classes.txt'

    conv_config = [
        {"out_channels": 64, "kernel_size": 3, "stride": 1, "padding": 1, "maxpool": 0},
        {"out_channels": 128, "kernel_size": 3, "stride": 1, "padding": 1, "maxpool": 2},
        {"out_channels": 256, "kernel_size": 3, "stride": 1, "padding": 1, "maxpool": 2},
        {"out_channels": 512, "kernel_size": 3, "stride": 1, "padding": 1, "maxpool": 2}
    ]
    fc_config = [ 512, 128,32]

    # Load data
    train_loader, val_loader = data_train_loader(validat_rate, train_path, class_path)
    test_loader = data_test_loader(test_path, class_path)

    # Initialize model
    model = CNNModel(
        input_channels=pic_layers,
        num_classes=num_classes,
        conv_layers_config=conv_config,
        fc_layers_config=fc_config,
        reg_lambda=0.001,
        device=device
    )

    # Train and test model
    train_losses, val_losses = train_model(model, train_loader, val_loader, num_epochs, learning_rate, device)
    test_model(model, test_loader, device)

    # Plot loss curves
    plt.plot(train_losses, label="Train")
    plt.plot(val_losses, label="Validation")
    plt.legend()
    plt.show()
    
    
  