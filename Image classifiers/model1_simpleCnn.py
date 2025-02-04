import torch
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import ToTensor
from torchvision.utils import make_grid
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
from torchvision import transforms


print(torch.__version__)
class ClassDataset(Dataset):
    def __init__(self, data, labels, transform=False):
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
    def __init__(self, input_channels=3, num_classes=10, conv_layers_config=None, fc_layers_config=None):
        """
        A dynamic CNN model using ModuleDict.
        Args:
            input_channels (int): Number of input channels (e.g., 3 for RGB images).
            num_classes (int): Number of output classes for classification.
            conv_layers_config (list of dict): Configuration for convolutional layers.
            fc_layers_config (list of int): Configuration for fully connected layers.
        """
        super(CNNModel, self).__init__()

        self.num_classes = num_classes
        self.features = nn.ModuleDict()

        # Default convolutional layers config if none provided
        if conv_layers_config is None:
            conv_layers_config = [
                {"out_channels": 32, "kernel_size": 3, "stride": 1, "padding": 1},
                {"out_channels": 64, "kernel_size": 3, "stride": 1, "padding": 1},
            ]

        # Add convolutional layers dynamically
        in_channels = input_channels
        for i, layer_config in enumerate(conv_layers_config):
            self.features[f"conv{i+1}"] = nn.Conv2d(
                in_channels=in_channels,
                out_channels=layer_config["out_channels"],
                kernel_size=layer_config["kernel_size"],
                stride=layer_config["stride"],
                padding=layer_config["padding"],
            )
            self.features[f"relu{i+1}"] = nn.ReLU()
            self.features[f"pool{i+1}"] = nn.MaxPool2d(kernel_size=2, stride=2)
            in_channels = layer_config["out_channels"]

        # Fully connected layer configuration
        self.flatten_size = None  # Will be dynamically determined
        self.fc_layers_config = fc_layers_config or [128]
        self.fc = nn.ModuleDict()  # Fully connected layers

        # Dropout
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        # Forward pass through convolutional layers
        for layer in self.features.values():
            x = layer(x)

        # Flatten the feature map
        if self.flatten_size is None:
            self.flatten_size = x.view(x.size(0), -1).shape[1]
            in_features = self.flatten_size
            for i, out_features in enumerate(self.fc_layers_config):
                self.fc[f"fc{i+1}"] = nn.Linear(in_features, out_features)
                in_features = out_features
            self.fc["output"] = nn.Linear(in_features, self.num_classes)

        x = x.view(x.size(0), -1)  # Flatten

        # Forward pass through fully connected layers
        for name, layer in self.fc.items():
            x = F.relu(layer(x)) if name != "output" else layer(x)
            if "fc" in name:  # Apply dropout only on hidden layers
                x = self.dropout(x)

        return x




def train_model(model, dataloader, num_epochs=100, lr=0.001, device="cuda" if torch.cuda.is_available() else "cpu"):
    """
    Trains a given CNN model using the provided DataLoader.

    Args:
        model (nn.Module): The CNN model to train.
        dataloader (DataLoader): DataLoader providing training data.
        num_epochs (int): Number of training epochs.
        lr (float): Learning rate.
        device (str): Device to train on ("cuda" or "cpu").
    """
    print(f"GPU is available:{torch.cuda.is_available()}")
    # Move model to device
    model.to(device)

    # Define optimizer and loss function
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    # Training loop
    for epoch in range(num_epochs):
        model.train()  # Set model to training mode
        running_loss = 0.0

        # Iterate over batches
        for inputs, labels in tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            inputs, labels = inputs.to(device), labels.to(device)

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            # Backward pass and optimization
            loss.backward()
            optimizer.step()

            # Accumulate loss
            running_loss += loss.item()

        # Print epoch loss
        epoch_loss = running_loss / len(dataloader)
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}")
    print("Training complete!")


def test_model(model, dataloader, device="cuda" if torch.cuda.is_available() else "cpu"):
    """
    Tests the given CNN model using the provided DataLoader.

    Args:
        model (nn.Module): The CNN model to test.
        dataloader (DataLoader): DataLoader providing test data.
        device (str): Device to test on ("cuda" or "cpu").
    """
    # Move model to device
    model.to(device)
    model.eval()  # Set model to evaluation mode

    correct = 0
    total = 0
    running_loss = 0.0
    criterion = nn.CrossEntropyLoss()

    with torch.no_grad():  # No need to calculate gradients during testing
        for inputs, labels in tqdm(dataloader, desc="Testing"):
            inputs, labels = inputs.to(device), labels.to(device)

            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            # Calculate accuracy
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            # Accumulate loss
            running_loss += loss.item()

    # Calculate average loss and accuracy
    test_loss = running_loss / len(dataloader)
    accuracy = 100 * correct / total

    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {accuracy:.2f}%")    
    
def get_total_params(model):
    total_params = sum(p.numel() for p in model.parameters())
    return total_params

def data_train_loader():
    train_data, labels, idx = load_data('Image classifiers\data\Model1\model1_train.pth')
    classes = load_class_names('Image classifiers\data\cifar100_classes.txt')
    unique_labels = sorted(set(labels))
    class_mapping = {label: i for i, label in enumerate(unique_labels)}
    remapped_labels = remap_labels(labels, class_mapping)
    # Define data augmentation transformations
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),  # Randomly flip the image horizontally
        transforms.RandomCrop(32, padding=4),  # Randomly crop and pad the image
        transforms.ToTensor(),  # Convert the image to a tensor
        transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2470, 0.2435, 0.2616])  # Normalize with CIFAR-100 mean and std
    ])
    
    my_dataset = ClassDataset(train_data, remapped_labels, transform=train_transform)
    dataloader = DataLoader(my_dataset, batch_size=batch_size, shuffle=True)
    return dataloader

def data_test_loader():
    train_data, labels, idx = load_data('Image classifiers\data\Model1\model1_test.pth')
    classes = load_class_names('Image classifiers\data\cifar100_classes.txt')
    unique_labels = sorted(set(labels))
    class_mapping = {label: i for i, label in enumerate(unique_labels)}
    remapped_labels = remap_labels(labels, class_mapping)
    my_dataset = ClassDataset(train_data, remapped_labels, transform=False)
    dataloader = DataLoader(my_dataset, batch_size=batch_size, shuffle=True)
    return dataloader



if __name__ == '__main__':
    batch_size=64
    train_loader=data_train_loader()
    test_loader=data_test_loader()
    
    conv_config = [
        {"out_channels": 16, "kernel_size": 3, "stride": 1, "padding": 1},
        {"out_channels": 32, "kernel_size": 3, "stride": 1, "padding": 1},
        {"out_channels": 64, "kernel_size": 3, "stride": 1, "padding": 1},
        {"out_channels": 128, "kernel_size": 3, "stride": 1, "padding": 1},
        {"out_channels": 128, "kernel_size": 3, "stride": 1, "padding": 1}
    ]
    
    fc_config = [512, 256, 128]
    
    
    
    model = CNNModel(input_channels=3, num_classes=5, conv_layers_config=conv_config, fc_layers_config=fc_config)
    print(model)
    total_params = get_total_params(model)
    print(f"Total number of parameters: {total_params}")
    
    train_model(model, train_loader, num_epochs=50, lr=0.001)
    test_model(model, test_loader)
    
  