import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
import torchvision.models as models
import torchvision.transforms as transforms
from tqdm import tqdm
import matplotlib.pyplot as plt

# Load dataset from .pth file
def load_data(data_path):
    raw_data = torch.load(data_path)
    data = raw_data['data']
    labels = raw_data['labels']
    indices = raw_data['indices']  # Indices from the original dataset (CIFAR100)
    return data, labels, indices

def remap_labels(labels, class_mapping):
    return [class_mapping[label] for label in labels]

# Prepare DataLoader with transform
def create_dataloader(images, labels, batch_size=32, shuffle=True):
    #images = torch.tensor(images).float()  # Convert images to float tensor if necessary
    unique_labels = sorted(set(labels))
    class_mapping = {label: i for i, label in enumerate(unique_labels)}
    remapped_labels = remap_labels(labels, class_mapping)
    labels = torch.tensor(remapped_labels).long()  # Convert labels to long tensor (for classification)
    dataset = TensorDataset(images, labels)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return dataloader

# Define ResNet18 model with control over trainable layers
class CustomResNet18(nn.Module):
    def __init__(self, num_classes, freeze_layers=False):
        super(CustomResNet18, self).__init__()
        self.resnet18 = models.resnet18(weights='IMAGENET1K_V1')
        
        # Adjust the first convolutional layer for 32x32 input
        self.resnet18.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.resnet18.maxpool = nn.Identity()  # Remove maxpool for smaller images
        self.resnet18.layer4 = nn.Identity()
        
        # Freeze layers if specified
        if freeze_layers:
            for param in self.resnet18.parameters():
                param.requires_grad = False
        
        in_features = 256
        self.resnet18.fc = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.resnet18(x)

# Training function
def train_model(model, train_loader,val_loader, criterion, optimizer, num_epochs=10, device='cpu'):
    model.to(device)
    
    tr_los=[]
    tr_ac=[]
    val_los=[]
    val_ac=[]
    for epoch in range(num_epochs):
        model.train()
        correct = 0
        total = 0
        running_loss = 0.0
        for inputs, labels in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}"):
            inputs, labels = inputs.to(device), labels.to(device)

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            # Backward pass and optimization
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        accuracy = 100 * correct / total
        tr_los.append(running_loss/len(train_loader))
        tr_ac.append(accuracy)
        
        
        model.eval()  # Set model to evaluation mode
        val_correct = 0
        val_total = 0
        val_running_loss = 0.0

        with torch.no_grad():  # No gradients needed for validation
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)

                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_running_loss += loss.item()

                _, predicted = torch.max(outputs, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()

        val_accuracy = 100 * val_correct / val_total
        val_los.append(val_running_loss/len(val_loader))
        val_ac.append(val_accuracy)
        
        
        
        print(f"                    Tr_Loss: {running_loss/len(train_loader):.4f}, val_loss: {val_running_loss/len(val_loader):.4f}, Tr_acc: {accuracy}, val_ac: {val_accuracy}")
    return tr_ac,val_ac,tr_los,val_los
# Test function
def test_model(model, test_loader, criterion, device='cpu'):
    model.to(device)
    model.eval()  # Set model to evaluation mode
    correct = 0
    total = 0
    running_loss = 0.0

    with torch.no_grad():  # No gradients needed for testing
        for inputs, labels in tqdm(test_loader, desc="Testing"):
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            running_loss += loss.item()

            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    print(f"Test Loss: {running_loss/len(test_loader):.4f}")
    print(f"Test Accuracy: {accuracy:.2f}%")

# Main script
def main():
    # Check for CUDA (GPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"-------Using device: {device}-------------")

    batch_size = 16
    learning_rate = 0.001
    epoch = 35
    valdata_ratio=0.1
    train_path = r"Image classifiers\data\Model1\model1_train.pth"  # Raw string for Windows paths
    test_path = r"Image classifiers\data\Model1\model1_test.pth"

    # Load data
    images, labels, indices = load_data(train_path)

    # Convert tensor to PIL Image for transformation
    images_pil = [transforms.ToPILImage()(img) for img in images]

    # Transformations for input images
    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Normalize the images
    ])

    images = torch.stack([transform(img) for img in images_pil])  # Apply transformation to each image
    train_size = int((1-valdata_ratio) * len(images))
    val_size = len(images) - train_size
    train_dataset, val_dataset = random_split(images, [train_size, val_size])
    # Create DataLoader
    train_images = images[train_dataset.indices]
    train_labels = [labels[i] for i in train_dataset.indices]

    train_loader = create_dataloader(train_images, train_labels, batch_size=batch_size)
    
    val_images = images[train_dataset.indices]
    val_labels = [labels[i] for i in train_dataset.indices]    
    val_loader = create_dataloader(val_images, val_labels, batch_size=batch_size)
    # Initialize model
    num_classes = len(torch.unique(torch.tensor(labels)))  # Number of unique classes
    model = CustomResNet18(num_classes=num_classes, freeze_layers=False)
    print(model)

    # Define loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Number of trainable parameters after freezing layers: {trainable_params}")
    # Train model
    tr_ac,val_ac,tr_los,val_los=train_model(model, train_loader,val_loader, criterion, optimizer, num_epochs=epoch, device=device)
    torch.save(model.state_dict(), "model_weights.pth")
    fig,ax=plt.subplots(ncols=2)
    ax[0].plot(tr_ac, label="Train_ac")
    ax[0].plot(val_ac, label="val_ac")
    ax[0].legend()
    ax[1].plot(tr_los, label="Train_los")
    ax[1].plot(val_los, label="val_los")
    ax[1].legend()
    plt.show()

    # Load and transform test data
    test_images, test_labels, test_indices = load_data(test_path)
    test_images_pil = [transforms.ToPILImage()(img) for img in test_images]
    test_images = torch.stack([transform(img) for img in test_images_pil])
    test_loader = create_dataloader(test_images, test_labels, batch_size=batch_size)

    # Test the model
    test_model(model, test_loader, criterion, device=device)

if __name__ == "__main__":
    main()
