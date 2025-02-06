import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import torchvision.models as models
import torchvision.transforms as transforms
from tqdm import tqdm

# Load dataset from .pth file
def load_data(data_path):
    raw_data = torch.load(data_path)
    data = raw_data['data']
    #print(data.shape)
    labels = raw_data['labels']
    # print(labels.type())
    indices = raw_data['indices'] #indice is the idx of the data in the original dataset (CIFAR100)
    return data, labels, indices

def remap_labels(labels, class_mapping):
    return [class_mapping[label] for label in labels]

# Prepare DataLoader with transform
def create_dataloader(images, labels, batch_size=32, shuffle=True):
    # Ensure the images and labels are tensors before creating the dataset
    images = torch.tensor(images).float()  # Convert images to float tensor if necessary
    unique_labels = sorted(set(labels))
    class_mapping = {label: i for i, label in enumerate(unique_labels)}
    remapped_labels = remap_labels(labels, class_mapping)
    labels=torch.tensor(remapped_labels)
    labels = torch.tensor(labels).long()  # Convert labels to long tensor (for classification)
    dataset = TensorDataset(images, labels)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return dataloader

# Define ResNet18 model with control over trainable layers
class CustomResNet18(nn.Module):
    def __init__(self, num_classes, freeze_layers=False):
        super(CustomResNet18, self).__init__()
        # Load pretrained ResNet18
        self.resnet18 = models.resnet18(pretrained=True)
        
        # Adjust the first convolutional layer for 32x32 input
        self.resnet18.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.resnet18.maxpool = nn.Identity()  # Remove maxpool for smaller images
        
        # Freeze layers if specified
        if freeze_layers:
            for param in self.resnet18.parameters():
                param.requires_grad = False
        
        # Replace the final fully connected layer
        in_features = self.resnet18.fc.in_features
        self.resnet18.fc = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.resnet18(x)

# Training function
def train_model(model, train_loader, criterion, optimizer, num_epochs=10, device='cpu'):
    model.to(device)
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
        print(f"Epoch {epoch+1}/{num_epochs}, Test_Loss: {running_loss/len(train_loader):.4f}, Test_acc={accuracy}")

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
    
    batch_size=16
    learning_rate=0.001
    epoch=3
    train_path = r"Image classifiers\data\Model1\model1_train.pth"  # Raw string for Windows paths
    test_path= r"Image classifiers\data\Model1\model1_test.pth"
    # Load data
    
    images, labels, indices = load_data(train_path)

    # Convert tensor to PIL Image for transformation
    # Loop through each image to convert and apply normalization
    images_pil = []
    for img in images:
        img = transforms.ToPILImage()(img)  # Convert to PIL Image
        images_pil.append(img)

    # Transformations and normalization for input images
    transform = transforms.Compose([
        transforms.Resize((32, 32)),  # Resize if needed (e.g., 32x32)
        transforms.ToTensor(),  # Converts images to tensor format
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Normalize the images
    ])

    images = torch.stack([transform(img) for img in images_pil])  # Apply transformation to each image
    
    # Create DataLoader
    train_loader = create_dataloader(images, labels, batch_size=batch_size)

    # Initialize model
    num_classes = len(torch.unique(torch.tensor(labels)))  # Number of unique classes
    model = CustomResNet18(num_classes=num_classes, freeze_layers=False)

    # Define loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Train model
    train_model(model, train_loader, criterion, optimizer, num_epochs=epoch)
    
    
    test_images, test_labels, test_indices = load_data(test_path)
    test_images_pil = []
    for img in test_images:
        img = transforms.ToPILImage()(img)  # Convert to PIL Image
        test_images_pil.append(img)
    test_images = torch.stack([transform(img) for img in test_images_pil])    
    test_loader = create_dataloader(test_images, test_labels, batch_size=batch_size)
    num_classes = len(torch.unique(torch.tensor(test_labels)))  # Number of unique classes
    test_model(model, test_loader, criterion)
        
if __name__ == "__main__":
    main()
