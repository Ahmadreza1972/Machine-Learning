import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
import torchvision.models as models
import torchvision.transforms as transforms
from tqdm import tqdm
import matplotlib.pyplot  as plt
from torchvision.utils import make_grid

# Load dataset from .pth file
def load_data(data_path):
    raw_data = torch.load(data_path)
    data = raw_data['data']
    labels = raw_data['labels']
    indices = raw_data['indices']  # Indices from the original dataset (CIFAR100)
    return data, labels, indices

def remap_labels(labels, class_mapping):
    return [class_mapping[label] for label in labels]

def load_class_names(filepath):
    with open(filepath, 'r') as file:
        classes = [line.strip() for line in file]
    return classes

def pic(test_unique_labels, classes, test_loader, predicted, lal):
    i = 0
    for inputs, labels in test_loader:
        # Get true labels for the current batch
        true_classes = [classes[test_unique_labels[label.item()]] for label in lal[i]]

        # Get predicted labels for the current batch
        my_predict = [classes[test_unique_labels[pred.item()]] for pred in predicted[i]]  # Assuming 'predicted' contains your predictions

        for im, true_label, pred_label in zip(inputs, true_classes, my_predict):
            # If 'inputs' is a batch, `im` will already be a single image. If it's a tensor of shape [C, H, W], don't add a batch dimension
            grid_img = make_grid(im.unsqueeze(0), nrow=1)  # Add batch dimension for grid
            plt.imshow(grid_img.permute(1, 2, 0))  # Reorder dimensions from [C, H, W] to [H, W, C]
            plt.title(f"Predicted: {pred_label}, True: {true_label}")
            plt.axis('off')  # Turn off axis labels for clarity
            plt.show()
        
        i += 1


# Prepare DataLoader with transform
def create_dataloader(images, labels, batch_size=32, shuffle=True):
    #images = torch.tensor(images).float()  # Convert images to float tensor if necessary
    unique_labels = sorted(set(labels))
    class_mapping = {label: i for i, label in enumerate(unique_labels)}
    remapped_labels = remap_labels(labels, class_mapping)
    labels = torch.tensor(remapped_labels).long()  # Convert labels to long tensor (for classification)
    dataset = TensorDataset(images, labels)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return dataloader,unique_labels

# Define ResNet18 model with control over trainable layers
class MobileNetV2ForCIFAR8M(nn.Module):
    def __init__(self, num_classes=10):
        super(MobileNetV2ForCIFAR8M, self).__init__()
        
        # Load the pre-trained MobileNetV2 model
        self.mobilenet_v2 = models.mobilenet_v2(pretrained=True)

        # Replace the final classifier layer to match the number of classes
        self.mobilenet_v2.classifier[1] = nn.Linear(self.mobilenet_v2.last_channel, num_classes)

    def forward(self, x):
        return self.mobilenet_v2(x)

# Training function
def train_model(model, train_loader,val_loader, criterion, optimizer, num_epochs=10, device='cpu'):
    model.to(device)
    tr_ac=[]
    val_ac=[]
    tr_los=[]
    val_los=[]
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
        tr_ac.append(accuracy)
        tr_los.append(running_loss/len(train_loader))
        
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
        val_ac.append(val_accuracy)
        val_los.append(val_running_loss/len(val_loader))
        
        
        print(f"Epoch {epoch+1}/{num_epochs}, Tr_Loss: {running_loss/len(train_loader):.4f}, val_loss: {val_running_loss/len(val_loader):.4f}, Tr_acc: {accuracy}, val_ac: {val_accuracy}")
    return tr_ac,val_ac,tr_los,val_los
# Test function
def test_model(model, test_loader, criterion, device='cpu'):
    model.to(device)
    model.eval()  # Set model to evaluation mode
    correct = 0
    total = 0
    running_loss = 0.0
    tot_predict=[]
    lal=[]
    with torch.no_grad():  # No gradients needed for testing
        for inputs, labels in tqdm(test_loader, desc="Testing"):
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            running_loss += loss.item()

            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            tot_predict.append(predicted)
            lal.append(labels)
    accuracy = 100 * correct / total
    print(f"Test Loss: {running_loss/len(test_loader):.4f}")
    print(f"Test Accuracy: {accuracy:.2f}%")
    return tot_predict,lal

# Main script
def main():
    # Check for CUDA (GPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"-------Using device: {device}-------------")
    validation_size=0.2
    batch_size = 16
    learning_rate = 0.001
    epoch = 1
    train_path = r"Image classifiers\data\Model3\model3_train.pth"  # Raw string for Windows paths
    test_path = r"Image classifiers\data\Model3\model3_test.pth"
    classes = load_class_names('Image classifiers\data\cifar100_classes.txt')
    # Load data
    images, labels, indices = load_data(train_path)

    # Convert tensor to PIL Image for transformation
    images_pil = [transforms.ToPILImage()(img) for img in images]

    # Transformations for input images
    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        #transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Normalize the images
    ])

    images = torch.stack([transform(img) for img in images_pil])  # Apply transformation to each image
    train_size = int((1-validation_size) * len(images))
    val_size = len(images) - train_size
    

    train_dataset, val_dataset = random_split(images, [train_size, val_size])
    # Create DataLoader
    train_images = images[train_dataset.indices]
    train_labels = [labels[i]  for i in train_dataset.indices]
    
    

    train_loader,train_unique_labels = create_dataloader(train_images, train_labels, batch_size=batch_size)
    
    val_images = images[val_dataset.indices]
    val_labels = [labels[i] for i in val_dataset.indices]    
    val_loader,val_unique_labels = create_dataloader(val_images, val_labels, batch_size=batch_size)
    
    
    # Initialize model
    num_classes = len(torch.unique(torch.tensor(labels)))  # Number of unique classes
    model = MobileNetV2ForCIFAR8M(num_classes=num_classes)

    # Define loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Number of trainable parameters after freezing layers: {trainable_params}")
    # Train model
    tr_ac,val_ac,tr_los,val_los=train_model(model, train_loader,val_loader, criterion, optimizer, num_epochs=epoch, device=device)

    # Load and transform test data
    test_images, test_labels, test_indices = load_data(test_path)
    test_images_pil = [transforms.ToPILImage()(img) for img in test_images]
    test_images = torch.stack([transform(img) for img in test_images_pil])
    test_loader,test_unique_labels = create_dataloader(test_images, test_labels, batch_size=batch_size)

    # Test the model
    predicted,lal=test_model(model, test_loader, criterion, device=device)

    #pic(test_unique_labels,classes,test_loader,predicted,lal)
        # Loop through each image in the batch

        
    
    

    
    
    fig,ax=plt.subplots(ncols=2)
    ax[0].plot(tr_ac, label="Train_ac")
    ax[0].plot(val_ac, label="val_ac")
    ax[0].legend()
    ax[1].plot(tr_los, label="Train_los")
    ax[1].plot(val_los, label="val_los")
    ax[1].legend()
    plt.show()

if __name__ == "__main__":
    main()
