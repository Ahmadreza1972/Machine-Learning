import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
import torchvision.models as models
import torchvision.transforms as transforms
from tqdm import tqdm
import matplotlib.pyplot  as plt
from torchvision.utils import make_grid
import torch.nn.functional as F


# Load dataset from .pth file
def load_data(data_path):
    raw_data = torch.load(data_path)
    data = raw_data['data']
    labels = raw_data['labels']
    indices = raw_data['indices']  # Indices from the original dataset (CIFAR100)
    return data, labels, indices

def remap_labels(labels, class_mapping):
    return [class_mapping[label] for label in labels]

def create_dataloader(images, labels, batch_size=32, shuffle=True):
    #images = torch.tensor(images).float()  # Convert images to float tensor if necessary
    unique_labels = sorted(set(labels))
    class_mapping = {label: i for i, label in enumerate(unique_labels)}
    remapped_labels = remap_labels(labels, class_mapping)
    labels = torch.tensor(remapped_labels).long()  # Convert labels to long tensor (for classification)
    dataset = TensorDataset(images, labels)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return dataloader

def get_data(test_path,batch_size):
    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Normalize the images
    ])
    # Load and transform test data
    test_images, test_labels, test_indices = load_data(test_path)
    test_images_pil = [transforms.ToPILImage()(img) for img in test_images]
    test_images = torch.stack([transform(img) for img in test_images_pil])
    test_loader = create_dataloader(test_images, test_labels, batch_size=batch_size)
    return test_loader

class CustomResNet18(nn.Module):
    def __init__(self, num_classes=5, freeze_layers=False):
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

class MobileNetV2ForCIFAR8M(nn.Module):
    def __init__(self, num_classes=5, num_layers_to_keep=10):
        super(MobileNetV2ForCIFAR8M, self).__init__()
        
        # Load the pre-trained MobileNetV2 model
        self.mobilenet_v2 = models.mobilenet_v2(pretrained=True)
        
        # Keep only the first `num_layers_to_keep` layers of the feature extractor
        self.mobilenet_v2.features = nn.Sequential(
            *list(self.mobilenet_v2.features[:num_layers_to_keep])
        )
        
        # Calculate the output size after the truncated feature extractor
        sample_input = torch.randn(1, 3, 32, 32)  # Example input size for MobileNetV2
        with torch.no_grad():
            output_shape = self.mobilenet_v2.features(sample_input).shape
            print("Output shape after truncation:", output_shape)
        flattened_features = output_shape[1] * output_shape[2] * output_shape[3]
        print("Flattened feature size:", flattened_features)
        # Replace the final classifier to match the new output size and number of classes
        self.mobilenet_v2.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flattened_features, 512),  # Example: Intermediate layer size
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)  # Final layer matches the number of classes
        )

    def forward(self, x):
        x = self.mobilenet_v2.features(x)
        return self.mobilenet_v2.classifier(x)
# Define the model class again (with the same architecture)



def get_predictions_and_probabilities(model, dataloader, device='cpu'):
    model.to(device)
    predictions = []
    probabilities = []
    pr_lable=[]
    Tr_lable=[]
    
    with torch.no_grad():  # No need to calculate gradients during inference
        for inputs, labels in tqdm(dataloader, desc="Making predictions"):
            inputs = inputs.to(device)
            
            # Forward pass
            outputs = model(inputs)
            
            # Get probabilities using softmax
            probs = F.softmax(outputs, dim=1)  # Apply softmax to get probabilities
            
            # Get predicted class (index of the maximum probability)
            _, predicted_classes = torch.max(probs, 1)
            
            predictions.extend(predicted_classes.tolist())
            probabilities.extend(probs.tolist())
            pr_lable.extend(predicted_classes.tolist()) 
            Tr_lable.extend(labels.tolist()) 
    
    return predictions, probabilities,pr_lable,Tr_lable


def main():
    
    path1=r"Image classifiers\data\Model1\model1_test.pth"
    #model = MobileNetV2ForCIFAR8M()
    model=CustomResNet18()
    model.load_state_dict(torch.load(r"model_weights.pth"))
    model.eval()  # Set the model to evaluation mode
    dataloader=get_data(path1,1)
    predictions, probabilities,pr_lable,Tr_lable=get_predictions_and_probabilities(model, dataloader, device='cpu')
    print(predictions, probabilities,pr_lable,Tr_lable)
    
if __name__ == "__main__":
    main()