import torch
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import ToTensor
from torchvision.utils import make_grid
import matplotlib.pyplot as plt

class CAS771Dataset(Dataset):
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


import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
import torch.optim as optim

def remap_labels(labels, class_mapping):
    return [class_mapping[label] for label in labels]


if __name__ == '__main__':
    net = Net()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
    train_data, labels, idx = load_data('Image classifiers\data\Model3\model3_train.pth')
    classes = load_class_names('Image classifiers\data\cifar100_classes.txt')
    unique_labels = sorted(set(labels))
    class_mapping = {label: i for i, label in enumerate(unique_labels)}
    print(f"Class mapping: {class_mapping}")
    
    # Remap the labels
    remapped_labels = remap_labels(labels, class_mapping)
    my_dataset = CAS771Dataset(train_data, remapped_labels, transform=False)
    dataloader = DataLoader(my_dataset, batch_size=5, shuffle=True)
    for epoch in range(2):  # loop over the dataset multiple times
        running_loss = 0.0
        for i, data in enumerate(dataloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data
            print(labels.shape)
            print(labels)
            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % 2000 == 1999:    # print every 2000 mini-batches
                print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}')
                running_loss = 0.0

    print('Finished Training')