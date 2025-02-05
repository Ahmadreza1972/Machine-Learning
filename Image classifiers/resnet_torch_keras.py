from tensorflow.keras.applications import ResNet50V2
from tensorflow.keras.applications.resnet_v2 import preprocess_input
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow as tf
from torch.utils.data import DataLoader, Dataset
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms
from torch import nn, optim
from torch.nn import functional as F
from torch.utils.data import random_split
from tqdm import tqdm
import matplotlib.pyplot as plt
import time


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

img_size = (32, 32)

base_model = ResNet50V2(input_shape=(*img_size, 3),
                         include_top=False,
                         weights='imagenet')
base_model.trainable = True
num_layer = len(base_model.layers)
print(f"number of layers in ResNet50V2: {num_layer}")

num_layer_fine_tune = 10
for layer in base_model.layers[:-num_layer_fine_tune]:
  layer.trainable = False
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


# Data Loader Functions
def data_train_loader(validat_rate, batch_size):
    # Load data
    train_data, labels, _ = load_data('Image classifiers/data/Model1/model1_train.pth')

    # Extract unique labels and create class mapping
    unique_labels = sorted(set(labels))
    class_mapping = {label: i for i, label in enumerate(unique_labels)}
    remapped_labels = remap_labels(labels, class_mapping)

    # Transformations
    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2470, 0.2435, 0.2616]),
    ])

    # Create dataset and split into train and validation
    dataset = ClassDataset(train_data, remapped_labels, transform=transform)
    train_size = int((1 - validat_rate) * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    # Data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Return train_loader, val_loader, and class names
    class_names = [str(label) for label in unique_labels]
    return train_loader, val_loader, class_names


def build_model(input_shape, num_classes):
  model = tf.keras.Sequential([
      layers.Input(shape=input_shape),
      base_model,
      layers.GlobalAveragePooling2D(),
      layers.Dense(32, activation='relu'),
      layers.Dropout(0.5),
      layers.Dense(num_classes, activation='softmax')
  ])
  model.compile(optimizer=tf.optimizers.Adam(learning_rate=0.001),
                 loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                 metrics=['accuracy'])
  return model

train_loader, val_loader, class_names=data_train_loader(0.2, 32)


def pytorch_to_tf_dataset(data_loader):
    # Get a single batch to infer shape and dtype
    sample_images, sample_labels = next(iter(data_loader))
    
    def generator():
        for images, labels in data_loader:
            # Convert PyTorch tensors to NumPy arrays and permute dimensions
            images = images.permute(0, 2, 3, 1).numpy()  # Convert (B, C, H, W) -> (B, H, W, C)
            labels = labels.numpy()
            yield images, labels
    
    return tf.data.Dataset.from_generator(
        generator,
        output_signature=(
            tf.TensorSpec(shape=(None, *sample_images.permute(0, 2, 3, 1).shape[1:]), dtype=tf.float32),
            tf.TensorSpec(shape=(None,), dtype=tf.int64)
        )
    )

# Convert PyTorch DataLoader to TensorFlow Dataset
tf_train_loader = pytorch_to_tf_dataset(train_loader)
tf_val_loader = pytorch_to_tf_dataset(val_loader)

# Prefetch data for better performance
tf_train_loader = tf_train_loader.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
tf_val_loader = tf_val_loader.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

model_tl_resnet50v2finetune = build_model(input_shape=(*img_size, 3), num_classes=len(class_names))
model_tl_resnet50v2finetune.summary()
early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)
reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3)
start = time.time()
history = model_tl_resnet50v2finetune.fit(
        tf_train_loader,
        epochs=50,
        validation_data=tf_val_loader,
        callbacks=[early_stop, reduce_lr]
)
stop = time.time()