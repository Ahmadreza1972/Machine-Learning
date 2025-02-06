import tensorflow as tf
from tensorflow.keras import layers, callbacks
from tensorflow.keras.applications import ResNet50V2
from torch.utils.data import DataLoader, Dataset, random_split
import torch
from torchvision import transforms
import time
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.metrics import classification_report

# --------------------- Dataset Class ---------------------

class ClassDataset(Dataset):
    """PyTorch Dataset class for loading and processing images and labels"""
    def __init__(self, data, labels, transform=None):
        self.data = data
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img = self.data[idx]
        label = self.labels[idx]
        img = transforms.ToPILImage()(img)  # Convert tensor to PIL Image
        if self.transform:
            img = self.transform(img)  # Apply transformations
        return img, label


class DataLoaderHelper:
    """Helper class for loading and preprocessing data"""
    @staticmethod
    def load_data(data_path):
        """Load training data, labels, and indices from a .pth file"""
        raw_data = torch.load(data_path)
        data = raw_data['data']
        labels = raw_data['labels']
        indices = raw_data['indices']
        return data, labels, indices

    @staticmethod
    def remap_labels(labels, class_mapping):
        """Remap the original labels to class indices"""
        return [class_mapping[label] for label in labels]

    @staticmethod
    def data_train_loader(validation_rate, batch_size, data_path):
        """Load and preprocess training data, then split into train and validation sets"""
        # Load data
        train_data, labels, _ = DataLoaderHelper.load_data(data_path)

        # Extract unique labels and create class mapping
        unique_labels = sorted(set(labels))
        class_mapping = {label: i for i, label in enumerate(unique_labels)}
        remapped_labels = DataLoaderHelper.remap_labels(labels, class_mapping)

        # Define transformations
        transform = transforms.Compose([
            #transforms.RandomHorizontalFlip(),
            #transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2470, 0.2435, 0.2616]),
        ])

        # Create dataset and split into train and validation sets
        dataset = ClassDataset(train_data, remapped_labels, transform=transform)
        train_size = int((1 - validation_rate) * len(dataset))
        val_size = len(dataset) - train_size
        train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

        # Data loaders
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

        class_names = [str(label) for label in unique_labels]
        return train_loader, val_loader, class_names
    
    @staticmethod
    def data_test_loader(batch_size, data_path):
        """Load and preprocess training data, then split into train and validation sets"""
        # Load data
        test_data, labels, _ = DataLoaderHelper.load_data(data_path)

        # Extract unique labels and create class mapping
        unique_labels = sorted(set(labels))
        class_mapping = {label: i for i, label in enumerate(unique_labels)}
        remapped_labels = DataLoaderHelper.remap_labels(labels, class_mapping)

        # Define transformations
        transform = transforms.Compose([
            #transforms.RandomHorizontalFlip(),
            #transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2470, 0.2435, 0.2616]),
        ])

        # Create dataset and split into train and validation sets
        test_dataset = ClassDataset(test_data, remapped_labels, transform=transform)


        # Data loaders
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

        class_names = [str(label) for label in unique_labels]
        return test_loader, class_names


# --------------------- Model Setup ---------------------

class ResNetModel:
    """Helper class for building and compiling the model"""
    @staticmethod
    def create_base_model(input_shape):
        """Load the ResNet50V2 model with weights from ImageNet"""
        base_model = ResNet50V2(input_shape=input_shape, include_top=False, weights='imagenet')
        base_model.trainable = True

        # Freeze all layers except the last few
        num_layer = len(base_model.layers)
        num_layer_fine_tune = 3
        for layer in base_model.layers[:-num_layer_fine_tune]:
            layer.trainable = False

        return base_model

    @staticmethod
    def build_model(input_shape, num_classes, base_model):
        """Build and compile the model by adding custom layers on top of the base model"""
        model = tf.keras.Sequential([
            layers.Input(shape=input_shape),
            base_model,
            layers.GlobalAveragePooling2D(),
            layers.Dense(64, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001)),
            layers.Dropout(0.5),
            layers.Dense(32, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001)),
            layers.Dropout(0.5),
            layers.Dense(num_classes, activation='softmax')
        ])
        lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=0.01,
        decay_steps=100000,
        decay_rate=0.96,
        staircase=True
        )
        model.compile(optimizer=tf.optimizers.Adam(learning_rate=0.01),
                      loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                      metrics=['accuracy'])
        return model


# --------------------- PyTorch to TensorFlow Data Conversion ---------------------

class DataConverter:
    """Helper class to convert PyTorch DataLoader to TensorFlow Dataset"""
    @staticmethod
    def pytorch_to_tf_dataset(data_loader):
        """Convert PyTorch DataLoader to TensorFlow Dataset"""
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


# --------------------- Training ---------------------

class ModelTrainer:
    """Helper class for training the model"""
    @staticmethod
    def prepare_datasets(train_loader, val_loader):
        """Convert PyTorch DataLoader to TensorFlow Dataset and prefetch data for performance"""
        tf_train_loader = DataConverter.pytorch_to_tf_dataset(train_loader)
        tf_val_loader = DataConverter.pytorch_to_tf_dataset(val_loader)

        tf_train_loader = tf_train_loader.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
        tf_val_loader = tf_val_loader.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

        # Repeat datasets to avoid running out of data
        tf_train_loader = tf_train_loader.repeat().prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
        tf_val_loader = tf_val_loader.repeat().prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

        return tf_train_loader, tf_val_loader

    @staticmethod
    def prepare_test_datasets(test_loader):
        """Convert PyTorch DataLoader to TensorFlow Dataset and prefetch data for performance"""
        tf_test_loader = DataConverter.pytorch_to_tf_dataset(test_loader)

        tf_test_loader = tf_test_loader.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

        # Repeat datasets to avoid running out of data
        tf_test_loader = tf_test_loader.repeat().prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

        return tf_test_loader

    @staticmethod
    def train_model(model, train_loader, val_loader, train_steps_per_epoch, val_steps_per_epoch):
        """Train the model with the provided data loaders"""
        reduce_lr = callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.75, patience=5)

        history = model.fit(
            train_loader,
            epochs=1,
            validation_data=val_loader,
            steps_per_epoch=train_steps_per_epoch,
            validation_steps=val_steps_per_epoch,
            callbacks=[reduce_lr]
        )
        return history

class model_evaluation:
    @staticmethod
    def model_evaluation(model, test,class_names):
        pred = model.predict(test,verbose=1)
        pred_class = np.argmax(pred, axis=-1)

        true_class = []
        for images, labels in test:
          true_class.extend(labels.numpy())
        true_class = np.array(true_class)

        
        accuracy = accuracy_score(true_class, pred_class)
        precision = precision_score(true_class, pred_class, average='weighted')
        recall = recall_score(true_class, pred_class, average='weighted')
        f1 = f1_score(true_class, pred_class, average='weighted')
        print(f'Accuracy Score: {accuracy:.2f} \nPrecision Score: {precision:.2f} \nF1 Score: {f1:.2f} \nRecall Score: {recall:.2f}')

        
        print('\nClassification Report:')
        print(classification_report(true_class, pred_class, target_names=class_names))

# --------------------- Execution ---------------------

def main():
    batch_size=16
    # Load data and prepare loaders
    data_path = 'Image classifiers/data/Model1/model1_train.pth'
    train_loader, val_loader, class_names = DataLoaderHelper.data_train_loader(0.1, batch_size, data_path)

    # Create and compile the model
    img_size = (32, 32, 3)
    base_model = ResNetModel.create_base_model(input_shape=img_size)
    model = ResNetModel.build_model(input_shape=img_size, num_classes=len(class_names), base_model=base_model)

    # Convert data to TensorFlow dataset and prefetch
    tf_train_loader, tf_val_loader = ModelTrainer.prepare_datasets(train_loader, val_loader)
    
    # Train the model
    train_steps_per_epoch = len(train_loader)
    val_steps_per_epoch = len(val_loader)
    print(f"Train steps per epoch: {train_steps_per_epoch}")
    print(f"Validation steps per epoch: {val_steps_per_epoch}")

    start = time.time()
    history = ModelTrainer.train_model(model, tf_train_loader, tf_val_loader, train_steps_per_epoch, val_steps_per_epoch)
    stop = time.time()
    print(f"Training completed in {stop - start:.2f} seconds")
    
    test_path_data='Image classifiers/data/Model1/model1_test.pth'
    test_loader,class_names=DataLoaderHelper.data_test_loader(batch_size, test_path_data)
    tf_test_loader=ModelTrainer.prepare_test_datasets(test_loader)
    model_evaluation.model_evaluation(model, tf_test_loader,class_names)

if __name__ == "__main__":
    print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
    main()
