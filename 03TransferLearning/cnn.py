import json
import matplotlib.pyplot as plt
import numpy as np
import os
import torch
import torchvision
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms
from tempfile import TemporaryDirectory
import json




import torch
import torch.nn as nn
from torchvision.models import efficientnet_b7, EfficientNet_B7_Weights

class CNN(nn.Module):
    """
    CNN model using a pre-trained EfficientNet-B7 as the base.
    The built-in classifier of EfficientNet-B7 is replaced with an identity
    so that the base model returns a feature vector of size 2560.
    A new classifier is added on top for fine-tuning.
    """
    def __init__(self, base_model, num_classes, unfreezed_layers=0):
        """
        Args:
            base_model: Pre-trained EfficientNet-B7 model with classifier replaced by Identity.
            num_classes: Number of output classes.
            unfreezed_layers: (Optional) Number of layers (from the end of base_model.features) to unfreeze.
        """
        super().__init__()
        self.base_model = base_model
        
        # Freeze all parameters in the base model
        for param in self.base_model.parameters():
            param.requires_grad = False

        # Optionally unfreeze the last few layers in the features module for fine-tuning.
        if unfreezed_layers > 0:
            # EfficientNet's convolutional body is stored in base_model.features (a Sequential container)
            features = list(self.base_model.features.children())
            for layer in features[-unfreezed_layers:]:
                for param in layer.parameters():
                    param.requires_grad = True

        # EfficientNet-B7 returns a flattened feature vector of size 2560.
        in_features = 2560
        self.classifier = nn.Sequential(
            nn.Linear(in_features, 1024),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(1024, num_classes),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        # Pass through the base model. The EfficientNet-B7 forward does:
        # features -> avgpool -> flatten -> classifier (here Identity)
        x = self.base_model(x)
        # x is now a tensor of shape (batch, 2560)
        x = self.classifier(x)
        return x


    def train_model(self, 
                    train_loader, 
                    valid_loader, 
                    optimizer, 
                    criterion, 
                    epochs, 
                    nepochs_to_save=10):
        """Train the model and save the best one based on validation accuracy.
        
        Args:
            train_loader: DataLoader with training data.
            valid_loader: DataLoader with validation data.
            optimizer: Optimizer to use during training.
            criterion: Loss function to use during training.
            epochs: Number of epochs to train the model.
            nepochs_to_save: Number of epochs to wait before saving the model.

        Returns:
            history: A dictionary with the training history.
        """
        with TemporaryDirectory() as temp_dir:
            best_model_path = os.path.join(temp_dir, 'best_model.pt')
            best_accuracy = 0.0
            torch.save(self.state_dict(), best_model_path)

            history = {'train_loss': [], 'train_accuracy': [], 'valid_loss': [], 'valid_accuracy': []}
            for epoch in range(epochs):
                self.train()
                train_loss = 0.0
                train_accuracy = 0.0
                for images, labels in train_loader:
                    optimizer.zero_grad()
                    outputs = self(images)
                    loss = criterion(outputs, labels)
                    loss.backward()
                    optimizer.step()
                    train_loss += loss.item()
                    train_accuracy += (outputs.argmax(1) == labels).sum().item()

                train_loss /= len(train_loader)
                train_accuracy /= len(train_loader.dataset)
                history['train_loss'].append(train_loss)
                history['train_accuracy'].append(train_accuracy)

                print(f'Epoch {epoch + 1}/{epochs} - '
                      f'Train Loss: {train_loss:.4f}, '
                      f'Train Accuracy: {train_accuracy:.4f}')
                
                
                self.eval()
                valid_loss = 0.0
                valid_accuracy = 0.0
                for images, labels in valid_loader:
                    outputs = self(images)
                    loss = criterion(outputs, labels)
                    valid_loss += loss.item()
                    valid_accuracy += (outputs.argmax(1) == labels).sum().item()

                valid_loss /= len(valid_loader)
                valid_accuracy /= len(valid_loader.dataset)
                history['valid_loss'].append(valid_loss)
                history['valid_accuracy'].append(valid_accuracy)

                print(f'Epoch {epoch + 1}/{epochs} - '
                        f'Validation Loss: {valid_loss:.4f}, '
                        f'Validation Accuracy: {valid_accuracy:.4f}')
                
                if epoch % nepochs_to_save == 0:
                    if valid_accuracy > best_accuracy:
                        best_accuracy = valid_accuracy
                        torch.save(self.state_dict(), best_model_path)
                
            torch.save(self.state_dict(), best_model_path)    
            self.load_state_dict(torch.load(best_model_path))
            return history
        
    def predict(self, data_loader):
        """Predict the classes of the images in the data loader.

        Args:
            data_loader: DataLoader with the images to predict.

        Returns:
            predicted_labels: Predicted classes.
        """
        self.eval()
        predicted_labels = []
        for images, _ in data_loader:
            outputs = self(images)
            predicted_labels.extend(outputs.argmax(1).tolist())
        return predicted_labels
        
    def save_model(self, filename: str):
        """Save the model to disk.

        Args:
            filename: Name of the file to save the model.
        """
        # If the directory does not exist, create it
        os.makedirs(os.path.dirname('models/'), exist_ok=True)

        # Full path to the model
        filename = os.path.join('models', filename)

        # Save the model
        torch.save(self.state_dict(), filename+'.pt')

    @staticmethod
    def _plot_training(history):
        """Plot the training history.

        Args:
            history: A dictionary with the training history.
        """
        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plt.plot(history['train_loss'], label='Train Loss')
        plt.plot(history['valid_loss'], label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()

        plt.subplot(1, 2, 2)
        plt.plot(history['train_accuracy'], label='Train Accuracy')
        plt.plot(history['valid_accuracy'], label='Validation Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()

        plt.show()

    

        

def load_data(train_dir, valid_dir, batch_size, img_size):
    """Load and transform the training and validation datasets.

    Args:
        train_dir: Path to the training dataset.
        valid_dir: Path to the validation dataset.
        batch_size: Number of images per batch.
        img_size: Expected size of the images.

    Returns:
        train_loader: DataLoader with the training dataset.
        valid_loader: DataLoader with the validation dataset.
        num_classes: Number of classes in the dataset.
    """
    train_transforms = transforms.Compose([
        transforms.RandomRotation(30), # Rotate the image by a random angle
        transforms.RandomResizedCrop(img_size), # Crop the image to a random size and aspect ratio
        transforms.RandomHorizontalFlip(), # Horizontally flip the image with a 50% probability
        transforms.ToTensor() 
    ])

    valid_transforms = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor() 
    ])

    train_data = torchvision.datasets.ImageFolder(train_dir, transform=train_transforms)
    valid_data = torchvision.datasets.ImageFolder(valid_dir, transform=valid_transforms)

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_data, batch_size=batch_size, shuffle=False)

    return train_loader, valid_loader, len(train_data.classes)

def load_model_weights(filename: str):
        """Load a model from disk.
        IMPORTANT: The model must be initialized before loading the weights.
        Args:
            filename: Name of the file to load the model.
        """
        # Full path to the model
        filename = os.path.join('models', filename)

        # Load the model
        state_dict = torch.load(filename+'.pt')
        return state_dict