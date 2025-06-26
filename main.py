import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms
import os
import matplotlib.pyplot as plt

# Define data transformations for training and testing
data_transforms = {
    'train': transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'test': transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
}

data_dir = r'Cataract\\processed_images'  # Replace with your dataset path

# Load the training and testing datasets
image_datasets = {
    'train': datasets.ImageFolder(os.path.join(data_dir, 'train'), data_transforms['train']),
    'test': datasets.ImageFolder(os.path.join(data_dir, 'test'), data_transforms['test'])
}
dataloaders = {
    'train': torch.utils.data.DataLoader(image_datasets['train'], batch_size=32, shuffle=True),
    'test': torch.utils.data.DataLoader(image_datasets['test'], batch_size=32, shuffle=False)
}
dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'test']}
class_names = image_datasets['train'].classes

# Set the device to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load a pre-trained ResNet50 model and modify the final layer for regression (severity prediction as percentage)
model_ft = models.resnet50(pretrained=True)
num_ftrs = model_ft.fc.in_features
model_ft.fc = nn.Linear(num_ftrs, 1)  # Output one value (severity percentage)
model_ft = model_ft.to(device)

# Loss function (MSE for regression) and optimizer
criterion = nn.MSELoss()  # For percentage prediction, we use Mean Squared Error
optimizer = optim.Adam(model_ft.parameters(), lr=0.0001)

# Training function with severity prediction
def train_model(model, criterion, optimizer, num_epochs=25):
    for epoch in range(num_epochs):
        print(f'Epoch {epoch}/{num_epochs - 1}')
        print('-' * 10)

        model.train()  # Set the model to training mode

        running_loss = 0.0

        # Iterate over data
        for inputs, labels in dataloaders['train']:
            inputs = inputs.to(device)
            labels = labels.to(device).float()

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs.squeeze(), labels)  # Predict severity
            loss.backward()
            optimizer.step()

            # Update running loss
            running_loss += loss.item() * inputs.size(0)

        # Calculate epoch loss
        epoch_loss = running_loss / dataset_sizes['train']
        print(f'Train Loss: {epoch_loss:.4f}')

    return model

# Testing function to evaluate severity percentage
def test_model(model):
    model.eval()  # Set model to evaluation mode
    total_severity_error = 0

    for inputs, labels in dataloaders['test']:
        inputs = inputs.to(device)
        labels = labels.to(device).float()

        with torch.no_grad():
            outputs = model(inputs)
            preds = outputs.squeeze()

        severity_error = torch.abs(preds - labels)  # Absolute error in severity prediction
        total_severity_error += torch.sum(severity_error).item()

    mean_severity_error = total_severity_error / dataset_sizes['test']
    print(f'Mean Severity Prediction Error: {mean_severity_error:.4f}%')

# Train the model
model_ft = train_model(model_ft, criterion, optimizer, num_epochs=10)

# Save the trained model
torch.save(model_ft.state_dict(), 'cataract_severity_model.pth')

# Test the model for severity prediction
test_model(model_ft)
