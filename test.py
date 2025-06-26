import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

# Set the device to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define the model structure (must match the original model used during training)
model_ft = models.resnet18(pretrained=False)  # pretrained=False since we're loading our own weights
num_ftrs = model_ft.fc.in_features
model_ft.fc = nn.Linear(num_ftrs, 2)  # Assuming binary classification (2 classes)
model_ft = model_ft.to(device)

# Load the saved model weights
model_ft.load_state_dict(torch.load('cataract_model.pth'))  # Adjust path if necessary
model_ft.eval()  # Set the model to evaluation mode

# Define the same transformations used for training
data_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Function to predict the class of a new image
def predict_image(image_path, model):
    image = Image.open(image_path)
    image = data_transforms(image).unsqueeze(0)  # Add batch dimension

    image = image.to(device)
    with torch.no_grad():
        outputs = model(image)
        _, predicted_class = torch.max(outputs, 1)
    
    class_name = 'Cataract' if predicted_class == 0 else 'No Cataract'  # Replace with actual class names
    return class_name

# Example usage
image_path = 'Cataract\\processed_images\\test\\normal\\image_260.png'  # Replace with the path to the new image
predicted_class = predict_image(image_path, model_ft)
print(f'Predicted Class: {predicted_class}')
