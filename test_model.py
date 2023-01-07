import os
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from torchvision import datasets, transforms


# Define the batch size, input size, and number of classes
batch_size = 1
input_size = (224, 224)
num_classes = 2

# Define a transform to preprocess the data
transform = transforms.Compose([
    
    transforms.Resize(input_size),
                        
                                transforms.ToTensor(),
                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

# Load the test data
path_to_test_data = "D:/Workzone/Datasets/bestphoto/test"
path_to_model= "preview_clasifier_20230106-232755.pt"

test_dataset = datasets.ImageFolder(root=path_to_test_data, transform=transform)
test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# Define the CNN classifier
class CNNClassifier(nn.Module):
    def __init__(self, input_size, num_classes):
        super(CNNClassifier, self).__init__()
        self.input_size = input_size
        self.num_classes = num_classes
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=8, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(50176, 128)
        self.fc2 = nn.Linear(128, num_classes)
        
    def forward(self, x):     
        x = self.conv1(x)
        x = self.pool(x)
        x = self.conv2(x)   
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = self.fc2(x)
        return x

# Create an instance of the CNN classifier
model = CNNClassifier(input_size=input_size, num_classes=num_classes).to(device)

# Load the trained model weights
model.load_state_dict(torch.load(path_to_model))

# Evaluate the model on the test data
model.eval()
with torch.no_grad():
    correct = 0
    total = 0
    for inputs, labels in tqdm(test_dataloader):
        # Move the data to the correct device
        inputs = inputs.to(device)
        labels = labels.to(device)

        # Forward pass
        outputs = model(inputs)

        # Get the predictions
        _, predicted = torch.max(outputs.data, 1)

        # Update the correct and total counts
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    # Print the accuracy
    print(f"Test accuracy: {correct / total:.4f}")