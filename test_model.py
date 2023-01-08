import os
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import argparse
from torchvision import datasets, transforms, models

def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(
        description='Binary image classifier tool')
    parser.add_argument('path',
                       help='The path that contains image files',
                        type=str)
    args = parser.parse_args()
    return args

args = parse_args()
path = args.path

if(path == None):
    print("Please specify the path to the image directory.")
    exit()


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
path_to_model= path

test_dataset = datasets.ImageFolder(root=path_to_test_data, transform=transform)
test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False
# Define the model
def get_model():
    model_ft = models.resnet18(pretrained=True)
    set_parameter_requires_grad(model_ft, True)
    num_ftrs = model_ft.fc.in_features
    model_ft.fc = nn.Linear(num_ftrs, num_classes)
    return model_ft
    


model = get_model().to(device)

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

        # Forward pass
        outputs = model(inputs)

        # Get the predictions
        _, predicted = torch.max(outputs.data, 1)

        # Update the correct and total counts
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    # Print the accuracy
    print(f"Test accuracy: {correct / total:.4f}")