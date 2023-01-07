import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from tqdm import tqdm
import time
import matplotlib.pyplot as plt
import pandas as pd
import os


# Define the hyperparameters
batch_size = 64
num_epochs = 150
input_size = (224, 224)
num_classes = 2
learning_rate = 3e-4

# Start the training timer
training_start_time = time.time()

# Define the paths to the training and validation data
path_to_train_data = "D:/Workzone/Datasets/bestphoto/train"
path_to_validation_data = "D:/Workzone/Datasets/bestphoto/validation"
path_to_training_folder  = "."

# Define the device to use for training
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Define a transform to preprocess the data
transform = transforms.Compose([
    transforms.ColorJitter(brightness=.5, hue=.3),
    transforms.RandomHorizontalFlip(),
    transforms.Resize(input_size),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

# Load the training data
train_dataset = datasets.ImageFolder(root=path_to_train_data, transform=transform)

train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
# Load the validation data
val_dataset = datasets.ImageFolder(root=path_to_validation_data, transform=transform)
val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=True, num_workers=4)

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
        self.softmax = nn.Softmax(dim=1)
        
    def forward(self, x):     
        x = self.conv1(x)
        x = self.pool(x)
        x = self.conv2(x)   
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.softmax(x)
        return x

# Create an instance of the CNN classifier
model = CNNClassifier(input_size=input_size, num_classes=num_classes).to(device)

# Define the loss function and optimizer
criterion = nn.NLLLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate )

# Initialize empty lists for the training and validation losses
train_losses = []
val_losses = []

# Define the training function
def train():
    # Train the model
    for epoch in range(num_epochs):
        start = time.time()

        epoch_loss = 0.0
        
        # Train the model on the training data
        model.train()
        for inputs, labels in tqdm(train_dataloader):
            # Move the data to the correct device
            inputs = inputs.to(device)
            labels = labels.to(device)

            # Clear the gradients
            optimizer.zero_grad()
            
            # Forward pass
            log_probs = model(inputs)
            
            # Compute the loss
            loss = criterion(log_probs, labels)
            epoch_loss += loss.item()
            
            # Backward pass
            loss.backward()
            
            # Update the weights
            optimizer.step()
            
        # Append the epoch loss to the training losses list
        train_losses.append(epoch_loss / len(train_dataloader))
        
        # Evaluate the model on the validation data
        model.eval()
        with torch.no_grad():
            correct = 0
            total = 0   
            val_loss = 0
            for inputs, labels in tqdm(val_dataloader):
                # Move the data to the correct device
                inputs = inputs.to(device)
                labels = labels.to(device)

                # Forward pass
                log_probs = model(inputs)

                # Compute the loss
                loss = criterion(log_probs, labels)
                val_loss += loss.item()

                _, predicted = torch.max(log_probs, 1)
 
                # Update the correct and total count
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                
            # Append the average validation loss to the validation losses list
            val_losses.append(val_loss / len(val_dataloader))

        # Calculate the accuracy
        accuracy = correct / total

        # Print the epoch loss and time elapsed
        print(f'Epoch {epoch+1} | Loss: {epoch_loss / len(train_dataloader):.4f} | Val Loss: {val_loss / len(val_dataloader):.4f} | Accuracy: {accuracy:.4f}" | Time: {time.time() - start:.2f}s')

        # Save a copy of the model every 10 epochs
        if (epoch+1) % 10 == 0:
            # Get the current time in a struct_time object
            now = time.gmtime()
 
            # Format the time stamp as a string and save the model
            time_stamp = time.strftime("%Y_%m_%d_%H_%M_%S", now)
            torch.save(model.state_dict(), os.path.join(path_to_training_folder, f"model_{epoch+1}_{time_stamp}.pt"))
            
            # Plot the learning curve
            save_learning_curve_data(train_losses, val_losses, epoch+1)

# Define the function to plot the learning curve
def save_learning_curve_data(train_losses, val_losses, epoch):
    # Create a dataframe from the lists of losses
    data = {'Epoch': range(1, epoch+1), 'Training Loss': train_losses, 'Validation Loss': val_losses}
    df = pd.DataFrame(data)
    
    # Get the current time
    now = time.gmtime()
    time_stamp = time.strftime("%Y_%m_%d_%H_%M_%S", now)
    
    # Save the dataframe as a CSV file
    df.to_csv(os.path.join(path_to_training_folder,f'learning_curve_epoch_{epoch}_{time_stamp}.csv'), index=False)


def save_model():
    # Get the current time in a struct_time object
    now = time.gmtime()
 
    # Format the time stamp as a string
    time_stamp = time.strftime("%Y_%m_%d_%H_%M_%S", now)
 
    # Generate a file name for the saved model
    model_name = f"model_{time_stamp}.pt"
 
    # Save the model
    torch.save(model.state_dict(), os.path.join(path_to_training_folder,model_name))
    print(f"Model saved as {model_name}")

def create_training_folder():
    global path_to_training_folder
    path_to_training_folder = os.path.join("training",time.strftime("%Y_%m_%d_%H_%M_%S", time.gmtime())+"/")

    # Create the training folder if it doesn't exist
    if not os.path.exists(path_to_training_folder):
        os.makedirs(path_to_training_folder)


if __name__ == "__main__":
    create_training_folder()
    train()
    save_model()
    save_learning_curve_data(train_losses, val_losses, num_epochs)
    print(f"Total training time: {time.strftime('%H:%M:%S', time.gmtime(time.time() - training_start_time))}")
