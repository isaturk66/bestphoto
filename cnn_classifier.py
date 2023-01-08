import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms, models
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
weight_decay = 1e-4
feature_extract = True

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


def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False
# Define the model
def get_model():
    model_ft = models.resnet18(pretrained=True)
    set_parameter_requires_grad(model_ft, feature_extract)
    num_ftrs = model_ft.fc.in_features
    model_ft.fc = nn.Linear(num_ftrs, num_classes)
    return model_ft
    

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
            outputs = model(inputs)
            # Compute the loss
            loss = criterion(outputs, labels)
            epoch_loss += loss.item()
            # Backward pass
            loss.backward()
            # Update the parameters
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
                outputs = model(inputs)
                # Apply softmax to the outputs
                probs = F.softmax(outputs, dim=1)

                # Compute the loss
                loss = criterion(outputs, labels)
                val_loss += loss.item()
 
                # Get the predicted class
                _, predicted = torch.max(outputs.data, 1)

                total += labels.size(0)
                correct += (predicted == labels).sum().item()

                # Print the probability and prediction for each image
                #for i in range(len(predicted)):
                #    print(f'Prediction: {predicted[i]}, Probability: {probs[i][predicted[i]]:.4f}')
                
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
            save_learning_curve_data(train_losses, val_losses, epoch+1)
            
            # Plot the learning curve

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
    global model
    global criterion
    global optimizer

    # Create the training folder
    create_training_folder()

    # Load the model to the correct device
    model = get_model().to(device)

    # Define the loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    # Start the timer
    training_start_time = time.time()

    train()
    save_model()
    save_learning_curve_data(train_losses, val_losses, num_epochs)
    print(f"Total training time: {time.strftime('%H:%M:%S', time.gmtime(time.time() - training_start_time))}")