
import os
from PIL import Image, ImageDraw, ImageFont, ImageTk
from tkinter import Tk, Label, BOTH
from tkinter.ttk import Frame
from tqdm import tqdm
import keyboard
import torch
import torch.nn as nn
from PIL import Image
import torch.nn.functional as F
from torchvision import datasets, transforms, models

# Define the batch size, input size, and number of classes
batch_size = 1
input_size = (224, 224)
num_classes = 2


resolution = (1920, 1080)

path_to_model= "./training/2023_01_07_20_18_35/model_20_2023_01_07_22_04_19.pt"
path_to_test_data = "D:/Workzone/Datasets/bestphoto/finaltestset"


# Define a transform to preprocess the data
transform = transforms.Compose([transforms.Resize(input_size),
                            transforms.ToTensor(),
                            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

# Load the images into a list
images = [f for f in os.listdir(path_to_test_data) if f.endswith('.png') or f.endswith('.jpg') or f.endswith('.jpeg')]

batch_index = 0

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



class ImageFrame(Frame):
    def __init__(self, parent, batch):
        Frame.__init__(self, parent)

        sorted_batch = sorted(batch, key=lambda x: x[1], reverse=True)


        imgs  = []
        labels = []

        for i in sorted_batch:
            imgs.append(i[0])
            labels.append(i[1])

        images = [Image.open(os.path.join(path_to_test_data , f)) for f in imgs]


        self.parent = parent
        self.batch = sorted_batch

        # The maximum width of the images
        max_width = 1400 ##resolution[0]
        max_height = 800 ##resolution[1]

        # Calculate the scaling factor for the images
        total_width = sum(i.size[0] for i in images)

        scale = max_width / total_width

        total_height = max(i.size[1]*scale for i in images)
        if total_height > max_height:
            scale = max_height / max(i.size[1] for i in images)


        # Create a blank image that is the size of all the images side by side
        width = sum(int(i.size[0] * scale) for i in images)
        height = max(int(i.size[1] * scale) for i in images)
        result = Image.new('RGB', (width, height))

        # Paste the images into the blank image
        x_offset = 0
        for im in images:
          im = im.resize((int(im.size[0] * scale), int(im.size[1] * scale)))
          result.paste(im, (x_offset,0))
          x_offset += im.size[0]

        # Draw the image names underneath each image
        draw = ImageDraw.Draw(result)
        font = ImageFont.truetype('arial.ttf', 16)
        x_offset = 0
        y_offset = height-30
        for im, f in zip(images, labels):
            draw.text((x_offset, y_offset), "{:.2%}".format(f), font=font, fill=(255,0,0))
            x_offset += im.size[0]*scale

        # Convert the image to a PhotoImage object
        self.image = ImageTk.PhotoImage(result)
        # Create a label with the image
        self.label = Label(self, image=self.image)

        self.label.pack(fill=BOTH, expand=True)


def next(frame):
    global batch_index

    # Check if the next frame would be out of range
    if batch_index + 1 >= len(batches):
        # If it is, start back at the beginning
        batch_index = 0
    else :
        batch_index += 1
    # Update the frame with the next images and filenames
    frame.batch = batches[batch_index]
    frame.image = ImageFrame(root, batches[batch_index]).image
    frame.label.configure(image=frame.image)


def previous(frame):
    global batch_index

    # Check if the next frame would be out of range
    if batch_index - 1 < 0:
        # If it is, start back at the beginning
        batch_index = len(batches) - 1
    else :
        batch_index -= 1
    # Update the frame with the next images and filenames
    frame.batch = batches[batch_index]
    frame.image = ImageFrame(root, batches[batch_index]).image
    frame.label.configure(image=frame.image)



def calculate_datamap():
    # Evaluate the model on the test data
    dataMap = {}
    model.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        for image in tqdm(images):

            
            index = image.split(".")[0].split("_")[0]
            variant = image.split(".")[0].split("_")[1]
            

            img = Image.open(os.path.join(path_to_test_data, image))
            img = transform(img).to(device)

            

            # Forward pass
            outputs = model(img.unsqueeze(0))

            # Apply softmax to the outputs
            probs = F.softmax(outputs, dim=1)

            # Get the predictions
            _, predicted = torch.max(outputs.data, 1)

            # Update the correct and total counts
            entry = [predicted[0].item(), probs[0][1].item()]

            
            try:
                dataMap[index][variant] = entry
            except KeyError:
                dataMap[index] = {}
                dataMap[index][variant] = entry
    return dataMap



dataMap = calculate_datamap()

batches = []
for key in tqdm(dataMap):
    batch =[]
    for variant in dataMap[key]:
        batch.append([f"{key}_{variant}.jpg", dataMap[key][variant][1]])
    batches.append(batch)        
        

# Create the main window
root = Tk()
root.title("Images")
root.geometry(f"{resolution[0]}x{resolution[1]}")


# Create the ImageFrame and add it to the window
frame = ImageFrame(root, batches[0])
frame.pack(fill=BOTH, expand=True)

# Set the right arrow key to trigger the print_annnnan function
keyboard.add_hotkey("right", next, args=(frame,))
keyboard.add_hotkey("left", previous, args=(frame,))

# Start a new thread to listen for key presses in the background
keyboard.start_recording()



# Run the Tkinter event loop
root.mainloop()