import matplotlib.pyplot as plt
import csv
import argparse

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



# Initialize lists to store data
epochs = []
training_loss = []
validation_loss = []

# Open .csv file and read data into lists
with open(path, 'r') as f:
  reader = csv.reader(f)
  next(reader)  # Skip header row
  for row in reader:
    epochs.append(int(row[0]))
    training_loss.append(float(row[1]))
    validation_loss.append(float(row[2]))

# Plot learning curve
plt.plot(epochs, training_loss, label='Training Loss')
plt.plot(epochs, validation_loss, label='Validation Loss')
plt.title('Learning Curve')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(loc='upper right')
plt.show()