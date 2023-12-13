import torch
import pandas as pd
from torchvision import transforms, models
from torchvision.models import resnet18, ResNet18_Weights
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# Importing the dataset class
from emotion_dataset import EmotionDataset

# Check for CUDA GPU availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load CSV file and split data
dataframe = pd.read_csv('FRData/data.csv')
train_df, test_df = train_test_split(dataframe, test_size=0.3)
test_df, val_df = train_test_split(test_df, test_size=0.5)

# Define transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Initialize datasets and dataloaders
train_dataset = EmotionDataset(dataframe=train_df, root_dir='FRData', transform=transform)
val_dataset = EmotionDataset(dataframe=val_df, root_dir='FRData', transform=transform)
test_dataset = EmotionDataset(dataframe=test_df, root_dir='FRData', transform=transform)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# Define model, loss function, optimizer
model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)


num_ftrs = model.fc.in_features
model.fc = torch.nn.Linear(num_ftrs, 6)

model = model.to(device)
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

# Training loop
num_epochs = 5

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    correct_predictions = 0
    total_predictions = 0

    for inputs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
        inputs, labels = inputs.to(device), labels.to(device)

        # Zero the parameter gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        running_loss += loss.item()

        # Backward pass and optimize
        loss.backward()
        optimizer.step()

        # Calculate accuracy
        _, predicted = torch.max(outputs, 1)
        correct_predictions += (predicted == labels).sum().item()
        total_predictions += labels.size(0)

    # Calculate average loss and accuracy over the epoch
    epoch_loss = running_loss / len(train_loader)
    epoch_accuracy = correct_predictions / total_predictions

    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}, Accuracy: {epoch_accuracy:.4f}")



torch.save(model.state_dict(), 'emotion_model.pth')