import torch
import pandas as pd
from torchvision import transforms, models
from torchvision.models import resnet18, ResNet18_Weights
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from PIL import Image
from tqdm import tqdm
import os

# Check for CUDA GPU availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class EmotionDataset(Dataset):
    def __init__(self, dataframe, root_dir, transform=None):
        self.dataframe = dataframe
        self.root_dir = root_dir
        self.transform = transform
        self.label_encoder = LabelEncoder()
        self.dataframe['encoded_labels'] = self.label_encoder.fit_transform(dataframe['label'])


    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, self.dataframe.iloc[idx, 1])
        image = Image.open(img_name)
        label = self.dataframe['encoded_labels'].iloc[idx]  # Corrected line
        label = torch.tensor(label, dtype=torch.long)

        if self.transform:
            image = self.transform(image)

        return image, label

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

train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False)

# Define model, loss function, optimizer
model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)


num_ftrs = model.fc.in_features
model.fc = torch.nn.Linear(num_ftrs, 6)

model = model.to(device)
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.001)

# Training loop
num_epochs = 5 # Set the number of epochs

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for inputs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
        inputs, labels = inputs.to(device), labels.to(device)

        # Zero the parameter gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, labels)

        # Backward pass and optimize
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader)}")


torch.save(model, 'emotion_model.pth')

# get data set stats
'''
print(f"Training set size: {len(train_df)}")
print(f"Validation set size: {len(val_df)}")
print(f"Test set size: {len(test_df)}")


print("Training set class distribution:")
print(train_df['label'].value_counts(normalize=True))

print("Validation set class distribution:")
print(val_df['label'].value_counts(normalize=True))

print("Test set class distribution:")
print(test_df['label'].value_counts(normalize=True))
'''

# get sample images from the dataset
'''
import matplotlib.pyplot as plt

def show_images(dataset, num_images=5):
    fig, axes = plt.subplots(1, num_images, figsize=(15, 3))
    for i in range(num_images):
        ax = axes[i]
        img, label = dataset[i]
        ax.imshow(img.permute(1, 2, 0))  # rearrange dimensions for plotting
        ax.set_title(label)
        ax.axis('off')
    plt.show()

# Show images from each dataset
show_images(train_dataset)
show_images(val_dataset)
show_images(test_dataset)
'''


