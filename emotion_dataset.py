from torch.utils.data import Dataset
from sklearn.preprocessing import LabelEncoder
from PIL import Image
import os
import torch

class EmotionDataset(Dataset):
    def __init__(self, dataframe, root_dir, transform=None):
        self.dataframe = dataframe
        self.root_dir = root_dir
        self.transform = transform
        self.label_encoder = LabelEncoder()
        self.dataframe['encoded_labels'] = self.label_encoder.fit_transform(dataframe['label'])
        self.class_names = self.label_encoder.classes_  # Store class names

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, self.dataframe.iloc[idx, 1])
        image = Image.open(img_name)
        label = self.dataframe['encoded_labels'].iloc[idx]
        label = torch.tensor(label, dtype=torch.long)

        if self.transform:
            image = self.transform(image)

        return image, label
    
    def get_class_names_with_indices(self):
        return {index: label for index, label in enumerate(self.class_names)}
    
