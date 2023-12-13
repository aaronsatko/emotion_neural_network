import torch
from torchvision import transforms
from PIL import Image
import os
from torchvision.models import resnet18

def load_model(model_path):
    # Instantiate the model with the new API call
    model = resnet18(weights=None)  
    num_ftrs = model.fc.in_features
    model.fc = torch.nn.Linear(num_ftrs, 6)  # Adjust as per your model

    # Load the state dict
    model.load_state_dict(torch.load(model_path))

    model.eval()
    return model


def process_image(image_path):
    """
    Process the image to the format required by the model
    """
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    image = Image.open(image_path)
    image = transform(image).unsqueeze(0)  # Add batch dimension
    return image

def predict(image_path, model):
    """
    Predict the class of the given image using the provided model.
    """
    image = process_image(image_path)
    with torch.no_grad():
        outputs = model(image)
        _, predicted = torch.max(outputs, 1)
        return predicted.item()

def predict_folder(folder_path, model):
    """
    Predict the class of all images in the given folder.
    """
    for filename in os.listdir(folder_path):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):  # Add other file types if needed
            image_path = os.path.join(folder_path, filename)
            prediction = predict(image_path, model)
            print(f'Image: {filename}, Predicted class: {prediction}')

if __name__ == "__main__":
    model_path = 'emotion_model.pth'
    folder_path = 'myImage'  # Folder containing images

    model = load_model(model_path)
    predict_folder(folder_path, model)
