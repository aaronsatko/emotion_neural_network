import torch
from torchvision import transforms
from PIL import Image
import os
from torchvision.models import resnet18
import cv2

def load_model(model_path):
    # Update model instantiation
    model = resnet18(weights=None)  
    num_ftrs = model.fc.in_features
    model.fc = torch.nn.Linear(num_ftrs, 6)  # Adjust the number of outputs based on your model
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model


def process_image(image_path):
    # Load the pre-trained Haar Cascade model for face detection
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    # Read the image and convert it to grayscale (required for Haar Cascade)
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Detect faces in the image
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)

    # If at least one face is detected, crop the first face
    if len(faces) > 0:
        x, y, w, h = faces[0]  # Coordinates of the first face
        face = image[y:y+h, x:x+w]
        pil_image = Image.fromarray(face)
    else:
        pil_image = Image.open(image_path)  # If no face is detected, use the original image

    # Rest of your transformation steps
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    pil_image = transform(pil_image).unsqueeze(0)  # Add batch dimension
    return pil_image

def predict(image_path, model, class_names):
    image = process_image(image_path)
    with torch.no_grad():
        outputs = model(image)
        _, predicted = torch.max(outputs, 1)
        predicted_class_name = class_names[predicted.item()]
        return predicted_class_name

def predict_folder(folder_path, model, class_names):
    for filename in os.listdir(folder_path):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            image_path = os.path.join(folder_path, filename)
            predicted_class = predict(image_path, model, class_names)
            print(f'Image: {filename}, Predicted class: {predicted_class}')

if __name__ == "__main__":
    model_path = 'emotion_model.pth'
    folder_path = 'myImage'
    class_names = ['Ahegao', 'Angry', 'Happy', 'Neutral', 'Sad', 'Surprise']

    model = load_model(model_path)
    predict_folder(folder_path, model, class_names)
