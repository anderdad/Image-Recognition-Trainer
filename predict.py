import torch
import argparse
import json
import os
from PIL import Image
from torchvision import transforms

classes = []
idx = {}
with open('config.json') as f:
    c=0
    data = json.load(f)
    classes = data['classes'] 
    for i in classes.split(","):
        idx[i] = c
        c+=1  
       
# Define the class labels
# Function to preprocess the image
def preprocess_image(image_path):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # Resize images to 224x224
        transforms.ToTensor(),          # Convert images to PyTorch tensors
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize with ImageNet mean and std
    ])
    image = Image.open(image_path).convert('RGB')
    return transform(image).unsqueeze(0)  # Add batch dimension

# Function to predict the class of an image
def predict_image(model, image_tensor, device):
    image_tensor = image_tensor.to(device)
    with torch.no_grad():
        outputs = model(image_tensor)
        _, preds = torch.max(outputs, 1)
    return preds.cpu().numpy()[0]

def main():
    parser = argparse.ArgumentParser(description="Evaluate an image using the trained ResNet50 model")
    parser.add_argument('--image_path', type=str, default='img/', help="Folder path to the image to be evaluated")
    parser.add_argument('--model_path', type=str, default="complete_model.pth", help="Path to the trained model")
    args = parser.parse_args()

    # Preprocess the image
    # Load the model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = torch.load(args.model_path)
    model.to(device)
    model.eval()
        
     
    for img_file in os.listdir(args.image_path):
        img_path = os.path.join(args.image_path, img_file)
        image_tensor = preprocess_image(img_path)
        # Predict the class
        predicted_class_idx = predict_image(model, image_tensor, device)
        predicted_class = classes.split(",")[predicted_class_idx]
        # predicted_class = idx[predicted_class_idx]
        print(f"The model predicts the {img_file} image is a: {predicted_class}")

if __name__ == "__main__":
    main()