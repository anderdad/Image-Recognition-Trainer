import os
import json
import torch
import timm
import argparse
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# Load configuration
train_path = ""
val_path = ""
classes = []

with open('config.json') as f:
    data = json.load(f)
    train_path = data['relative_training_path']
    val_path = data['relative_validation_path']
    classes = data['classes']

# Function to create class to index mapping
def idx_json():
    idx = {}
    c = 0
    for i in classes.split(","):
        idx[i] = c
        c += 1
    return idx

def save_predictions(model, dataloader, device, output_file):
    model.eval()
    predictions = []
    labels = []
    
    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            predictions.extend(preds.cpu().numpy())
            labels.extend(targets.cpu().numpy())
    
    # Save predictions and labels to a file
    with open(output_file, 'w') as f:
        for pred, label in zip(predictions, labels):
            f.write(f"{pred},{label}\n")

def main():
    parser = argparse.ArgumentParser(description="Generate final model or output predictions using ResNet50 model")
    parser.add_argument('mode', choices=['final_model', 'predict'], help="Mode: 'final_model' to save the final model, 'predict' to output predictions")
    parser.add_argument('--checkpoint', type=str, default="resnet50_best.pth.tar", help="Path to save/load the model checkpoint")
    args = parser.parse_args()

    # Step 1: Define transformations
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # Resize images to 224x224
        transforms.ToTensor(),          # Convert images to PyTorch tensors
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize with ImageNet mean and std
    ])

    # Step 2: Load the datasets
    if args.mode == 'predict':
        dataset = datasets.ImageFolder(root=val_path, transform=transform)
    else:
        dataset = datasets.ImageFolder(root=train_path, transform=transform)
    dataset.class_to_idx = idx_json()

    # Step 3: Create data loader
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True if args.mode == 'final_model' else False, num_workers=4)

    # Step 4: Load or initialize the model
    model = timm.create_model('resnet50', pretrained=True, num_classes=len(classes))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    if args.mode == 'final_model':
        # Save the final trained model
        torch.save(model.state_dict(), "final_model.pth")
        print("Final model saved to final_model.pth")
    else:
        # Load the model from checkpoint
        if os.path.exists(args.checkpoint):
            model.load_state_dict(torch.load(args.checkpoint))
            print(f"Loaded model from checkpoint: {args.checkpoint}")
        else:
            print(f"Checkpoint not found: {args.checkpoint}")
            return
        # Save predictions
        save_predictions(model, dataloader, device, "predictions.csv")
        print("Predictions saved to predictions.csv")

if __name__ == "__main__":
    main()