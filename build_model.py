import os
import timm
import csv
import json
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from multiprocessing import freeze_support  # Import freeze_support
from torch.optim.lr_scheduler import StepLR

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
    for c in classes:
        idx.append({c: classes.index(c)})
    return idx
              
# Function to save checkpoint
def save_checkpoint(state, filename="checkpoint.pth.tar"):
    torch.save(state, filename)

def main():
    # Step 1: Define transformations
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # Resize images to 224x224
        transforms.ToTensor(),          # Convert images to PyTorch tensors
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # ```Normalize with ImageNet mean and std
    ])

    # Step 2: Load the datasets
    #train_dataset = datasets.ImageFolder(root='./animals/base/train', transform=transform )
    #train_dataset.class_to_idx = {'Duiker': 0, 'Leopard': 1, 'Lion': 2, 'WildDog': 3, 'Hyena': 4, 'WartHog': 5, 'Jackal': 6}

    train_dataset = datasets.ImageFolder(root=train_path, transform=transform)
    train_dataset.class_to_idx = idx_json()


    val_dataset = datasets.ImageFolder(root=val_path, transform=transform)
    val_dataset.class_to_idx = idx_json()
    
    
    #val_dataset = datasets.ImageFolder(root='./animals/base/val', transform=transform)
    #val_dataset.class_to_idx = {'Duiker': 0, 'Leopard': 1, 'Lion': 2, 'WildDog': 3, 'Hyena': 4, 'WartHog': 5, 'Jackal': 6}
   
    # Step 3: Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4)

    # Step 4: Load the model
    checkpoint_path = "./resnet50_best.pth.tar"
    if os.path.exists(checkpoint_path):
        model = timm.create_model(
            'resnet50', 
            pretrained=False,
            num_classes= len(classes),
            checkpoint_path=checkpoint_path
        )
        print(f"Loaded model from checkpoint: {checkpoint_path}")
    else:
        model = timm.create_model(
            'resnet50', 
            pretrained=False,  
            num_classes=len(classes)
        )
        print("No checkpoint found, initialized model base models ## No pretrained weights")

    # Step 5: Set up the training loop
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    scheduler = StepLR(optimizer, step_size=5, gamma=0.1)  # Reduce LR by a factor of 0.1 every 5 epochs

    # Move model to GPU if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    # Training loop
    num_epochs = 30
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for i, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            
            # Print loss for every 10 batches
            if i % 10 == 0:
                print(f"Batch {i}, Loss: {loss.item()}")
            
        
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {running_loss/len(train_loader)}")
        
        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
        }, filename=checkpoint_path)
        print(f"Checkpoint saved after training at epoch {epoch+1}")
   
        # Validation loop
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        print(f"Validation Loss: {val_loss/len(val_loader)}, Accuracy: {100 * correct / total}%")
          
        # Save checkpoint after validation
        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
        }, filename=checkpoint_path)
        print(f"Checkpoint saved after validation at epoch {epoch+1}")

        # Step the scheduler    
        scheduler.step()

if __name__ == '__main__':
    freeze_support()
    main()