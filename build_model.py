import os
import timm
import csv
import json
import torch
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from multiprocessing import freeze_support  # Import freeze_support
from torch.optim.lr_scheduler import StepLR
from torchvision.models import ResNet50_Weights

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
        c+=1  
        
    return idx
              
# Function to save checkpoint
def save_checkpoint(state, filename="checkpoint.pth.tar"):
    torch.save(state, filename)

def main(ttype):
    # Step 1: Define transformations
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # Resize images to 224x224
        transforms.ToTensor(),          # Convert images to PyTorch tensors
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # ```Normalize with ImageNet mean and std
    ])

    # Step 2: Load the datasets
    train_dataset = datasets.ImageFolder(root=train_path, transform=transform)
    train_dataset.class_to_idx = idx_json()


    val_dataset = datasets.ImageFolder(root=val_path, transform=transform)
    val_dataset.class_to_idx = idx_json()
    
    # Step 3: Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4)

    # Step 4: Load the model
    idx = idx_json()
    # checkpoint_path = "./resnet50_best.pth.tar"
    # if os.path.exists(checkpoint_path):
    #     model = timm.create_model(
    #         'resnet50', 
    #         pretrained=False,
    #         num_classes= len(idx),
    #         checkpoint_path=checkpoint_path
    #     )
    #     print(f"Loaded model from checkpoint: {checkpoint_path}")
    # else:
    #     model = timm.create_model(
    #         'resnet50', 
    #         pretrained=False,  
    #         num_classes=len(idx)
    #     )
    #     print("No checkpoint found, initialized model base models ## No pretrained weights")
    ################### new ###########################
        
        # Load the ResNet50 model with pre-trained weights
    model = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)

    # Modify the final layer to match the number of classes in your dataset
    num_classes = len(idx)  
    model.fc = torch.nn.Linear(model.fc.in_features, num_classes)

    # Print the model architecture (optional)
    print(model)

    # Now you can proceed with training the model
    # Define your loss function, optimizer, and learning rate scheduler here
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    # Example learning rate scheduler (Step LR)
    #scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

    ####################################################
    # Step 5: Set up the training loop
    criterion = torch.nn.CrossEntropyLoss()
    #optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    if ttype == 1:
        scheduler = StepLR(optimizer, step_size=7, gamma=0.01)  # Reduce LR by a factor of 0.1 every 5 epochs
    elif ttype == 2:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=5)  # Cosine annealing
    elif ttype == 3:
        optimizer = torch.optim.SGD(model.parameters(), lr=0.0001)
        scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=0.0001, max_lr=0.01)

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

def training_swicth():
    # 1. Step LR
    # 2. Cosine Annealing
    # 3. Cyclic Learning Rates
    # q. Quit
    
    while True:
        print("Traing a Resent50 model over 30 epochs using the Adam optimizer")
        print("Select the training type: ")
        print("1. stepLR")
        print("2. Cosine Annealing")
        print("3. Cyclic Learning Rates")
        print("q. Quit")
        
        choice = input()
        
        if choice == '1':
            print("Training with Learning Rate Scheduling,  step every 5 epoch , gamma=0.1")
            break
        elif choice == '2':
            print("Training with Cosine Annealing with a T MAX of 10")
            break
        elif choice == '3':
            print("Training with Cyclic Learning Rates with base learning rate 0.0001, max =0.01")
            break
        elif choice == 'q':
            print("Exiting program")
            exit()
        else:
            print("Invalid choice. Please select 1, 2, 3 or (q) to quit")
    return int(choice)
        
if __name__ == '__main__':  
    ttype = training_swicth()
    freeze_support()
    main(ttype)