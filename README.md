# Image Recognition Trainer
### Trains a resnet50 base model to provide a confidence score of an supplied image across 7 classes.

* Tree structure for training images
* Class names are based on the folder names
* Class names need to align with the class_to_idx
```
  .
  ├── README.md
  ├── base
  │   ├── train
  │   │   ├── class1
  |   |   |     ├── image_1.jpg
  |   |   |     ├── image_2.jpg
  |   |   |     |    ...
  |   |   |     └── image_n.jpg
  │   │   ├── class2
  |   |   |     ├── image_1.jpg
  |   |   |     ├── image_2.jpg
  |   |   |     |    ...
  |   |   |     └── image_n.jpg
  │   │   ├── class3
  │   │   ├── class4
  │   │   ├── class5
  │   │   ├── class6
  │   │   └── class7
  │   └── val
  │       ├── class1
  |       |     ├── image_1.jpg
  |       |     ├── image_2.jpg
  |       |     |    ...
  |       |     └── image_n.jpg
  │       ├── class2
  │       ├── class3
  │       ├── class4
  │       ├── class5
  │       ├── class6
  │       └── class7
  └── build_model.py
```

### Class names need to align with the code in Step 2.
### For Example: I was attempting to train against 7 African Animal Species 


### My classes in code:

```python
  # Step 2: Load the datasets
    train_dataset = datasets.ImageFolder(root='./animals/base/train', transform=transform )
    train_dataset.class_to_idx = {'Duiker': 0, 'Leopard': 1, 'Lion': 2, 'WildDog': 3, 'Hyena': 4, 'WartHog': 5, 'Jackal': 6}

    val_dataset = datasets.ImageFolder(root='./animals/base/val', transform=transform)
    val_dataset.class_to_idx = {'Duiker': 0, 'Leopard': 1, 'Lion': 2, 'WildDog': 3, 'Hyena': 4, 'WartHog': 5, 'Jackal': 
```    
### And my tree strucure:


```
  .
  ├── base
  ├── train
  │  ├── Duiker
  │  ├── Hyena
  │  ├── Jackal
  │  ├── Leopard
  │  ├── Lion
  │  ├── WartHog
  │  └── WildDog
  └── val
    ├── Duiker
    ├── Hyena
    ├── Jackal
    ├── Leopard
    ├── Lion
    ├── WartHog
    └── WildDog
```

# V1.1
## Ok. so we're all learning.... ;)
### model checkpoint saving a dataset does not make!
* Added generateDS.py to load the checkpoint model and output the dataset
* Once you are happy with your checkpoint model run ```python.exe .\create.py ```  (arg requied)
* This will load the ```resnet50_best.pth.tar``` :
* args (or arrrgs if you're a pirate)
```python.exe create.py final_model --checkpoint resnet50_checkpoint.pth```  to load your tuned checkpoint and build the model 
* or 
```python.exe generateDS.py predict --checkpoint resnet50_checkpoint.pth``` to run preditions against the model with your validation set 

### Added comfort 
* added ```config.json``` to make it easier to customize the script to your own needs  

* you still need to make sure the classes string numbers and entries match the folders