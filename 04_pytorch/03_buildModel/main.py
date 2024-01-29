import numpy as np
import torch 
from torch.utils.data import DataLoader
import torchvision
from torchvision import transforms
from torchvision.models.resnet import resnet50

import os
import argparse
from tqdm import tqdm

from utils.dataset import CustomDataset
from archs import SimpleVGG

# Define Transforms 
transform = transforms.Compose([
    transforms.Resize([224, 224]), 
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# Define Dataset 
train_dataset = CustomDataset(root = './datasets', transform = transform, mode = 'train')
val_dataset = CustomDataset(root = './datasets', transform = transform, mode = 'val')
print(f"Length of train, validation dataset: {len(train_dataset)}, {len(val_dataset)}")

# Pass to DataLoader 
train_dataloader = DataLoader(train_dataset, batch_size = 4, shuffle = True, num_workers = 2)
val_dataloader = DataLoader(val_dataset, batch_size = 4, shuffle = False, num_workers = 2)

# Index to change labels 
index_for_class = {'apples': 0, 'peaches': 1, 'tomatoes': 2}
    
# Device 
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using {device} for inference")

# Model 
model = SimpleVGG(3)
model.to(device)
breakpoint()
# Loss, Optimizer 
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr = 0.0001)

breakpoint()

# Train Loop 
for ep in range(10):
    model.train()       # Set model in training mode
    size = len(train_dataloader)
    
    for batch, (images, labels) in enumerate(tqdm(train_dataloader)):
        # Make labels to indices
        label_to_index = [index_for_class[label] for label in labels]
        labels = torch.tensor(label_to_index)
        
        # Send (images, labels) to device 
        images = images.to(device)
        labels = labels.to(device)
        
        # Compute prediction and loss
        pred = model(images)
        loss = criterion(pred, labels)
        
        # Backpropagation 
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        
        if batch % 10 == 0:
            print(f"Epoch : {ep}, Loss : {loss.item()}")

    # Validation 
    model.eval()                # Set model in validation mode 
    total = 0 
    correct = 0 

    with torch.no_grad():       # No gradients computed during validation mode 
        for (images, labels) in val_dataloader:
            total += len(images)
            
            # Make labels to indices 
            label_to_index = [index_for_class[label] for label in labels]
            labels = torch.tensor(label_to_index)
            
            # Send (images, labels) to device 
            images = images.to(device)
            labels = labels.to(device)
            
            # Forward pass : Compute prediction and loss 
            outputs = model(images)
            
            _, pred = torch.max(outputs, dim=1)
            
            correct += (pred == labels).sum().item() 
            
        print(f"After training epoch {ep}, Validation accuracy: {correct/total}")
        
# Check if 'outputs' directory exists. If not, create one.
if not os.path.exists('./outputs'):
    os.makedirs('./outputs')

# Save checkpoint after all epochs
torch.save(model.state_dict(), './outputs/checkpoint.pth')