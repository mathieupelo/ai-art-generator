import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torch import nn, optim
from torch.optim.lr_scheduler import StepLR
from torchvision import transforms
from torchvision.models import resnet50
from torchvision.datasets import ImageFolder

# Define the dataset class
class MyDataset(Dataset):
    def __init__(self, csv_file, transform=None):
        self.data = pd.read_csv(csv_file)
        self.transform = transform
        self.classes = self.data.iloc[:, 1].unique()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image_path = self.data.iloc[idx, 0]
        image = Image.open(image_path).convert("RGB")
        label = self.data.iloc[idx, 1]
        
        if self.transform:
            image = self.transform(image)
        
        return image, label

# Define the data transformations
data_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Create the dataset
dataset = MyDataset('TKT.csv', transform=data_transform)

# Create the data loader
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# Load the pre-trained ResNet50 model
model = resnet50(pretrained=True)

# Replace the last fully connected layer with a new one
num_features = model.fc.in_features
model.fc = nn.Linear(num_features, len(dataset.classes))

# Define the loss function
criterion = nn.CrossEntropyLoss()

# Define the optimizer
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

# Define the learning rate scheduler
scheduler = StepLR(optimizer, step_size=7, gamma=0.1)

# Train the model
num_epochs = 10
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    
    for images, labels in dataloader:
        images = images.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()
        
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * images.size(0)
    
    scheduler.step()
    
    epoch_loss = running_loss / len(dataset)
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}")

# Save the finetuned model
torch.save(model.state_dict(), 'finetuned_model.pth')
