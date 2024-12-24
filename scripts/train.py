import os
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torch.nn as nn
import torch.optim as optim
from datetime import datetime
from tqdm import tqdm
from model import UltraSmallCNN  # Import the model

# Dataset transformations
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# Data loaders
train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)

# Model
model = UltraSmallCNN()
device = torch.device('cpu')
model = model.to(device)

# Training setup
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# Training function
def train_model(train_loader, model, criterion, optimizer, num_epochs=1):
    model.train()
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total Parameters: {total_params}")

    for epoch in range(num_epochs):
        running_loss = 0.0
        correct_predictions = 0
        total_predictions = 0

        for inputs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            _, predicted = torch.max(outputs.data, 1)
            total_predictions += labels.size(0)
            correct_predictions += (predicted == labels).sum().item()
            running_loss += loss.item()

        accuracy = (correct_predictions / total_predictions) * 100
        print(f"Epoch {epoch+1}: Loss: {running_loss / len(train_loader):.4f}, Accuracy: {accuracy:.2f}%")
        return accuracy, total_params

# Save model with suffix

def save_model(model):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_dir = './model'
    
    # Create the 'model' directory if it doesn't exist
    os.makedirs(model_dir, exist_ok=True)

    # Define the model file path
    model_path = f"{model_dir}/mnist_model_{timestamp}.pth"
    
    # Save the model
    torch.save(model.state_dict(), model_path)
    
    print(f"Model saved to {model_path}")
    return model_path


# Main
if __name__ == "__main__":
    accuracy, total_params = train_model(train_loader, model, criterion, optimizer)
    save_model(model)
