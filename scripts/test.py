import torch
import sys
import os
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from model import UltraSmallCNN
from scripts.train import train_model  # Import the train function

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

def test_model_properties():
    # Initialize the model
    model = UltraSmallCNN()

    # Check total parameters
    total_params = sum(p.numel() for p in model.parameters())
    assert total_params < 25000, f"Model exceeds 25,000 parameters. Found {total_params} parameters."
    print(f"Total parameters test passed. {total_params} parameters.")

    # Mock the data loader (you can use a minimal dataset for testing purposes)
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)

    # Mock the criterion and optimizer
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    # Train the model for 1 epoch to check accuracy
    accuracy, _ = train_model(train_loader=train_loader, model=model, criterion=criterion, optimizer=optimizer, num_epochs=1)
    
    assert accuracy >= 95, f"Accuracy less than 95%. Found {accuracy:.2f}%."
    print(f"Accuracy test passed with {accuracy:.2f}%.")

if __name__ == "__main__":
    test_model_properties()
