import torch
from model import UltraSmallCNN
from scripts.train import train_model  # Import the train function

def test_model_properties():
    # Initialize the model
    model = UltraSmallCNN()

    # Check total parameters
    total_params = sum(p.numel() for p in model.parameters())
    assert total_params < 25000, f"Model exceeds 25,000 parameters. Found {total_params} parameters."
    print(f"Total parameters test passed. {total_params} parameters.")

    # Train the model for 1 epoch to check accuracy
    accuracy, _ = train_model(train_loader=None, model=model, criterion=None, optimizer=None, num_epochs=1)  # You can skip the actual data loader for the test
    assert accuracy >= 95, f"Accuracy less than 95%. Found {accuracy:.2f}%."
    print(f"Accuracy test passed with {accuracy:.2f}%.")

if __name__ == "__main__":
    test_model_properties()
