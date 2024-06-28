import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import requests
import json
import syft as sy

# Setup PySyft
hook = sy.TorchHook(torch)

# Define a simple neural network
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(20, 50)
        self.fc2 = nn.Linear(50, 2)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Function to load synthetic data
def load_data():
    # Load synthetic data and labels from CSV files
    data = pd.read_csv('synthetic_data.csv').values
    labels = pd.read_csv('synthetic_labels.csv').values.flatten()
    return torch.tensor(data, dtype=torch.float32), torch.tensor(labels, dtype=torch.long)

# Function to train the model on local data
def train_model(model, data, labels, epochs=1):
    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01)
    for epoch in range(epochs):
        # Zero the parameter gradients
        optimizer.zero_grad()
        # Forward pass
        outputs = model(data)
        # Compute loss
        loss = criterion(outputs, labels)
        # Backward pass and optimize
        loss.backward()
        optimizer.step()
    return model

# Function to send the model update to the server
def send_model_update(model, server_url, client):
    # Get the model weights
    model_weights = [param.data for param in model.parameters()]
    # Encrypt the model weights using PySyft
    encrypted_weights = [weight.fix_precision().share(client) for weight in model_weights]
    # Convert weights to JSON format
    weights_json = json.dumps([w.tolist() for w in encrypted_weights])
    # Send weights to the server
    response = requests.post(server_url + "/update_model", json={'weights': weights_json})
    # Get aggregated weights from server response
    return response.json()['aggregated_weights']

# Function to set the model weights received from the server
def set_model_weights(model, weights, client):
    # Decrypt the aggregated weights using PySyft
    decrypted_weights = [torch.tensor(w).get().float_precision() for w in weights]
    # Set the decrypted weights to the model
    for param, new_weight in zip(model.parameters(), decrypted_weights):
        param.data.copy_(new_weight)

# Main function to coordinate training and communication
def main(server_url, num_rounds):
    # Load synthetic data
    data, labels = load_data()
    # Initialize the model
    model = Net()
    # Create a PySyft VirtualWorker for the client
    client = sy.VirtualWorker(hook, id="client")
    
    for round in range(num_rounds):
        # Train the model on local data
        model = train_model(model, data, labels, epochs=5)
        # Send the model updates to the server and receive aggregated weights
        aggregated_weights = send_model_update(model, server_url, client)
        # Set the aggregated weights to the model
        set_model_weights(model, aggregated_weights, client)

# Entry point of the script
if __name__ == "__main__":
    # Define the server URL
    server_url = "http://<server-ip>:5000"
    # Define the number of training and aggregation rounds
    num_rounds = 10
    # Execute the main function
    main(server_url, num_rounds)
