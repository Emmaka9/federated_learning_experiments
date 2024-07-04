import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import requests
import phe as paillier
import json

# Initialize Paillier encryption
public_key, private_key = paillier.generate_paillier_keypair()

# Function to generate synthetic data
def generate_synthetic_data(num_samples=1000):
    np.random.seed(42)
    data = np.random.rand(num_samples, 20)
    labels = np.random.randint(2, size=num_samples)
    return data, labels

# Generate and save synthetic data
data, labels = generate_synthetic_data()
pd.DataFrame(data).to_csv('synthetic_data.csv', index=False)
pd.DataFrame(labels).to_csv('synthetic_labels.csv', index=False)

# Load data
def load_data():
    data = pd.read_csv('synthetic_data.csv').values
    labels = pd.read_csv('synthetic_labels.csv').values.flatten()
    return torch.tensor(data, dtype=torch.float32), torch.tensor(labels, dtype=torch.long)

# Define a simple neural network
class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(20, 50)
        self.fc2 = nn.Linear(50, 2)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Fetch initial model from server
def fetch_initial_model(server_url):
    response = requests.get(server_url + "/send_initial_model")
    if response.status_code != 200:
        print(f"Error: Received status code {response.status_code}")
        print(f"Response text: {response.text}")
        response.raise_for_status()
    initial_weights = response.json()['weights']
    return initial_weights

# Train the model
def train_model(model, data, labels, epochs=1):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01)
    for epoch in range(epochs):
        optimizer.zero_grad()
        outputs = model(data)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
    return model

# Encrypt model weights
def encrypt_weights(weights, public_key):
    encrypted_weights = []
    for weight in weights:
        encrypted_weight = [public_key.encrypt(float(x)) for x in weight]
        encrypted_weights.append([(e.ciphertext(), e.exponent) for e in encrypted_weight])
    return encrypted_weights

# Send model weights to the server
def send_model_update(model, server_url, public_key):
    model_weights = [param.data.numpy().tolist() for param in model.parameters()]
    encrypted_weights = encrypt_weights(model_weights, public_key)
    response = requests.post(server_url + "/update_model", json={'weights': encrypted_weights})
    if response.status_code != 200:
        print(f"Error: Received status code {response.status_code}")
        print(f"Response text: {response.text}")
        response.raise_for_status()
    try:
        aggregated_weights = response.json()['aggregated_weights']
    except requests.exceptions.JSONDecodeError as e:
        print("JSONDecodeError:", e)
        print("Response text:", response.text)
        raise
    return [torch.tensor(w) for w in aggregated_weights]

# Load and set model weights
def set_model_weights(model, weights):
    with torch.no_grad():
        for param, new_weight in zip(model.parameters(), weights):
            param.copy_(new_weight)

# Main function
def main(server_url, num_rounds):
    # Fetch initial model
    initial_weights = fetch_initial_model(server_url)
    data, labels = load_data()
    model = SimpleNN()
    set_model_weights(model, [torch.tensor(w) for w in initial_weights])
    for round in range(num_rounds):
        model = train_model(model, data, labels, epochs=5)
        aggregated_weights = send_model_update(model, server_url, public_key)
        set_model_weights(model, aggregated_weights)

if __name__ == "__main__":
    server_url = "http://10.0.0.163:5000"
    num_rounds = 10
    main(server_url, num_rounds)
