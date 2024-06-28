from flask import Flask, request, jsonify
import torch
import torch.nn as nn
import numpy as np
import json
import matplotlib.pyplot as plt
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

# Initialize Flask app
app = Flask(__name__)

# Load global model
global_model = Net()

# Number of clients and rounds
num_clients = 1
num_rounds = 10

# Create a PySyft VirtualWorker for the server
server = sy.VirtualWorker(hook, id="server")

# Function to aggregate client weights securely
def aggregate_weights(client_weights):
    # Aggregate weights by summing them and dividing by the number of clients
    aggregated_weights = []
    for weights_list_tuple in zip(*client_weights):
        aggregated_weights.append(
            sum([w.get() for w in weights_list_tuple]) / len(weights_list_tuple)
        )
    return aggregated_weights

# Function to evaluate the global model on validation data
def evaluate_model(model):
    # Load validation data
    data, labels = load_validation_data()
    # Define loss function
    criterion = nn.CrossEntropyLoss()
    # Compute outputs
    outputs = model(data)
    # Compute loss and accuracy
    loss = criterion(outputs, labels).item()
    accuracy = (outputs.argmax(1) == labels).float().mean().item()
    return loss, accuracy

# Endpoint to receive model updates from clients
@app.route("/update_model", methods=["POST"])
def update_model():
    global global_model
    # Get the encrypted weights from the client
    data = request.get_json()
    client_weights = json.loads(data['weights'])
    # Decrypt the weights using PySyft
    decrypted_weights = [torch.tensor(w).get().float_precision() for w in client_weights]
    # Aggregate the decrypted weights
    aggregated_weights = aggregate_weights([decrypted_weights])
    # Set the aggregated weights to the global model
    for param, new_weight in zip(global_model.parameters(), aggregated_weights):
        param.data.copy_(new_weight)
    # Evaluate the global model
    loss, accuracy = evaluate_model(global_model)
    print(f"Round completed. Loss: {loss}, Accuracy: {accuracy}")
    # Plot loss and accuracy
    plt.plot(range(num_rounds), [loss], label='Loss')
    plt.plot(range(num_rounds), [accuracy], label='Accuracy')
    plt.legend()
    plt.show()
    # Send aggregated weights back to the client
    return jsonify({'aggregated_weights': json.dumps([w.tolist() for w in aggregated_weights])})

# Run the Flask app
if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)
