from flask import Flask, request, jsonify
import torch
import torch.nn as nn
import numpy as np
import phe as paillier
import json
import pickle
import base64
import torch.optim as optim
import matplotlib.pyplot as plt

# Initialize Flask app
app = Flask(__name__)

# Define a simple neural network model
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(2048, 50)
        self.fc2 = nn.Linear(50, 2)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Create a global model using transfer learning
pretrained_model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet50', pretrained=True)
global_model = Net()

# Remove the final layer of the pretrained model and add our custom layers
pretrained_model.fc = nn.Sequential(
    nn.Linear(pretrained_model.fc.in_features, 2048),
    nn.ReLU(),
    nn.Linear(2048, 50),
    nn.ReLU(),
    nn.Linear(50, 2)
)

# Use the pretrained model to initialize our global model
global_model = pretrained_model

# Paillier key generation
public_key, private_key = paillier.generate_paillier_keypair()

# Lists to store performance metrics
losses = []
accuracies = []
rounds = []

# Load validation data (replace with actual validation data loading logic)
def load_validation_data():
    return torch.tensor(np.random.rand(200, 2048), dtype=torch.float32), torch.tensor(np.random.randint(2, size=200), dtype=torch.long)

# Load test data (replace with actual test data loading logic)
def load_test_data():
    return torch.tensor(np.random.rand(200, 2048), dtype=torch.float32), torch.tensor(np.random.randint(2, size=200), dtype=torch.long)

# Aggregate weights from clients
def aggregate_weights(weights):
    num_clients = len(weights)
    new_weights = [sum(weight) / num_clients for weight in zip(*weights)]
    return new_weights

# Evaluate the global model on data
def evaluate_model(model, data, labels):
    criterion = nn.CrossEntropyLoss()
    outputs = model(data)
    loss = criterion(outputs, labels).item()
    accuracy = (outputs.argmax(1) == labels).float().mean().item()
    return loss, accuracy

def decrypt_weights(encrypted_weights, private_key):
    decrypted_weights = []
    for ew in encrypted_weights:
        ew = json.loads(ew)
        decrypted_weights.append(private_key.decrypt(paillier.EncryptedNumber(public_key, ew[0], ew[1])))
    return decrypted_weights

@app.route("/send_initial_model", methods=["GET"])
def send_initial_model():
    model_weights = [param.data.numpy().tolist() for param in global_model.parameters()]
    return jsonify({'weights': model_weights})

@app.route("/update_model", methods=["POST"])
def update_model():
    global global_model, losses, accuracies, rounds
    try:
        data = request.get_json()
        if not data or 'weights' not in data:
            return jsonify({'error': 'Invalid data received'}), 400

        encrypted_weights = data['weights']
        client_weights = decrypt_weights(encrypted_weights, private_key)
        aggregated_weights = aggregate_weights([list(global_model.parameters()), client_weights])
        with torch.no_grad():
            for param, new_weight in zip(global_model.parameters(), aggregated_weights):
                param.copy_(torch.tensor(new_weight, dtype=param.dtype))
        
        # Load validation data and evaluate the model
        val_data, val_labels = load_validation_data()
        val_loss, val_accuracy = evaluate_model(global_model, val_data, val_labels)
        
        # Load test data and evaluate the model
        test_data, test_labels = load_test_data()
        test_loss, test_accuracy = evaluate_model(global_model, test_data, test_labels)

        # Store the metrics
        losses.append(test_loss)
        accuracies.append(test_accuracy)
        rounds.append(len(rounds) + 1)

        # Print the metrics
        round_num = len(rounds)
        print(f"Round {round_num} completed. Validation Loss: {val_loss}, Validation Accuracy: {val_accuracy}")
        print(f"Test Loss: {test_loss}, Test Accuracy: {test_accuracy}")

        # Plot the performance metrics and save to a file
        plt.figure()
        plt.plot(rounds, losses, label='Test Loss')
        plt.plot(rounds, accuracies, label='Test Accuracy')
        plt.xlabel('Round')
        plt.ylabel('Metric')
        plt.legend()
        plt.title('Model Performance over Rounds')
        plt.savefig(f'performance_metrics_round_{round_num}.png')
        plt.close()

        # Detach the tensors and convert to numpy arrays
        aggregated_weights = [w.detach().numpy().tolist() for w in aggregated_weights]
        
        return jsonify({'aggregated_weights': aggregated_weights})
    except Exception as e:
        print(f"An error occurred: {e}")
        return jsonify({'error': str(e)}), 500

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)