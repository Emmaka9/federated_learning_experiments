from flask import Flask, request, jsonify
import torch
import torch.nn as nn
import numpy as np
import phe as paillier
import json
import matplotlib.pyplot as plt

# Initialize Flask app
app = Flask(__name__)

# Define a simple neural network model
class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(20, 50)
        self.fc2 = nn.Linear(50, 2)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Create a global model
global_model = SimpleNN()

# Paillier key generation
public_key, private_key = paillier.generate_paillier_keypair()

# Lists to store performance metrics
losses = []
accuracies = []
rounds = []

# Function to load validation data (replace with actual validation data loading logic)
def load_validation_data():
    return torch.tensor(np.random.rand(200, 20), dtype=torch.float32), torch.tensor(np.random.randint(2, size=200), dtype=torch.long)

# Function to load test data (replace with actual test data loading logic)
def load_test_data():
    return torch.tensor(np.random.rand(200, 20), dtype=torch.float32), torch.tensor(np.random.randint(2, size=200), dtype=torch.long)

# Function to aggregate encrypted weights from clients
def aggregate_weights(encrypted_weights):
    # Decrypt each weight using the private key
    decrypted_weights = [[private_key.decrypt(paillier.EncryptedNumber(public_key, *ew)) for ew in zip(*layer)] for layer in zip(*encrypted_weights)]
    # Average the decrypted weights
    aggregated_weights = [[sum(dw) / len(encrypted_weights) for dw in zip(*layer)] for layer in decrypted_weights]
    return aggregated_weights

# Function to evaluate the global model on data
def evaluate_model(model, data, labels):
    criterion = nn.CrossEntropyLoss()
    outputs = model(data)
    loss = criterion(outputs, labels).item()
    accuracy = (outputs.argmax(1) == labels).float().mean().item()
    return loss, accuracy

# Function to decrypt encrypted weights
def decrypt_weights(encrypted_weights, private_key):
    decrypted_weights = []
    for ew in encrypted_weights:
        decrypted_weights.append(private_key.decrypt(paillier.EncryptedNumber(public_key, ew[0], ew[1])))
    return decrypted_weights

@app.route("/send_initial_model", methods=["GET"])
def send_initial_model():
    # Send initial model weights to the client
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
        # Decrypt and aggregate weights received from the client
        decrypted_weights = [decrypt_weights(ew, private_key) for ew in encrypted_weights]
        aggregated_weights = aggregate_weights(decrypted_weights)

        # Update global model with aggregated weights
        with torch.no_grad():
            for param, new_weight in zip(global_model.parameters(), aggregated_weights):
                param.copy_(torch.tensor(new_weight, dtype=param.dtype))
        
        # Evaluate the updated global model on validation and test data
        val_data, val_labels = load_validation_data()
        val_loss, val_accuracy = evaluate_model(global_model, val_data, val_labels)
        
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
        aggregated_weights = [[w.detach().numpy().tolist() for w in aggregated_weights]]

        return jsonify({'aggregated_weights': aggregated_weights})
    except Exception as e:
        print(f"An error occurred: {e}")
        return jsonify({'error': str(e)}), 500

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)
