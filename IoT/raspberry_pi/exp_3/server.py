import torch
from flask import Flask, request, jsonify, send_file
import copy
import matplotlib.pyplot as plt
import os


# Define a simple neural net
class SimpleNN(torch.nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc1 = torch.nn.Linear(10, 10)
        self.fc2 = torch.nn.Linear(10, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x
    
# Flask server
app = Flask(__name__)

# Initialize the global model
global_model = SimpleNN()
num_clients = 2
num_rounds = 10
client_model = []

# Synthetic dataset for server evaluation
def generate_synthetic_data(size=100):
    X = torch.randn(size, 10)
    y = torch.randn(size, 1)
    return X, y

def federated_averaging(client_models):
    global_weights = copy.deepcopy(client_models[0].state_dict())
    for k in global_weights.keys():
        for i in range(1, len(client_models)):
            global_weights[k] += client_models[i].state_dict()[k]
        global_weights[k] = torch.div(global_weights[k], len(client_models))
    return global_weights


def evaluate_model(model, data):
    model.eval() # eval mode
    with torch.no_grad():
        X, y = data
        preds = model(X)
        loss = torch.nn.MSELoss()(preds, y)
        print(f'Evaluation loss: {loss.item()}')
    return loss.item()

# Aggregation and Evaluation logic
@app.route('/send_model', methods=['POST'])
def receive_model():
    global client_models
    model_data = request.json['model']

    # Convert received model parameters back into the model format
    client_model = SimpleNN()
    client_model.load_state_dict(torch.load(model_data['model_path']))
    client_models.append(client_model)

    # Aggregate models once all clients have sent their models
    if len(client_models) == num_clients:
        aggregated_weights = federated_averaging(client_models)
        global_model.load_state_dict(aggregated_weights)

        # Evaluate model after aggregation
        print('Aggregated model performace: ')
        test_data = generate_synthetic_data(size=100)
        loss = evaluate_model(global_model, test_data)

        # Save the aggregated model
        global_model_path = 'global_model.pth'
        torch.save(global_model.state_dict(), global_model_path)

        client_model = [] # Reset for the next round
        return jsonify({'status': 'aggregated', 'loss': loss})
    
    return jsonify({'status': 'waiting'})


# Endpoint for clients to fetch the global model
@app.route('/get_global_model', methods=['GET'])
def send_global_model():
    global_model_path = 'global_model.pth'
    if os.path.exists(global_model_path):
        return send_file(global_model_path, as_attachment=True)
    else:
        return jsonify({'status': 'no_model_available'}), 404
    
    

if __name__=='__main__':
    app.run(host='0.0.0.0', port=5000)
