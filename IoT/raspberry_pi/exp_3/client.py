# imports
import torch
import os
import requests

# neural net defn
class SimpleNN(torch.nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc1 = torch.nn.Linear(10, 10)
        self.fc2 = torch.nn.Linear(10, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# generate synthetic dataset for training
def generate_synthetic_data(size=500):
    X = torch.randn(size, 10)
    y = torch.randn(size, 1)
    return X, y

# training the model for five epochs
def train_model(model, data, epochs=5):
    model.train()
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    X, y = data
    for epoch in range(epochs):
        optimizer.zero_grad()
        preds = model(X)
        loss = criterion(preds, y)
        loss.backward()
        optimizer.step()
        print(f'Epoch {epoch + 1}/ {epochs}, Loss: {loss.item()}')
    return model

# Send the trained model to the server
def send_model_to_server(model, server_url, client_id):
    model_path = f'client_{client_id}_model.pth'
    torch.save(model.state_dict(), model_path)
    with open(model_path, 'rb') as f:
        response = requests.post(f'{server_url}/send_model', json={'model_path': model_path})
        print(response.json())
    os.remove(model_path) # clean up the local model file

if __name__=="__main__":
    client_id = 1
    model = SimpleNN()

    # Train model with synthetic data
    training_data = generate_synthetic_data(size=500)
    trained_model = train_model(model, training_data, epochs=5)

    # Send trained model to server
    server_url = 'http://10.0.0.163:5000'
    send_model_to_server(trained_model, server_url, client_id)