import torch
import torch.nn as nn
import torch.optim as optim
import requests
import os

# Define a simple neural network model
class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(10, 10)
        self.fc2 = nn.Linear(10, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Synthetic dataset for training
def generate_synthetic_data(size=100):
    X = torch.randn(size, 10)
    y = torch.randn(size, 1)
    return X, y

# Training the model for 5 epochs
def train_model(model, data, epochs=5):
    model.train()
    criterion = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01)
    X, y = data
    for epoch in range(epochs):
        optimizer.zero_grad()
        outputs = model(X)
        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()
        print(f'Epoch {epoch + 1}/{epochs}, Loss: {loss.item()}')
    return model

# Fetch the global model from the server
def get_global_model(server_url):
    response = requests.get(f'{server_url}/get_global_model', stream=True)
    if response.status_code == 200:
        with open('global_model.pth', 'wb') as f:
            for chunk in response.iter_content(chunk_size=1024):
                f.write(chunk)
        print("Global model downloaded.")
        return True
    else:
        print("No model available on the server.")
        return False

# Load global model into the local model
def load_global_model(model):
    model.load_state_dict(torch.load('global_model.pth'))

# Send the trained model to the server
def send_model_to_server(model, server_url, client_id):
    model_path = f'client_{client_id}_model.pth'
    torch.save(model.state_dict(), model_path)
    
    with open(model_path, 'rb') as f:
        response = requests.post(f'{server_url}/send_model', files={'file': f})
        
        if response.status_code != 200:
            print(f"Error: Received status code {response.status_code}")
            print(f"Response content: {response.text}")
        else:
            try:
                print(response.json())
            except requests.exceptions.JSONDecodeError:
                print("Error: Unable to parse JSON response")
                print(f"Response content: {response.text}")
    
    os.remove(model_path)

def main():

    client_id = 1  # Adjust client_id for each Raspberry Pi client
    model = SimpleNN()
    
    # Fetch and load the global model from the server
    server_url = 'http://10.0.0.163:5000'  # Replace with actual server IP
    if get_global_model(server_url):
        load_global_model(model)
    
    # Train model with synthetic data
    train_data = generate_synthetic_data(size=100)
    trained_model = train_model(model, train_data, epochs=5)
    
    # Send trained model to server
    send_model_to_server(trained_model, server_url, client_id)

    



if __name__ == "__main__":
    num_rounds = 10
    for i in len(num_rounds):
        main()