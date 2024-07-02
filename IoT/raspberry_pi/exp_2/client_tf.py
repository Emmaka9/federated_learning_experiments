import tensorflow as tf
import numpy as np
import pandas as pd
import requests

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

# Load data from CSV files
def load_data():
    data = pd.read_csv('synthetic_data.csv').values
    labels = pd.read_csv('synthetic_labels.csv').values.flatten()
    return data, labels

# Define a simple neural network model
def create_model():
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(50, activation='relu', input_shape=(20,)),
        tf.keras.layers.Dense(2, activation='softmax')
    ])
    model.compile(optimizer='sgd', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

# Train the model on local data
def train_model(model, data, labels, epochs=1):
    model.fit(data, labels, epochs=epochs)
    return model

# Send model weights to the server
def send_model_update(model, server_url):
    model_weights = model.get_weights()
    weights_json = tf.keras.models.save_model(model, 'temp_model.h5', save_format='h5')
    with open('temp_model.h5', 'rb') as f:
        response = requests.post(server_url + "/update_model", files={'model': f})
    return response.json()['aggregated_weights']

# Load and set model weights
def set_model_weights(model, weights):
    model.set_weights(weights)

# Main function to orchestrate training and communication
def main(server_url, num_rounds):
    data, labels = load_data()
    model = create_model()
    for round in range(num_rounds):
        model = train_model(model, data, labels, epochs=5)
        aggregated_weights = send_model_update(model, server_url)
        set_model_weights(model, aggregated_weights)

if __name__ == "__main__":
    server_url = "10.0.0.163:5000"
    num_rounds = 10
    main(server_url, num_rounds)
