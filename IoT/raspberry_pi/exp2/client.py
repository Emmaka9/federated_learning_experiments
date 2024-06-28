import tensorflow as tf
import numpy as np
import pandas as pd
import requests
import json

# Function to load synthetic data
def load_data():
    # Load data from CSV files
    data = pd.read_csv('synthetic_data.csv').values
    labels = pd.read_csv('synthetic_labels.csv').values.flatten()
    return data, labels

# Function to train the model
def train_model(model, data, labels, epochs=1):
    # Train the model with the provided data and labels
    model.fit(data, labels, epochs=epochs)
    return model

# Function to send the model update to the server
def send_model_update(model, server_url):
    # Get the model weights
    model_weights = model.get_weights()
    # Convert weights to JSON format
    weights_json = json.dumps([w.tolist() for w in model_weights])
    # Send weights to the server
    response = requests.post(server_url + "/update_model", json={'weights': weights_json})
    # Get aggregated weights from server response
    return response.json()['aggregated_weights']

# Function to set the model weights
def set_model_weights(model, weights):
    # Set the model weights with the received aggregated weights
    model.set_weights([np.array(w) for w in weights])

# Main function to orchestrate the training and communication
def main(server_url, num_rounds):
    # Load data
    data, labels = load_data()
    # Load pre-trained model
    model = tf.keras.models.load_model('pretrained_model.h5')
    
    for round in range(num_rounds):
        # Train model on local data
        model = train_model(model, data, labels, epochs=5)
        # Send model updates to server and receive aggregated weights
        aggregated_weights = send_model_update(model, server_url)
        # Set the received aggregated weights to the model
        set_model_weights(model, aggregated_weights)

# Entry point of the script
if __name__ == "__main__":
    # Server URL
    server_url = "http://<server-ip>:5000"
    # Number of rounds for training and aggregation
    num_rounds = 10
    # Execute main function
    main(server_url, num_rounds)
