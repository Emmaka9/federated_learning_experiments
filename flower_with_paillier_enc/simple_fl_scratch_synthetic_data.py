# -*- coding: utf-8 -*-
"""simple_fl_scratch_synthetic_data.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/github/Emmaka9/federated_learning_experiments/blob/main/flower_with_paillier_enc/simple_fl_scratch_synthetic_data.ipynb

### Install packages
"""

# !pip install protobuf==3.20.3
# !pip install -U flwr["simulation"]

# !pip install phe

"""### Imports"""

# !python.__version__



from phe import paillier
import numpy as np

import flwr as fl
from flwr.common import Metrics
from typing import List, Tuple

import torch
from torch import nn


# DEVICE = (
#         "cuda"
# if torch.cuda.is_available()
# else "mps"
# if torch.backends.mps.is_available()
# else "cpu"
# )
DEVICE = torch.device("cpu")

"""### Set up Paillier Enc"""

def generate_keys():
    public_key, private_key = paillier.generate_paillier_keypair()
    return public_key, private_key

def encrypt_vector(public_key, array):
    print("=========Encrypting==========")
    flat_array = array.flatten()
    encrypted_flat_array = [public_key.encrypt(float(i)).ciphertext() for i in flat_array]
    encrypted_array = np.array(encrypted_flat_array).reshape(array.shape)
    print("++++++++++Encryption Successfull+++++++++++")
    return encrypted_array



def decrypt_vector(private_key, vector):
    return np.array([private_key.decrypt(v) for v in vector])

# Generate Paillier Keys
encryptor, decryptor = generate_keys()

encryptor, decryptor

"""### A simple model"""

# define a simple model
class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.linear = nn.Linear(10, 2)

    def forward(self, x):
        return self.linear(x)


model = SimpleModel()

"""### Data generation"""

def generate_data(samples = 100):
    X = np.random.randn(samples, 10).astype(np.float32) # 10 features
    y = (np.sum(X, axis=1)) + np.random.randn(samples).astype(np.float32) # sum of features + noise
    return X, y

"""### Training and Evaluation functions"""

def train(model, train_loader, criterion, optimizer, epochs):

    # certain types of layers that have different behavior during training vs testing (eval) such as dropout, batch_normalization layers
    model.train() # set the model to training mode
    total_loss = 0.0

    # training loop
    for epoch in range(epochs):
        for x, y in train_loader: # an iterable that provides batches of data. Each iteration yields a batch of inputs(x) and y.
            # reset the grads of all model params before loss calc.
            optimizer.zero_grad() # gradients are accumulated in buffers whenever .backward() is called.
            output = model(x) # forward computation defined in models forward function
            # .view(-1,1) reshape y to ensure it has the correct shape for the loss func. Converts y it to column vector.
            loss = criterion(output, y.view(-1, 1))
            # computes gradient of the loss wrt all model parameters (or any tensor with requires_grad=True)

            loss.backward() # optimizer in the next step will use these gradients to update the parameters.
            optimizer.step() # updates the model params. update rule - sgd, adam, ...
            total_loss += loss.item()
        print(f"Epoch {epoch+1}: train loss {total_loss}")
    return total_loss / len(train_loader)


def validate(model, val_loader, criterion):
    model.eval() # eval mode
    total_loss = 0.0

    with torch.no_grad():
        for x, y in val_loader:
            output = model(x)
            loss = criterion(output, y.view(-1, 1))
            total_loss += loss.item()
    return total_loss / len(val_loader)


def test(model, test_loader, criterion):
    return validate(model, test_loader, criterion)

"""### Flower Client and Server setup"""

from collections import OrderedDict

# Define flwr client

class Client(fl.client.NumPyClient):
    def __init__(self, model, train_loader, val_loader, test_loader):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.criterion = nn.MSELoss()
        self.optimizer = torch.optim.SGD(model.parameters(), lr = 0.01)


    def get_parameters(self, config) -> List[np.ndarray]:
        # retrieve the current state of the model's params from the client's local mdoel.
        '''
        1. extract params
        2. convert params from tensors to numpy arrays. Flwr operates with numpy arrays for parameter exchange
        to ensure framework agnosticism and to facilitate serialization
        3. return params - returns a list of numpy arrays, where each array correspond to params of a particular layer or part
        of the model

        Process:
        -> model.state_dict() - state_dict of a pytorch model is a python dict that maps each layer to its parameter tensor.
        state_dict().items() - returns a list of key-val pairs where keys are strings representing the names of the layers, and
        values are the parameters' tensors of those layers.
        -> val.cpu() move the val tensor to the cpu if it isn't already there. Tensors on GPU cannot be directly converted to
        numpy arrays, and operations involving numpy arrays typically require tensors to be on CPU.
        -> .numpy() converts pytorch tensors to numpy arrs. used for param serialization and are easy to handle across different
        computing environments.
        '''
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]

    def set_parameters(self, parameters: List[np.ndarray]) -> None:
        # update the local model with parameters received from the server
        # arg - params: parameter from the global model
        '''
        In this step, the aggregated updates from multiple clients are distributed back to each client
        1. receive params - takes a list of numpy arrays as input
        2. Update model - convert numpy arrays back to pytorch tensors and ensure each param tensor in the model
        is updated accordingly
        3. Synchronization - the update syncs the local model with the global model state as maintained by the server.
        '''

        params_dict = zip(self.model.state_dict().keys(), parameters) # tuples: each tuple contains a parameter name and corresponding
        # new parameter value (as a numpy arr)
        state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params_dict}) # new state dict
        self.model.load_state_dict(state_dict, strict = True) # update the model params with the new_state_dict. strict=True ensures
        # the keys in the state_dict match exactly with the keys in the model's current state_dict.


    def fit(self, parameters, config):
        '''
        '''
        self.set_parameters(parameters)
        loss = train(self.model, self.train_loader, self.criterion, self.optimizer, epochs=1)
        updated_params = self.get_parameters(self.model)
        encrypted_params = [encrypt_vector(encryptor, p) for p in updated_params]
        return updated_params, len(self.train_loader), {'loss' : loss}

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        loss = validate(self.model, self.val_loader, self.criterion)
        return loss, len(self.val_loader), {'loss': loss}

"""### Data Setup"""



"""### Client Configuration

Having 10 clients would mean having 10 instances of FlowerClient in memory.
Doing this on a single machine can quickly exhaust the available memory resources, even if only a subset of these clients
participates in a single round of federated learning.

In addition to the regular capabilities where server and clients run on multiple machines, Flower, therefore,
provides special simulation capabilities that create FlowerClient instances only when they are actually necessary for
training or evaluation. To enable the Flower framework to create clients when necessary, we need to implement a function
called client_fn that creates a FlowerClient instance on demand.
"""



def client_fn(cid: str) -> fl.client.Client:
    # Create a flwr client representing a single organization


    # load model
    model = SimpleModel().to(DEVICE)

    # Load data
    # each client gets a different trainloader/valloader, so each client
    # will train and evaluate on their own unique data
    X_train, y_train = generate_data(1000)
    X_val, y_val = generate_data(200)
    X_test, y_test = generate_data(200)

    train_loader = [(torch.tensor(X_train[i : i+32]), torch.tensor(y_train[i : i+32])) for i in range(0, len(y_train), 32)]
    val_loader = [(torch.tensor(X_val[i : i+32]), torch.tensor(y_val[i : i+32])) for i in range(0, len(y_val), 32)]
    test_loader = [(torch.tensor(X_test[i : i+32]), torch.tensor(y_test[i : i+32])) for i in range(0, len(y_test), 32)]

    # create a single flwr client representing a single org
    return Client(model, train_loader, val_loader, test_loader).to_client()

"""client_fn which allows Flower to create FlowerClient instances whenever it needs to call fit or evaluate on one particular client."""

# def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
#     # Multiply accuracy of each client by number of examples used
#     accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
#     examples = [num_examples for num_examples, _ in metrics]

#     # Aggregate and return custom metric (weighted average)
#     return {"accuracy": sum(accuracies) / sum(examples)}

# create fedavg strategy
strategy = fl.server.strategy.FedAvg(
    fraction_fit = 1.0, # sample 100% of available clients for training
    fraction_evaluate=0.5, # sample 50% of avaliable clients for evaluation
    min_fit_clients=10, # never sample less than 10 clients for training
    min_evaluate_clients=5, #never sample less than 5 clients for eval
    min_available_clients=10, # wait until all 10 clients are available
    #evaluate_metrics_aggregation_fn=weighted_average,  # <-- pass the metric aggregation function

)

# Specify the rosources each of your clients used. By default, each client will be allocated 1x cpu and 0x gpus
client_resources = {'num_cpus': 1, 'num_gpus': 0.0}
if DEVICE.type =='cuda':
    # here we are assigning an entire gpu for each client.
    client_resources = {'num_cpus': 1, 'num_gpus': 1.0}
    # see documentation for details


# Start simulation
fl.simulation.start_simulation(
    client_fn=client_fn,
    num_clients=10,
    config=fl.server.ServerConfig(num_rounds=5),
    strategy=strategy,
    client_resources=client_resources,
)

print("++++++++++Simulation Completed!!+++++++++++++++")
print("============Execution Success!!================")