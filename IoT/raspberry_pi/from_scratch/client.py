import paho.mqtt.client as mqtt
import json
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
#from generate_data import generate_data


# Generate synthetic data for a simple regression
np.random.seed(42)
x = np.random.rand(100, 1) * 10 # Features
y = 2 * x + 1 + np.random.randn(100, 1) * 2 # Targets with some noise

# convert to dataframe
data = pd.DataFrame(np.hstack((x,y)),  columns=['x', 'y'])
#data.to_csv('dataset.csv', index=False)
# Load data
#data = pd.read_csv('dataset.csv')
X = data[['x']]
y = data['y']

# initial model
model = LinearRegression()


def on_connect(client, userdata, flags, rc):
	print("Connected with result code"+str(rc))
	client.subscribe("federated/model")


def on_message(client, userdata, msg):
	global model
	updated_model = json.loads(msg.payload)
	print('Received updated model:', updated_model)
	
	# update model with received parameters
	model.coef_ = np.array([updated_model['coef']])
	model.intercept_ = updated_model['intercept']
	
	model.fit(X, y)
	# prepare new model params to send back
	model_params = {'coef' : model.coef_[0], 'intercept': model.intercept_}
	client.publish('federated/model_updates', json.dumps(model_params))	



client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION1, "Pi Client")
client.on_connect = on_connect
client.on_message = on_message

client.connect("10.0.0.163", 1883, 60)
client.loop_start() # start network loop

# initial training and sending of model_params
model.fit(X,y)
initial_params = {'coef' : model.coef_[0], 'intercept': intercept_}
client.publish('federated/model_updates', json.dumps(initial_params))

# keep the script alive to allow cont. training
import time
while True:
	time.sleep(1)
