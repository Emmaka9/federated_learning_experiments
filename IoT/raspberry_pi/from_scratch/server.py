import paho.mqtt.client as mqtt
import json


# Dummy model to start aggregation
aggregated_model = {'coef' : 0, 'intercept' : 0}

def on_connect(client, userdata, flags, rc):
    print('Connected with result code ' + str(rc))
    client.subscribe('federated/model_updates')

def on_message(client, userdata, msg):
    global aggregated_model
    model_update = json.loads(msg.payload)
    print('Received model update:', model_update)
    #client.publish('federated/model', json.dumps("aggregation begins"))
    #print('message sent = ' + 'aggregation')

    # Aggregation (average)
    aggregated_model['coef'] = (aggregated_model['coef'] + model_update['coef']) / 2
    aggregated_model['intercept'] = (aggregated_model['intercept'] + model_update['intercept']) / 2

    # send back the aggregated model
    client.publish('federated/model', json.dumps(aggregated_model))

client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION1 ,"Server")
client.on_connect = on_connect
client.on_message = on_message

client.connect('10.0.0.163', 1883, 60)
client.loop_forever()
