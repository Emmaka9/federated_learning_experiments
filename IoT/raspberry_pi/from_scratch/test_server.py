

import paho.mqtt.client as mqtt

def on_connect(client, userdata, flags, rc):
    if rc == 0:
        print("Connected successfully")
        client.subscribe("test/topic")
    else:
        print("Connection failed with code", rc)

def on_message(client, userdata, msg):
    print(f"Message received: {msg.payload.decode()} on topic {msg.topic}")


broker_address = "10.0.0.249"  # Raspberry Pi's IP
port = 1883
# username = "yourusername"
# password = "yourpassword"

client = mqtt.Client("Windows_Subscriber")
# client.username_pw_set(username, password)
client.on_connect = on_connect
client.on_message = on_message

client.connect(broker_address, port=port)
client.loop_forever()
