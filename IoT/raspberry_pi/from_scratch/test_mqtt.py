import paho.mqtt.client as mqtt

# MQTT parameters
broker_address = "localhost"  # Broker is on the same machine
port = 1883
topic = "test/topic"
# username = "yourusername"
# password = "yourpassword"

# Create a client instance and specify the callback API version
client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION1, "Pi_Publisher")
# client.username_pw_set(username, password)

# Connect to the broker
client.connect(broker_address, port=port)

# Publish a message
client.publish(topic, "Hello from Raspberry Pi!")
print(f"Message sent to {topic}")

