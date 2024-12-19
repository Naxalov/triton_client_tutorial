import numpy as np
import tritonclient.http as http_client

URL='192.168.0.81:8000'

# Create a client
client = http_client.InferenceServerClient(URL)

# Check a Triton server health
health = client.is_server_ready()
if not health.is_ready:
    raise Exception("Inference server is not ready")

# Run inference