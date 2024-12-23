import numpy as np
import tritonclient.http as httpclient
# URL='192.168.0.81:8000'

# Define the server URL
TRITON_SERVER_URL = "192.168.0.81:8000"  # Change to your Triton server address if different
MODEL_NAME = "densenet_onnx"

# Create a Triton client
client = httpclient.InferenceServerClient(url=TRITON_SERVER_URL)

# Check server health
if not client.is_server_live():
    raise Exception("Triton server is not live. Ensure it's running.")

# Get model metadata
metadata = client.get_model_metadata(MODEL_NAME)
print(metadata)
