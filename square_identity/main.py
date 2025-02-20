import tritonclient.grpc as grpcclient
import numpy as np

# Define the server URL
TRITON_SERVER_URL = "192.168.0.66:8001"  # gRPC port is usually 8001
MODEL_NAME = "square_identity"

# Create a Triton client
client = grpcclient.InferenceServerClient(url=TRITON_SERVER_URL)

# Generate dummy input data (1 sample, 3 channels, 112x112 image)
input_shape = (480, 640, 3)  # Batch size 1, RGB image
input_data = np.random.randint(0, 256, size=input_shape).astype(np.uint8)
input_data = np.expand_dims(input_data, axis=0)  # Add batch dimension

# Convert input data to Triton inputs
inputs = [grpcclient.InferInput("INPUT0", input_data.shape, "UINT8")]
inputs[0].set_data_from_numpy(input_data)

# Define Triton output
outputs = [grpcclient.InferRequestedOutput("OUTPUT0")]

# Run inference
response = client.infer(model_name=MODEL_NAME, inputs=inputs, outputs=outputs)

# Retrieve and process output
output_data = response.as_numpy("OUTPUT0")
print(f'Model input: {input_data[0, 0, :]}')
print(f'Model output: {output_data[0, 0, :]}')
