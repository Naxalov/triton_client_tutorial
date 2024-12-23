import numpy as np
import tritonclient.http as httpclient
# URL='192.168.0.81:8000'

# Define the server URL
TRITON_SERVER_URL = "192.168.0.81:8000"  # Change to your Triton server address if different
MODEL_NAME = "resnet101"

# Create a Triton client
client = httpclient.InferenceServerClient(url=TRITON_SERVER_URL)

# Check server health
if not client.is_server_live():
    raise Exception("Triton server is not live. Ensure it's running.")

# Get model metadata
metadata = client.get_model_metadata(MODEL_NAME)
print(metadata)

# Generate dummy input data (1 sample, 3 channels, 112x112 image)
input_shape = (1, 3, 112, 112)  # Batch size 1, RGB image
input_data = np.random.rand(*input_shape).astype(np.float32)

# Preprocess input (e.g., normalization, if needed)
# Assuming standard normalization for images: [0, 1] range
input_data /= 255.0

# Define Triton input
inputs = [
    httpclient.InferInput(name="input.1", shape=input_shape, datatype="FP32")
]
inputs[0].set_data_from_numpy(input_data)

# Define Triton outputs
outputs = [
    httpclient.InferRequestedOutput(name="730"),  # Feature vector output
    httpclient.InferRequestedOutput(name="onnx::Div_729"),  # Additional scalar output
]

# Perform inference
response = client.infer(model_name=MODEL_NAME, inputs=inputs, outputs=outputs)

# Retrieve and process outputs
feature_vector = response.as_numpy("730")
scalar_output = response.as_numpy("onnx::Div_729")

# Print results
print(f"Feature Vector (Shape {feature_vector.shape}):\n{feature_vector}")
print(f"Scalar Output (Shape {scalar_output.shape}): {scalar_output[0, 0]}")