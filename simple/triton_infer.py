import numpy as np
import tritonclient.http as httpclient
# URL='192.168.0.81:8000'

# Define the server URL
TRITON_SERVER_URL = "192.168.0.81:8000"  # Change to your Triton server address if different
MODEL_NAME = "simple"

# Create a Triton client
client = httpclient.InferenceServerClient(url=TRITON_SERVER_URL)

# Check server health
if not client.is_server_live():
    raise Exception("Triton server is not live. Ensure it's running.")

# Define input data
batch_size = 1  # Number of samples in a batch
input_shape = (batch_size, 16)

input0_data = np.random.randint(0, 100, size=input_shape, dtype=np.int32)
input1_data = np.random.randint(0, 100, size=input_shape, dtype=np.int32)

print(f"Input0: \n{input0_data}")
print(f"Input1: \n{input1_data}")

# Create input and output tensors
inputs = [
    httpclient.InferInput("INPUT0", input0_data.shape, "INT32"),
    httpclient.InferInput("INPUT1", input1_data.shape, "INT32"),
]

# Set input data
inputs[0].set_data_from_numpy(input0_data)
inputs[1].set_data_from_numpy(input1_data)

outputs = [
    httpclient.InferRequestedOutput("OUTPUT0"),
    httpclient.InferRequestedOutput("OUTPUT1"),
]

# Perform inference
response = client.infer(model_name=MODEL_NAME, inputs=inputs, outputs=outputs)

# Retrieve and display results
output0_data = response.as_numpy("OUTPUT0")
output1_data = response.as_numpy("OUTPUT1")

print(f"Output0 (Addition): \n{output0_data}")
print(f"Output1 (Subtraction): \n{output1_data}")
