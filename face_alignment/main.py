import numpy as np
import tritonclient.http as httpclient

# Define the server URL
TRITON_SERVER_URL = "192.168.0.81:8000"

# Define the model name
MODEL_NAME = "face_alignment" 

# Define the input shape
INPUT_SHAPE = 4



# Create a Triton client
client = httpclient.InferenceServerClient(url=TRITON_SERVER_URL)

# Generate dummy input data 
input0_data = np.random.rand(INPUT_SHAPE).astype(np.float32)
input1_data = np.random.rand(INPUT_SHAPE).astype(np.float32)

# Define Triton input
inputs = [
    httpclient.InferInput(name="INPUT0", shape=input0_data.shape, datatype="FP32"),
    httpclient.InferInput(name="INPUT1", shape=input1_data.shape, datatype="FP32"),
    
]

# Attach the data to the inputs
inputs[0].set_data_from_numpy(input0_data)
inputs[1].set_data_from_numpy(input1_data)

# Define Triton output

outputs = [
    httpclient.InferRequestedOutput(name="OUTPUT0"),
    httpclient.InferRequestedOutput(name="OUTPUT1"),
]

# Run inference
response = client.infer(model_name=MODEL_NAME, inputs=inputs, outputs=outputs)

# Retrieve and process output
output0_data = response.as_numpy("OUTPUT0")
output1_data = response.as_numpy("OUTPUT1")
print(f"Output0: \n{output0_data}")
print(f"Output1: \n{output1_data}")