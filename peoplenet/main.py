import tritonclient.grpc as grpcclient
import numpy as np

# Define the server URL
TRITON_SERVER_URL = "192.168.0.66:8001"  # gRPC port is usually 8001

# Model details
MODEL_NAME = "peoplenet"
INPUT_NAME = "input_1:0"
OUTPUT_NAMES = ["output_bbox/BiasAdd:0", "output_cov/Sigmoid:0"]
INPUT_SHAPE = (3, 544, 960)
INPUT_DTYPE = np.float32

# Create a Triton client
client = grpcclient.InferenceServerClient(url=TRITON_SERVER_URL)

# Generate dummy input data (1 sample, 3 channels, 544x960 image)
# Notice we first create data in (H, W, C) then transpose to (C, H, W).
input_shape_hw = (544, 960, 3)  # H, W, C
input_data = np.random.randint(0, 256, size=input_shape_hw).astype(np.float32)

# Transpose to (C, H, W) => (3, 544, 960)
input_data = np.transpose(input_data, (2, 0, 1))

# Now add the batch dimension => (1, 3, 544, 960)
input_data = np.expand_dims(input_data, axis=0)

# Convert input data to Triton inputs
inputs = [grpcclient.InferInput(INPUT_NAME, input_data.shape, "FP32")]
inputs[0].set_data_from_numpy(input_data)

# Define Triton outputs
outputs = [
    grpcclient.InferRequestedOutput("output_bbox/BiasAdd:0"),
    grpcclient.InferRequestedOutput("output_cov/Sigmoid:0")
]

# Run inference
response = client.infer(model_name=MODEL_NAME, inputs=inputs, outputs=outputs)

# Retrieve and process output
output_bbox = response.as_numpy("output_bbox/BiasAdd:0")
output_cov  = response.as_numpy("output_cov/Sigmoid:0")

print(f"Output bbox shape: {output_bbox.shape}")
print(f"Output cov shape:  {output_cov.shape}")
