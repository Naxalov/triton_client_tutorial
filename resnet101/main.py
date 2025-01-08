import numpy as np
import time
import tritonclient.http as httpclient
import cv2
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
input_shape = (112, 112,3)  # Batch size 1, RGB image
input_data = np.random.randint(0, 256, size=input_shape).astype(np.uint8)

input_mean = 127.5
input_std = 127.5

print(input_data.shape)

#  image processing using blob
blob = cv2.dnn.blobFromImage(
    input_data,
    scalefactor=1.0 / input_std,
    size=(112, 112),
    mean=(input_mean, input_mean, input_mean),
    swapRB=False,
)
input_data = blob.astype(np.float32) # Convert to float32 for Triton


# Define Triton input
inputs = [
    httpclient.InferInput(name="input.1", shape=input_data.shape, datatype="FP32")
]
inputs[0].set_data_from_numpy(input_data)

# Define Triton outputs
outputs = [
    httpclient.InferRequestedOutput(name="730"),  # Feature vector output
    httpclient.InferRequestedOutput(name="onnx::Div_729"),  # Additional scalar output
]

# Perform inference in a loop to check performance
num_iterations = 1000
total_time = 0

for idx, _ in enumerate(range(num_iterations)):
    start_time = time.time()
    response = client.infer(model_name=MODEL_NAME, inputs=inputs, outputs=outputs)
    end_time = time.time()
    total_time += end_time - start_time

    # Retrieve and process outputs
    feature_vector = response.as_numpy("730")
    scalar_output = response.as_numpy("onnx::Div_729")

    # Print results with iteration index
    print(f"Iteration {idx}:")
    # print(f"Feature Vector (Shape {feature_vector.shape}):\n{feature_vector}")
    # print(f"Scalar Output (Shape {scalar_output.shape}): {scalar_output[0, 0]}")

average_time = total_time / num_iterations
print(f"Average inference time over {num_iterations} iterations: {average_time:.6f} seconds")