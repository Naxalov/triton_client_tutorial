import numpy as np
import tritonclient.http as httpclient
from PIL import Image
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

# Read from the image file
image_path = 'car.jpg'
input_shape = (1, 3, 224, 224)
# Load the image
image = Image.open(image_path).convert("RGB")
# Resize the image to the desired input shape
image = image.resize((input_shape[2], input_shape[3]))
# Convert the image to a numpy array
image = np.array(image).astype(np.float32)
# Transpose the image to the correct format
image = image.transpose(2, 0, 1)
# Add a batch dimension
image = np.expand_dims(image, axis=0)

# Preprocess input (e.g., normalization, if needed)
# Assuming standard normalization for images: [0, 1] range
image /= 255.0

# Define Triton input
inputs = [
    httpclient.InferInput(name="data_0", shape=input_shape, datatype="FP32")
]
inputs[0].set_data_from_numpy(image)

# Define Triton output
outputs = [
    httpclient.InferRequestedOutput(name="fc6_1")
]

# Perform inference
response = client.infer(model_name=MODEL_NAME, inputs=inputs, outputs=outputs)

# Retrieve and process output
output_data = response.as_numpy("fc6_1")
print(f"Model Output (logits):\n{output_data}")

# Post-process output (e.g., softmax and top predictions)
# Convert to 1D array for processing
output_data = output_data.flatten()
softmax = np.exp(output_data) / np.sum(np.exp(output_data))

# Get top-5 predictions
top_5_indices = np.argsort(softmax)[-5:][::-1]
print("\nTop-5 Predictions:")
for i in top_5_indices:
    print(f"Class {i}, Probability: {softmax[i]:.4f}")