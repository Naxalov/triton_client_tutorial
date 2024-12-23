import numpy as np
import tritonclient.http as http_client

URL='192.168.0.81:8000'

# Create a client
client = http_client.InferenceServerClient(URL)

# Check a Triton server health
health = client.is_server_ready()
if not health:
    raise Exception("Inference server is not ready")


# Get the list of models
models = client.get_model_repository_index()
print("Available Models:")
for model in models:
    print(f"- {model['name']} (Version: {model.get('version', 'latest')})")

# # Model and input/output names
# model_name = "simple"  # Replace with your actual model name
# input_name = "input_0"  # Replace with the actual input name from your model's config.pbtxt
# output_name = "output_0"  # Replace with the actual output name from your model's config.pbtxt


# # Create input data
# input_data = np.array([1, 2, 3, 4, 5], dtype=np.float32)

# # Convert input data to Triton inputs
# infer_input = http_client.InferInput(input_name, input_data.shape, 'FP32')
# infer_input.set_data_from_numpy(input_data)



# # Specify the requested output
# infer_output = http_client.InferRequestedOutput(output_name)

# # Run inference
# results = client.infer(model_name=model_name, inputs=[infer_input], outputs=[infer_output])

# # Get the output data
# output_data = results.as_numpy(output_name)
# print(output_data)