import numpy as np
import tritonclient.http as httpclient
import cv2



def preprocess_image(image_path, input_shape):
    """
    Load and preprocess the image to match the model's input requirements.
    
    Parameters:
    - image_path (str): Path to the input image.
    - input_shape (tuple): Expected input shape (H, W, C).
    
    Returns:
    - np.ndarray: Preprocessed image ready for inference.
    """
    # Load image using OpenCV
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Image not found or unable to read at path: {image_path}")
    
    # Convert BGR to RGB
    # image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Resize image to match model's expected input size
    resized_image = cv2.resize(image, (input_shape[1], input_shape[0]))
    
   
    
    return resized_image

# Define the server URL
TRITON_SERVER_URL = "213.230.96.104:6000"

# Define the model name
MODEL_NAME = "face_alignment" 


# Define input and output tensor names based on config.pbtxt
input_name = "INPUT_IMAGE"
output_name = "ALIGNED_FACE"

# Define input shape based on config.pbtxt (H, W, C)
input_shape = (320, 320, 3)  # Example; adjust if different

IMAGE_PATH = '/Users/naxalov/github/cradle/triton_client_tutorial/naxalov.JPG'

# Preprocess the image
input_data = preprocess_image(IMAGE_PATH, input_shape)

# Create a Triton client
client = httpclient.InferenceServerClient(url=TRITON_SERVER_URL)


# Define Triton input
inputs = httpclient.InferInput(input_name, input_data.shape, "UINT8")


# Attach the data to the inputs
inputs.set_data_from_numpy(input_data)

# Define Triton output
outputs = httpclient.InferRequestedOutput(output_name)
# Run inference
response = client.infer(model_name=MODEL_NAME, inputs=[inputs], outputs=[outputs])

# Retrieve and process output
output_data = response.as_numpy(output_name)
print(f"Output: \n{output_data.shape}")
# Save the image to file
cv2.imwrite("aligned_face.jpg", output_data)
