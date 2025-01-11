import requests
import time
import numpy as np
import cv2

SERVER_ADDRESS = "192.168.0.81:8000"  # Change to your server address
MODEL_NAME = "resnet101"  # Change to your model name

url = f"http://{SERVER_ADDRESS}/v2/models/{MODEL_NAME}/infer"

def infer(input_data):
    # Send the input data to the server
    response = requests.post(url, json=input_data)
    
    # Check if the request was successful
    if response.status_code == 200:
        return response.json()
    else:
        raise Exception(f"Error: {response.status_code} - {response.text}")
 
 

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

# Example usage
input_data = {
    "inputs": [
        {
            "name": "input.1",
            "shape": [1, 3, 112, 112],
            "datatype": "FP32",
            "data": input_data.tolist()
        }
    ]
}


result = infer(input_data)

print(result)

# # Perform inference in a loop to check performance
# num_iterations = 1000
# total_time = 0
# try:
#     for idx, _ in enumerate(range(1000)):
#         start_time = time.time()
#         result = infer(input_data)
#         end_time = time.time()
#         total_time += (end_time - start_time)
#         print(idx)
    
# except Exception as e:
#     print(e)



# average_time = total_time / num_iterations
# print(f"Average inference time over {num_iterations} iterations: {average_time:.6f} seconds")