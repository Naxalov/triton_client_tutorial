import requests
import time

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
    

# Example usage
input_data = {
    "inputs": [
        {
            "name": "input.1",
            "shape": [1, 3, 112, 112],
            "datatype": "FP32",
            "data": [0] * (1 * 3 * 112 * 112)  # Replace with actual input data
        }
    ]
}


# Perform inference in a loop to check performance
num_iterations = 1000
total_time = 0
try:
    for idx, _ in enumerate(range(1000)):
        start_time = time.time()
        result = infer(input_data)
        end_time = time.time()
        total_time += (end_time - start_time)
        print(idx)
    
except Exception as e:
    print(e)



average_time = total_time / num_iterations
print(f"Average inference time over {num_iterations} iterations: {average_time:.6f} seconds")