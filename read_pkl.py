#%%
import pickle
import numpy as np

# Replace with the path to your file, e.g., 'outputs/gr1_dishwasher_debug/example_test/stats.pkl'
file_path = '/home/lifan/Documents/GitHub/minbc/outputs/gr1_dishwasher_debug/example_test/norm.pkl' 

with open(file_path, 'rb') as f:
    data = pickle.load(f)

# Print the type of data (usually dict)
print(f"Data type: {type(data)}")

# If it is a dictionary, print keys and shapes of the values
if isinstance(data, dict):
    for key, value in data.items():
        print(f"\nKey: {key}")
        if isinstance(value, np.ndarray):
            print(f"  Shape: {value.shape}")
            print(f"  Content: {value}")
        elif isinstance(value, dict):
             print(f"  Keys: {value.keys()}") # Nested dictionary
        else:
            print(f"  Value: {value}")
else:
    print(data)