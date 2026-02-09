#%%
import numpy as np

# Load the array from the file
data = np.load('/home/lifan/Documents/GitHub/minbc/data/train/470-560-300-100/index_pad_flow/index_pad_000000.npy')[::5,::5]

# Display basic information about the loaded array
print(data.shape)
print(data.dtype)
print(data)