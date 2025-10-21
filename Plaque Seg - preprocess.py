from skimage import io, exposure
from skimage.registration import optical_flow_tvl1
from scipy.ndimage import map_coordinates, gaussian_filter
import numpy as np
import os
import h5py
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import filedialog

# GUI to select the input file
root = tk.Tk()
root.withdraw()  # Hide the root window

# Open file dialog to select the input TIFF file
input_tiff_path = filedialog.askopenfilename(title="Select the input TIFF file", filetypes=[("TIFF files", "*.tif *.tiff")])

# Load the single-channel TIFF file
plaque_stack = io.imread(input_tiff_path)

# Ensure the input TIFF is a 3D stack (frames, height, width)
if plaque_stack.ndim != 3:
    raise ValueError("Input TIFF should be a 3D stack (frames, height, width).")

output_plaque_h5_path = '/Users/elinestas/Desktop/VAβs/Data/3. Preprocessed/Plaques/{mouse_id}.{session_index}.{region}.h5'

# Adapt based on mouse_id, session_index, and region
mouse_id = 'Test'
session_index = '1'
region = '1'

# Normalize and apply CLAHE to each slice individually
saturated_prctile = [2, 98]
normalized_stack = np.zeros_like(plaque_stack, dtype=np.float32)
clahe_stack = np.zeros_like(normalized_stack, dtype=np.float32)

for i, image in enumerate(plaque_stack):
    # More aggressive normalization for the first few slices
    if i < 20:
        lower_bound = np.percentile(image, 1)
        upper_bound = np.percentile(image, 99)
    else:
        lower_bound = np.percentile(image, saturated_prctile[0])
        upper_bound = np.percentile(image, saturated_prctile[1])
    
    normalized_image = (image - lower_bound) / (upper_bound - lower_bound)
    normalized_image = np.clip(normalized_image, 0, 1)
    normalized_stack[i] = normalized_image

    # Apply CLAHE for better contrast adjustment with a lower clip limit
    clahe_image = exposure.equalize_adapthist(normalized_image, clip_limit=0.005)
    clahe_stack[i] = clahe_image

# Apply Gaussian filter for noise reduction
denoised_stack = np.zeros_like(clahe_stack, dtype=np.float32)
for i, image in enumerate(clahe_stack):
    denoised_image = gaussian_filter(image, sigma=1)
    denoised_stack[i] = denoised_image

# Display slices 10, 100, 150 from the denoised stack for verification before correcting motion artifacts
slices_to_check = [10, 80, 100]
plt.figure(figsize=(15, 5))
for i, slice_idx in enumerate(slices_to_check):
    plt.subplot(1, 3, i + 1)
    plt.imshow(denoised_stack[slice_idx], cmap='gray')
    plt.title(f'Slice {slice_idx}')
    plt.axis('off')
plt.show()

# Save the preprocessed plaque stack as HDF5
output_file_path = output_plaque_h5_path.format(mouse_id=mouse_id, session_index=session_index, region=region)
os.makedirs(os.path.dirname(output_file_path), exist_ok=True)
with h5py.File(output_file_path, 'w') as f:
    f.create_dataset('image_stack', data=denoised_stack)
print(f"Preprocessed data saved to {output_file_path}")
