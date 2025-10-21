import napari
from skimage import io, img_as_float, exposure
import numpy as np

# Paths to TIFF files
plaque_tiff_path = '/Users/elinestas/Desktop/VAβs/Data/1. Raw/Session 1/223783 Ch1 12.07 S.tif'
vessel_tiff_path = '/Users/elinestas/Desktop/VAβs/Data/1. Raw/Session 1/223783 Ch2 12.07 S.tif'

# Load TIFF stacks
try:
    plaque_stack = io.imread(plaque_tiff_path)
    vessel_stack = io.imread(vessel_tiff_path)
except Exception as e:
    print(f"Error loading images: {e}")
    raise

# Convert images to float
plaque_stack = img_as_float(plaque_stack)
vessel_stack = img_as_float(vessel_stack)

# Parameters for visualization
z_spacing = 5  # Micrometers per slice
xy_spacing = 500 / 512  # Micrometers per pixel in x and y directions
max_depth = 950  # in micrometers
slice_950 = int(max_depth / z_spacing)  # Slice limit

# Find the last non-overexposed frame in the vessel stack
def find_last_non_overexposed(stack, overexposure_threshold=0.95):
    num_slices = stack.shape[0]
    for i in range(num_slices - 1, -1, -1):
        if np.mean(stack[i] >= 0.95) < overexposure_threshold:
            return i
    return num_slices - 1

last_non_overexposed_idx = find_last_non_overexposed(vessel_stack)

# Normalize vessel stack based on last non-overexposed frame
def normalize_slices_based_on_reference(stack, reference_frame_idx, lower=0.3, upper=100):
    p_lower, p_upper = np.percentile(stack[reference_frame_idx], (lower, upper))
    for i in range(reference_frame_idx, stack.shape[0]):
        stack[i] = exposure.rescale_intensity(stack[i], in_range=(p_lower, p_upper), out_range=(0, 1))
    return stack

vessel_stack = normalize_slices_based_on_reference(vessel_stack, last_non_overexposed_idx)

# Trim stacks to 850 µm depth
plaque_stack = plaque_stack[:slice_950, :, :]
vessel_stack = vessel_stack[:slice_950, :, :]

# Create a mask for slices 1-8 that contains only yellow
yellow_mask = np.zeros_like(plaque_stack)  # Blank mask
yellow_mask[:17] = plaque_stack[:17]  # Apply only to slices 1-8

# Remove slices 1-8 from plaque & vessel stacks (so they don't overlap with yellow)
plaque_stack[0:17] = 0  
vessel_stack[0:17] = 0  

from tifffile import imsave
import numpy as np

# Ensure input arrays are float32 and clipped
yellow = np.clip(yellow_mask, 0, 1).astype(np.float32)
cyan = np.clip(plaque_stack, 0, 1).astype(np.float32)
magenta = np.clip(vessel_stack, 0, 1).astype(np.float32)

# Stack as (Z, C, Y, X)
# Channel 0: yellow, Channel 1: cyan, Channel 2: magenta
composite_stack = np.stack([yellow, cyan, magenta], axis=1)

# Save
output_path = '/Users/elinestas/Desktop/composite_3channel.tif'
imsave(output_path, composite_stack, imagej=True)
print(f"Saved multi-channel grayscale TIFF to: {output_path}")



# Open Napari viewer
viewer = napari.Viewer(ndisplay=3)

# Add yellow-highlighted slices 1-8
viewer.add_image(yellow_mask, colormap='yellow', blending='additive', name='Highlighted Slices',
                 scale=(z_spacing, xy_spacing, xy_spacing))

# Add plaques as light blue (cyan) from slice 9 onwards
viewer.add_image(plaque_stack, colormap='cyan', blending='additive', name='Plaques',
                 scale=(z_spacing, xy_spacing, xy_spacing))

# Add vessels as magenta from slice 9 onwards
viewer.add_image(vessel_stack, colormap='magenta', blending='additive', name='Vessels',
                 scale=(z_spacing, xy_spacing, xy_spacing))

viewer.theme = 'light'
viewer.camera.angles = (0, 45, 180)
viewer.camera.zoom = 1.5
viewer.scale_bar.visible = True
viewer.scale_bar.unit = 'µm'

# Run Napari
napari.run()
