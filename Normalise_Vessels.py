import numpy as np
import tifffile as tiff
import os
from skimage.exposure import match_histograms

# Define the file paths for each week, region, and mouse
file_paths = {
    'V': {
        '58283': {
            1: '/Users/elinestas/Desktop/VAβs/Data/1. Raw/Session 1/58283 Ch2 11.07 V.tif',
            2: '/Users/elinestas/Desktop/VAβs/Data/1. Raw/Session 2/58283 Ch2 9.08 V.tif'
        },
    },
    'S': {
        '58283': {
            1: '/Users/elinestas/Desktop/VAβs/Data/1. Raw/Session 1/58283 Ch2 11.07 S.tif',
            2: '/Users/elinestas/Desktop/VAβs/Data/1. Raw/Session 2/58283 Ch2 9.08 S.tif'
        },
    },
}

# Define the starting slice for each stack
starting_slices = {
    'V': {
        '58283': {
            1: 0,  # Start from slice 10 in session 1 for mouse 223785 in region V
            2: 0   # Start from slice 5 in session 2 for mouse 223785 in region V
        },
    },
    'S': {
        '58283': {
            1: 4,  # Start from slice 12 in session 1 for mouse 223785 in region S
            2: 6   # Start from slice 8 in session 2 for mouse 223785 in region S
        },
    },
}

output_path = '/Users/elinestas/Desktop/VAβs/Data/2. Normalised/Vessels/{mouse_id}.{session_index}.{region}.tif'

def histogram_match_stack_with_mask(stack, reference_stack):
    """Perform histogram matching of a stack to a reference stack, preserving black background."""
    min_slices = min(stack.shape[0], reference_stack.shape[0])
    matched_stack = np.empty_like(stack[:min_slices])
    
    for i in range(min_slices):
        original_image = stack[i]
        matched_image = match_histograms(original_image, reference_stack[i])
        
        # Create a mask for the background (assume background is near zero)
        mask = original_image <= 0
        
        # Preserve the original background pixels
        matched_image[mask] = original_image[mask]
        
        matched_stack[i] = matched_image

    return matched_stack, stack[min_slices:]

def save_normalized_stack(normalized_stack, extra_stack, output_file):
    """Save the normalized stack and any extra slices to a .tif file."""
    if extra_stack.size > 0:
        full_stack = np.concatenate((normalized_stack, extra_stack), axis=0)
    else:
        full_stack = normalized_stack
        
    tiff.imwrite(output_file, full_stack.astype(np.float32))

# Function to process and normalize stacks for a specific region
def process_region_stacks(region, mice_data):
    for mouse_id, sessions in mice_data.items():
        # Load the reference stack (e.g., the first session as reference)
        reference_file_path = sessions[1]
        reference_stack = tiff.imread(reference_file_path)
        reference_stack = reference_stack[starting_slices[region][mouse_id][1]:]  # Apply starting slice

        for session_index, file_path in sessions.items():
            stack = tiff.imread(file_path)
            stack = stack[starting_slices[region][mouse_id][session_index]:]  # Apply starting slice
            
            # Perform histogram matching of the current stack to the reference stack
            matched_stack, extra_stack = histogram_match_stack_with_mask(stack, reference_stack)
            
            # Generate the output file path
            output_file = output_path.format(mouse_id=mouse_id, session_index=session_index, region=region)
            
            # Save the thresholded and matched stack
            save_normalized_stack(matched_stack, extra_stack, output_file)

# Normalize and save stacks for V and S separately
for region, mice_data in file_paths.items():
    process_region_stacks(region, mice_data)
